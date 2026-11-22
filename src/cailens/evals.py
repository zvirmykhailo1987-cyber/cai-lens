# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Evals: run the pipeline on a split of synthetic forms and compare every field with the answer key.

The numbers that matter for an insurer are not only accuracy: how many documents go straight through
(«automatico»), and how many of THOSE carry a wrong key field - the error a person would never see.
Deadlines are scored for being correct and for being safe (never later than the true legal term).
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from cailens import form, formats, llm
from cailens.agents import rules
from cailens.paths import HOME
from cailens.pipeline import process
from cailens.schema import CaseTruth, Result
from cailens.synth import generate, seeds
from cailens.truth import AMBIGUOUS, expected

RESULTS = HOME / "evals" / "results"
LLM_STEPS = ("handwriting", "sketch", "verifier")
DEFECT_SIGNAL = {  # how a defect should show up in the result
    "firma_B_mancante": ("issue", "firma_mancante"), "firma_A_mancante": ("issue", "firma_mancante"),
    "totale_errato": ("issue", "totale_"), "campo_B_mancante": ("issue", "dati_B_mancanti"),
    "data_successiva": ("issue", "data_successiva"), "cf_errato": ("issue", "cf_non_valido"),
    "denuncia_unilaterale": ("issue", "denuncia_unilaterale"), "correzione": ("route", "revisione"),
}


def true_deadline(t: CaseTruth) -> int:
    if t.feriti:
        return 90
    return 30 if t.A.firma and t.B.firma else 60


def same(fid: str, want: str | None, got: str | None) -> bool:
    return formats.compact(fid, want) == formats.compact(fid, got)


def score_case(t: CaseTruth, r: Result) -> dict[str, Any]:
    exp = expected(t)
    row: dict[str, Any] = {"case_id": t.case_id, "capture": t.capture, "defects": t.defects, "route": r.route,
                           "registered": r.registered, "boxes": Counter(), "text": {}, "wrong_critical": []}
    for fid in form.CHECKBOX_FIELDS:
        want, got = exp[fid], (r.readings.get(fid).value if fid in r.readings else None)  # type: ignore[union-attr]
        row["boxes"][f"{want}->{got}"] += 1
    row["signatures"] = {s: (exp[f"{s}.firma"], r.readings[f"{s}.firma"].value if f"{s}.firma" in r.readings else None)
                         for s in form.SIDES}
    for fid in form.TEXT_FIELDS:
        rd = r.readings.get(fid)
        if rd is None or rd.agent != "handwriting_vlm" or rd.evidence.get("errore"):
            continue
        row["text"][fid] = {"ok": same(fid, exp[fid], rd.value), "conf": rd.confidence,
                            "want": exp[fid], "got": rd.value}
    for fid in rules.CRITICAL_TEXT:
        cell = row["text"].get(fid)
        if cell and not cell["ok"]:
            row["wrong_critical"].append(fid)
    for s in form.SIDES:
        want, got = exp[f"{s}.firma"], row["signatures"][s][1]
        if got not in (want, "incerto"):
            row["wrong_critical"].append(f"{s}.firma")
    for fid in form.CHECKBOX_FIELDS:
        got = r.readings[fid].value if fid in r.readings else None
        if exp[fid] != AMBIGUOUS and got not in (exp[fid], "incerto"):
            row["wrong_critical"].append(fid)
    row["deadline"] = {"true": true_deadline(t), "got": r.deadline.days, "provisional": r.deadline.provisional}
    codes = {i.code for i in r.issues}
    row["defect_hits"] = {}
    for d in t.defects:
        if d == "totale_errato" and all(p.totale is None or p.totale == str(len(p.circostanze)) for p in (t.A, t.B)):
            continue  # overridden by «totale_mancante» on the same side: there is no wrong total to find
        kind, sig = DEFECT_SIGNAL.get(d, (None, None))
        if kind == "issue" and sig is not None:
            row["defect_hits"][d] = any(sig in c for c in codes)
        elif kind == "route":
            row["defect_hits"][d] = r.route == sig
    row["scenario"] = (t.scenario, r.scenario)
    row["impact"] = {s: (exp[f"{s}.impatto"], r.readings[f"{s}.impatto"].value if f"{s}.impatto" in r.readings else None)
                     for s in form.SIDES}
    row["cost_usd"] = sum(s.cost_usd for s in r.steps)
    row["ms"] = sum(s.ms for s in r.steps)
    row["llm_ms"] = sum(s.ms for s in r.steps if s.step in LLM_STEPS)
    row["cached"] = all(s.note.startswith("cache") for s in r.steps if s.step in LLM_STEPS and s.ok)
    row["llm_ok"] = all(s.ok for s in r.steps if s.step in LLM_STEPS)
    row["verified"] = any(s.step == "verifier" for s in r.steps)
    row["auto_but_wrong"] = r.route == "automatico" and bool(row["wrong_critical"])
    return row


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    boxes: Counter[str] = Counter()
    for r in rows:
        boxes.update(r["boxes"])

    def b(want: str, got: str) -> int:
        return boxes.get(f"{want}->{got}", 0)

    marked, empty, amb = (sum(v for k, v in boxes.items() if k.startswith(f"{w}->")) for w in ("si", "no", AMBIGUOUS))
    sig = Counter(f"{w}->{g}" for r in rows for w, g in r["signatures"].values())
    per_field: dict[str, list[bool]] = defaultdict(list)
    auto_field: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for fid, cell in r["text"].items():
            per_field[fid].append(cell["ok"])
            if cell["conf"] >= rules.TEXT_MIN:
                auto_field[fid].append(cell["ok"])
    text_total = sum(len(v) for v in per_field.values())
    text_ok = sum(sum(v) for v in per_field.values())
    dl = [r["deadline"] for r in rows if r["deadline"]["got"] is not None]
    defects: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for d, hit in r["defect_hits"].items():
            defects[d].append(hit)
    scen = [r["scenario"] for r in rows if r["scenario"][1] is not None]
    imp = [pair for r in rows for pair in r["impact"].values() if pair[0] and pair[1] is not None]
    auto = [r for r in rows if r["route"] == "automatico"]
    live = [r for r in rows if r["llm_ok"]]  # cassette answers keep the cost and latency of the original call
    return {
        "documents": n,
        "registered_pct": _pct(sum(r["registered"] for r in rows), n),
        "checkbox": {
            "marked": marked, "empty": empty, "scratched_out": amb,
            "marked_read_si_pct": _pct(b("si", "si"), marked), "marked_read_no": b("si", "no"),
            "empty_read_no_pct": _pct(b("no", "no"), empty), "empty_read_si": b("no", "si"),
            "sent_to_review": b("si", "incerto") + b("no", "incerto"),
            "scratched_sent_to_review_pct": _pct(b(AMBIGUOUS, "incerto"), amb),
        },
        "signature": {k: v for k, v in sorted(sig.items())},
        "text": {
            "fields_scored": text_total, "exact_pct": _pct(text_ok, text_total),
            "per_field_pct": {f: _pct(sum(v), len(v)) for f, v in sorted(per_field.items())},
            "high_confidence_exact_pct": _pct(sum(sum(v) for v in auto_field.values()),
                                              sum(len(v) for v in auto_field.values())),
        },
        "deadline": {
            "scored": len(dl), "correct_pct": _pct(sum(d["true"] == d["got"] for d in dl), len(dl)),
            "never_later_pct": _pct(sum(d["got"] <= d["true"] for d in dl), len(dl)),
            "provisional": sum(d["provisional"] for d in dl),
        },
        "defects_flagged_pct": {d: _pct(sum(v), len(v)) for d, v in sorted(defects.items())},
        "defects_n": {d: len(v) for d, v in sorted(defects.items())},
        "scenario_pct": _pct(sum(a == g for a, g in scen), len(scen)),
        "impact_pct": _pct(sum(a == g for a, g in imp), len(imp)),
        "routing": {"automatico_pct": _pct(len(auto), n), "automatico": len(auto),
                    "automatico_with_wrong_key_field": sum(r["auto_but_wrong"] for r in rows),
                    "wrong_fields": Counter(f for r in rows if r["auto_but_wrong"] for f in r["wrong_critical"])},
        "cost": {
            "usd_per_doc_live": round(statistics.mean(r["cost_usd"] for r in live), 5) if live else None,
            "llm_seconds_per_doc_live": round(statistics.mean(r["llm_ms"] for r in live) / 1000, 2) if live else None,
            "cv_ms_per_doc": round(statistics.median(r["ms"] - r["llm_ms"] for r in rows)) if rows else None,
            "usd_total_recorded": round(sum(r["cost_usd"] for r in rows), 4),
        },
        "by_capture": {c: {"documents": len(g), "automatico_pct": _pct(sum(r["route"] == "automatico" for r in g), len(g))}
                       for c in ("scansione", "foto") if (g := [r for r in rows if r["capture"] == c])},
    }


def run(split: str, n: int, *, use_llm: bool = True, replay_only: bool = False, save: bool = True) -> dict[str, Any]:
    t0 = time.time()
    rows = []
    for s in seeds(split, n):
        truth, img = generate(s, split)
        result = process(img, truth.case_id, truth.received_on, cache_id=truth.case_id,
                         replay_only=replay_only, use_llm=use_llm)
        rows.append(score_case(truth, result))
    report = {"split": split, "n": n, "use_llm": use_llm, "at": datetime.now(UTC).isoformat(timespec="seconds"),
              "seconds": round(time.time() - t0, 1), "summary": summarize(rows),
              "failures": [{k: r[k] for k in ("case_id", "defects", "route", "wrong_critical")}
                           | {"text_errors": {f: (c["want"], c["got"]) for f, c in r["text"].items() if not c["ok"]}}
                           for r in rows if r["wrong_critical"] or any(not c["ok"] for c in r["text"].values())]}
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        tag = "llm" if use_llm else "cv"
        (RESULTS / f"{split}-{tag}-{n}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                                          encoding="utf-8")
    return report


def run_baseline(split: str = "dev", n: int = 40, replay_only: bool = False, save: bool = True) -> dict[str, Any]:
    """Hybrid pipeline vs. one vision call on the whole page, on the same documents."""
    from cailens.agents import baseline, registrar

    stats: dict[str, Counter[str]] = {"ibrido": Counter(), "solo_modello": Counter()}
    for s in seeds(split, n):
        truth, img = generate(s, split)
        exp = expected(truth)
        hybrid = process(img, truth.case_id, truth.received_on, cache_id=truth.case_id, replay_only=replay_only)
        base, log = baseline.read(registrar.register(img), f"{truth.case_id}:base", replay_only)
        for name, readings, cost, ms in (("ibrido", hybrid.readings, sum(x.cost_usd for x in hybrid.steps),
                                          sum(x.ms for x in hybrid.steps)),
                                         ("solo_modello", base, log.cost_usd, log.ms)):
            c = stats[name]
            if not readings:
                c["failed"] += 1
                continue
            c["docs"] += 1
            c["cost_microusd"] += int(cost * 1e6)
            c["ms"] += ms
            for fid in form.CHECKBOX_FIELDS:
                want, got = exp[fid], readings[fid].value if fid in readings else None
                if want == AMBIGUOUS:
                    c["scratched"] += 1
                    c["scratched_flagged"] += got == "incerto"
                    continue
                c["boxes"] += 1
                c["box_flagged"] += got == "incerto"
                c["box_wrong"] += got not in (want, "incerto")
            for side in form.SIDES:
                got = readings[f"{side}.firma"].value if f"{side}.firma" in readings else None
                c["signatures"] += 1
                c["signature_wrong"] += got not in (exp[f"{side}.firma"], "incerto")
            for fid in form.TEXT_FIELDS:
                if fid in readings:
                    c["text"] += 1
                    c["text_ok"] += same(fid, exp[fid], readings[fid].value)
    table = {name: {"documents": c["docs"], "failed_calls": c["failed"],
                    "box_wrong": c["box_wrong"], "boxes": c["boxes"], "box_sent_to_review": c["box_flagged"],
                    "scratched_flagged": f"{c['scratched_flagged']}/{c['scratched']}",
                    "signature_wrong": c["signature_wrong"], "signatures": c["signatures"],
                    "text_exact_pct": _pct(c["text_ok"], c["text"]),
                    "usd_per_doc": round(c["cost_microusd"] / 1e6 / max(1, c["docs"]), 4),
                    "seconds_per_doc": round(c["ms"] / 1000 / max(1, c["docs"]), 2)}
             for name, c in stats.items()}
    report = {"split": split, "n": n, "at": datetime.now(UTC).isoformat(timespec="seconds"), "comparison": table}
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"baseline-{split}-{n}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    return report


def load_local_env() -> None:
    llm.load_local_env()


def main() -> None:  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Run the evals on a synthetic split")
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-llm", action="store_true", help="computer vision and rules only")
    ap.add_argument("--replay", action="store_true", help="cassette only, never call the API")
    ap.add_argument("--baseline", action="store_true", help="hybrid vs one vision call on the whole page")
    args = ap.parse_args()
    load_local_env()
    if args.baseline:
        print(json.dumps(run_baseline(args.split, args.n, args.replay)["comparison"], ensure_ascii=False, indent=1))
        return
    rep = run(args.split, args.n, use_llm=not args.no_llm, replay_only=args.replay)
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=1, default=str))




if __name__ == "__main__":  # pragma: no cover
    main()
