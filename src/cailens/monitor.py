# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Process monitor: where the time goes, why documents go to a person, and what to automate next.

Everything here is measured from the step log and the review log - no assumed minutes or savings.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

REASON_KINDS = (  # (keyword in the router's reason, label, what would remove it)
    ("Modulo non allineato", "Allineamento", "foto guidata dall'app: inquadratura con i quattro angoli"),
    ("Foto di qualità bassa", "Qualità della foto", "controllo di nitidezza al momento dello scatto"),
    ("Lettura della scrittura", "Modello non disponibile", "coda di riprova automatica"),
    ("Seconda lettura discordante", "Letture discordanti", "dati di targa e polizza da banca dati (ANIA/ACI), non da lettura"),
    ("Caselle incerte", "Caselle incerte", "modulo digitale CAI (Reg. IVASS 56/2025): niente croci a mano"),
    ("Firma", "Firme da verificare", "firma elettronica (FEA) nel modulo digitale"),
    ("Campi chiave poco leggibili", "Scrittura poco leggibile", "dati di targa e polizza da banca dati, non da lettura"),
    ("feriti", "Feriti", "nessuna: richiede un liquidatore"),
    ("Termine dell'offerta provvisorio", "Termine provvisorio", "si risolve con la verifica di firme e feriti"),
)
COHERENCE = ("Coerenza del modulo", "verifiche alla compilazione nel modulo digitale")


def kind_of(reason: str) -> tuple[str, str]:
    for key, label, fix in REASON_KINDS:
        if key in reason:
            return label, fix
    return COHERENCE


def summary(cases: list[dict[str, Any]], events: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_step[e["step"]].append(e)
    steps = []
    for step in ("registrar", "checkbox", "signature", "handwriting", "sketch", "verifier", "regole"):
        rows = by_step.get(step, [])
        live = [r["ms"] for r in rows if r["ms"] > 0]
        steps.append({
            "step": step, "runs": len(rows),
            "median_ms": round(statistics.median(live)) if live else None,
            "p90_ms": round(sorted(live)[int(0.9 * (len(live) - 1))]) if live else None,
            "failures": sum(1 for r in rows if not r["ok"]),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        })
    routes = Counter(c["result"]["route"] for c in cases)
    kinds: Counter[str] = Counter()
    fixes: dict[str, str] = {}
    for c in cases:
        seen = set()
        for reason in c["result"]["reasons"]:
            label, fix = kind_of(reason)
            if label not in seen:
                kinds[label] += 1
                fixes[label] = fix
                seen.add(label)
    corrected = Counter(r["field"] for r in reviews if r["action"] == "correzione")
    confirmed = Counter(r["field"] for r in reviews if r["action"] == "conferma")
    n = len(cases) or 1
    return {
        "cases": len(cases),
        "routes": dict(routes),
        "auto_pct": round(100 * routes.get("automatico", 0) / n, 1),
        "steps": steps,
        "max_median_ms": max([s["median_ms"] or 0 for s in steps] or [1]) or 1,
        "reason_kinds": [{"label": k, "cases": v, "pct": round(100 * v / n, 1), "fix": fixes[k]}
                         for k, v in kinds.most_common()],
        "reviews": len(reviews),
        "corrected": corrected.most_common(8),
        "confirmed": confirmed.most_common(8),
        "agent_right_pct": round(100 * sum(confirmed.values()) / max(1, len(reviews)), 1) if reviews else None,
    }
