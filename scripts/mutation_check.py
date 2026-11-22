# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir

"""Break the code on purpose and check that the tests notice. Writes docs/mutation-results.json.

Coverage says a line ran. This says a test fails when the line is wrong. Each mutation removes one
guarantee from a temporary copy of the project, and the whole suite runs against that copy. A mutation
nothing catches is reported as "survived", not hidden: the fix is a missing test, never deleting the
mutation.

    python scripts/mutation_check.py           # all mutations (~1 suite run each, stops at the first failure)
    python scripts/mutation_check.py --check   # only verify every mutation still applies to the code
"""

import datetime as dt
import json
import os
import re
import shutil
import subprocess  # nosec B404 — runs this repo's own test suite only
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SKIP = {".venv", "data", ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "docs", ".coverage", "evals", ".env.local"}

MUTATIONS = [
    ('Declared total never compared with the crosses', 'src/cailens/agents/rules.py', 'elif int(total) != len(yes):', 'elif False:'),
    ('Both signatures no longer give 30 days', 'src/cailens/agents/rules.py', 'if all(v == "si" for v in sig):', 'if False:'),
    ('An uncertain signature gets the longer term', 'src/cailens/agents/rules.py', 'days, basis = 30, ("Firma incerta', 'days, basis = 60, ("Firma incerta'),
    ('Injuries no longer give 90 days', 'src/cailens/agents/rules.py', 'return Deadline(days=90,', 'return Deadline(days=60,'),
    ('A poor photo is not sent to a person', 'src/cailens/agents/rules.py', '    if quality_note:\n', '    if False:\n'),
    ('Uncertain boxes are not sent to a person', 'src/cailens/agents/rules.py', 'if unsure_boxes:', 'if False:'),
    ('Codice fiscale checksum ignored', 'src/cailens/agents/rules.py', 'if _v(readings, fid) and not formats.cf_valid(_v(readings, fid)):', 'if False:'),
    ('Accident date after receipt accepted', 'src/cailens/agents/rules.py', 'elif when and when > received_on:', 'elif False:'),
    ('Missing plate or insurer of B accepted', 'src/cailens/agents/rules.py', 'for fid in ("B.targa", "B.compagnia"):', 'for fid in ():'),
    ('Same plate for A and B accepted', 'src/cailens/agents/rules.py', 'if ta and tb and ta == tb:', 'if False:'),
    ('Sketch never compared with the boxes', 'src/cailens/agents/rules.py', 'if boxes and not boxes & SCENARIO_BOXES[scen.value]:', 'if False:'),
    ('A scratched-out cross read as a cross', 'src/cailens/agents/checkbox.py', 'if ratio >= t.scratched:', 'if False:'),
    ('Pen dots counted as strokes', 'src/cailens/agents/checkbox.py', '            and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) > MIN_SPAN]', '            ]'),
    ('Fold lines not removed before measuring', 'src/cailens/agents/checkbox.py', 'reg.strokes = (ink & (1 - lines)).astype(np.uint8)', 'reg.strokes = ink'),
    ('Poor photo keeps full confidence on crosses', 'src/cailens/agents/checkbox.py', '            conf = min(conf, 0.5)', '            conf = conf'),
    ('Poor photo keeps full confidence on signatures', 'src/cailens/agents/signature.py', '            conf = min(conf, 0.5)', '            conf = conf'),
    ("An empty signature box is never 'no'", 'src/cailens/agents/signature.py', 'if pixels <= t.absent_px:', 'if False:'),
    ('Registrar trusts a page without the printed frame', 'src/cailens/agents/registrar.py', 'ok = s >= MIN_SCORE and not note and density <= MAX_INK', 'ok = s >= MIN_SCORE'),
    ('Quality gate always passes', 'src/cailens/agents/registrar.py', 'return self.ok and self.score >= GOOD_SCORE and self.sharpness >= MIN_SHARPNESS', 'return self.ok'),
    ('An illegible field gets high confidence', 'src/cailens/agents/handwriting.py', '        return None, 0.2', '        return None, 0.9'),
    ('A value with the wrong shape gets high confidence', 'src/cailens/agents/handwriting.py', '        return value, 0.4', '        return value, 0.92'),
    ('Cassette never replayed (every eval pays again)', 'src/cailens/llm.py', 'if path and path.exists():', 'if False:'),
    ('A model refusal is treated as an answer', 'src/cailens/llm.py', 'if msg.stop_reason == "refusal":', 'if False:'),
    ('Schema not closed for structured output', 'src/cailens/llm.py', 'out["additionalProperties"] = False', 'pass'),
    ('Pipeline reads a page it could not align', 'src/cailens/pipeline.py', '    if not reg.ok:\n', '    if False:\n'),
    ('A review stores the old value', 'src/cailens/review.py', 'value=rv["new_value"]', 'value=rv["old_value"]'),
    ('CSRF token not checked', 'src/cailens/web.py', 'return user is not None and hmac.compare_digest(token.encode(), user["csrf"].encode())', 'return user is not None'),
    ('Demo visitors share one scope', 'src/cailens/web.py', 'return hashlib.sha256(token.encode()).hexdigest()[:16] if demo_mode else "operatore"', 'return "condiviso"'),
    ('Generation not rate limited', 'src/cailens/web.py', '            if len(q) >= self.per_ip:\n                return "Troppe richieste', '            if False:\n                return "Troppe richieste'),
    ('Login attempts never blocked', 'src/cailens/web.py', 'if (msg := login_limiter.blocked(ip)) is not None:', 'if False:'),
    ('Open redirect after sign-in', 'src/cailens/web.py', 'return target if not parts.scheme and not parts.netloc and not target.startswith("//") else "/"', 'return target'),
    ('Content-Security-Policy header dropped', 'src/cailens/web.py', 'resp.headers["Content-Security-Policy"] = (', 'resp.headers["X-Unused"] = ('),
    ('Any image name served from the case folder', 'src/cailens/web.py', 'if kind not in ("pagina", "originale"):', 'if False:'),
    ('MCP tool opens files outside its inbox', 'src/cailens/mcp_server.py', 'if not path.is_relative_to(root):', 'if False:'),
    ('A disagreement between the two readings is ignored', 'src/cailens/agents/verifier.py', 'if formats.compact(fid, second) != formats.compact(fid, first.value):', 'if False:'),
    ("Review view drops the second reading's veto", 'src/cailens/review.py', 'if disagree := disagreements(readings):', 'if False and (disagree := disagreements(readings)):'),
    ('Cassette write failure loses a paid answer', 'src/cailens/llm.py', 'except OSError:\n            meta["cassette"]', 'except ZeroDivisionError:\n            meta["cassette"]'),
    ('Verifier down: identifiers keep full confidence', 'src/cailens/agents/verifier.py', 'out = {f: r.model_copy(update={"confidence": min(r.confidence, 0.5)}) for f, r in readings.items() if f in FIELDS}', 'out = {}'),
]


def copy_project(target: Path) -> None:
    shutil.copytree(ROOT, target, ignore=lambda _d, names: [n for n in names if n in SKIP], dirs_exist_ok=True)


def run_tests(cwd: Path) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    # fixed argv (this interpreter + pytest), no shell
    done = subprocess.run(  # nosec B603
        [PYTHON, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    found = re.findall(r"^FAILED (\S+)", done.stdout, re.M) or re.findall(r"^ERROR (\S+)", done.stdout, re.M)
    return done.returncode == 0, found[0] if found else ""


def apply(text: str, old: str, new: str, name: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"Mutation no longer applies exactly once, fix the script: {name}")
    return text.replace(old, new, 1)


def main() -> None:
    for name, relative, old, new in MUTATIONS:
        apply((ROOT / relative).read_text(encoding="utf-8"), old, new, name)
    if "--check" in sys.argv:
        print(len(MUTATIONS), "mutations apply")
        return
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "copy"
        copy_project(work)
        passed, failing = run_tests(work)
        if not passed:
            raise SystemExit(f"The unmodified copy does not pass ({failing}), so nothing here could be trusted.")
        for name, relative, old, new in MUTATIONS:
            path = work / relative
            original = path.read_text(encoding="utf-8")
            path.write_text(apply(original, old, new, name), encoding="utf-8", newline="\n")
            passed, caught_by = run_tests(work)
            path.write_text(original, encoding="utf-8", newline="\n")
            results.append(
                {"bug": name, "file": relative, "caught": not passed, "caught_by": caught_by if not passed else None}
            )
            print(("caught  " if not passed else "SURVIVED"), name, caught_by, flush=True)
    summary = {
        "measured": dt.date.today().isoformat(),
        "bugs_injected": len(results),
        "caught": sum(row["caught"] for row in results),
        "results": results,
    }
    (ROOT / "docs" / "mutation-results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(summary["caught"], "of", summary["bugs_injected"])


if __name__ == "__main__":
    main()
