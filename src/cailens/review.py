# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""A person's corrections on top of what the agents read, and what follows from them.

Nothing the agents produced is overwritten: each review is appended, and the effective reading of a field
is the reviewer's last word if there is one. Rules, deadline and routing are then recomputed from the
effective readings, so confirming an uncertain signature really does turn a provisional 30-day term into
a final one - and the case is ready for the liquidator when nothing is left to look at.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cailens import form
from cailens.agents import rules, verifier
from cailens.schema import Reading, Result

REVIEWER = "revisore"
MARK_FIELDS = (*form.CHECKBOX_FIELDS, *(f"{s}.firma" for s in form.SIDES))


def disagreements(readings: dict[str, Reading]) -> list[str]:
    """Identifiers where the verifier's second reading differs and no person has decided yet."""
    return [f for f in verifier.FIELDS
            if (r := readings.get(f)) is not None and r.agent != REVIEWER and "seconda_lettura" in r.evidence]


def effective(result: Result, reviews: dict[str, dict[str, Any]], received_on: date) -> dict[str, Any]:
    readings: dict[str, Reading] = dict(result.readings)
    for fid, rv in reviews.items():
        readings[fid] = Reading(field=fid, value=rv["new_value"], confidence=1.0, agent=REVIEWER,
                                evidence={"da": rv["old_value"], "chi": rv["reviewer"], "quando": rv["at"]})
    if not result.registered:
        return {"readings": readings, "issues": [], "deadline": result.deadline, "route": "revisione",
                "reasons": result.reasons, "reviewed": len(reviews), "status": "da_leggere"}
    marks_left = [f for f in MARK_FIELDS if f not in reviews]
    quality_note = result.quality_note if marks_left else ""
    hw_ok = result.handwriting_ok or all(f in reviews for f in rules.CRITICAL_TEXT)
    issues = rules.validate(readings, received_on, text_ok=hw_ok)
    dl = rules.deadline(readings, received_on)
    route, reasons = rules.route(readings, issues, dl, True, hw_ok, quality_note)
    if disagree := disagreements(readings):  # the rules alone would let these through: keep the second look's veto
        route, reasons = "revisione", [f"Seconda lettura discordante: {', '.join(disagree)}.", *reasons]
    status = ("pronta" if reviews else "automatica") if route == "automatico" else "in_revisione" if reviews else "da_rivedere"
    return {"readings": readings, "issues": issues, "deadline": dl, "route": route, "reasons": reasons,
            "reviewed": len(reviews), "status": status}


def needs_look(result: Result, eff: dict[str, Any]) -> list[str]:
    """Fields a person should look at, most important first: uncertain marks, weak key text, issue fields."""
    out: list[str] = []
    readings: dict[str, Reading] = eff["readings"]
    for fid in MARK_FIELDS:
        r = readings.get(fid)
        if r and r.agent != REVIEWER and (r.value == "incerto" or r.confidence < rules.TEXT_MIN):
            out.append(fid)
    out += [f for f in disagreements(readings) if f not in out]
    for fid in rules.CRITICAL_TEXT:
        r = readings.get(fid)
        if fid in out:
            continue
        if r and r.agent != REVIEWER and r.confidence < rules.TEXT_MIN and not (r.value is None and r.confidence >= 0.8):
            out.append(fid)
    for issue in eff["issues"]:
        for fid in issue.fields:
            r = readings.get(fid)
            if fid not in out and (r is None or r.agent != REVIEWER) and fid != "schizzo":
                out.append(fid)
    return out
