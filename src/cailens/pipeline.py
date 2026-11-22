# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""The orchestrator: an explicit sequence of agents with a log entry for every step.

    registrar -> checkbox -> signature -> handwriting -> sketch -> validator -> deadline -> router
                                                                    -> verifier (only before «automatico»)

It is a fixed state machine on purpose: each agent has one job and a typed contract, the order never
depends on what a model says, and a failing agent degrades the result (fields go to review) instead of
stopping it. If registration fails, nothing else reads the page.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import TypeVar

from PIL import Image

from cailens.agents import checkbox, handwriting, registrar, rules, signature, sketch, verifier
from cailens.agents.registrar import Registration
from cailens.schema import Deadline, Reading, Result, StepLog

EventSink = Callable[[str, StepLog], None]
T = TypeVar("T")


def process(img: Image.Image, case_id: str, received_on: date, *, cache_id: str | None = None,
            replay_only: bool = False, use_llm: bool = True, on_step: EventSink | None = None,
            keep: list[Registration] | None = None) -> Result:
    """Run every agent on one image. ``keep``, if given, receives the registration (for the page image)."""
    steps: list[StepLog] = []

    def log(entry: StepLog) -> None:
        steps.append(entry)
        if on_step:
            on_step(case_id, entry)

    def timed(name: str, fn: Callable[[], T]) -> T:
        t0 = time.perf_counter()
        out = fn()
        log(StepLog(step=name, ms=int((time.perf_counter() - t0) * 1000), ok=True))
        return out

    reg = timed("registrar", lambda: registrar.register(img))
    if not isinstance(reg, registrar.Registration):  # pragma: no cover - a programming error, not an input error
        raise TypeError("registrar returned an unexpected object")
    steps[-1].ok, steps[-1].note = reg.ok, reg.note or f"punteggio {reg.score:.2f}, nitidezza {reg.sharpness:.0f}"
    if keep is not None:
        keep.append(reg)
    readings: dict[str, Reading] = {}
    if not reg.ok:
        dl = Deadline(days=None, due=None, provisional=True, basis="Modulo non letto: termine da stabilire a mano.")
        verdict, reasons = rules.route(readings, [], dl, False, False)
        return Result(case_id=case_id, registered=False, registration_score=round(reg.score, 3), readings={},
                      issues=[], deadline=dl, route=verdict, reasons=reasons, steps=steps)  # type: ignore[arg-type]

    readings.update(timed("checkbox", lambda: checkbox.read(reg)))
    readings.update(timed("signature", lambda: signature.read(reg)))
    hw_ok = False
    if use_llm:
        hw, entry = handwriting.read(reg, cache_id and f"{cache_id}:hw", replay_only)
        log(entry)
        readings.update(hw)
        hw_ok = entry.ok
        sk, entry = sketch.read(reg, cache_id and f"{cache_id}:sk", replay_only)
        log(entry)
        readings.update(sk)
    else:
        log(StepLog(step="handwriting", ms=0, ok=False, note="modello disattivato"))

    t0 = time.perf_counter()
    issues = rules.validate(readings, received_on, text_ok=hw_ok)
    dl = rules.deadline(readings, received_on)
    verdict, reasons = rules.route(readings, issues, dl, True, hw_ok, reg.quality_note())
    if verdict == "automatico" and use_llm:
        fixed, disagree, entry = verifier.verify(reg, readings, cache_id and f"{cache_id}:ver", replay_only)
        log(entry)
        readings.update(fixed)
        verdict, reasons = rules.route(readings, issues, dl, True, hw_ok, reg.quality_note())
        if disagree:
            reasons.insert(0, f"Seconda lettura discordante: {', '.join(disagree)}.")
            verdict = "revisione"
    log(StepLog(step="regole", ms=int((time.perf_counter() - t0) * 1000), ok=True,
                note=f"{len(issues)} segnalazioni, esito {verdict}"))
    scen = readings.get("scenario")
    return Result(case_id=case_id, registered=True, registration_score=round(reg.score, 3), readings=readings,
                  issues=issues, deadline=dl, route=verdict, reasons=reasons, steps=steps,  # type: ignore[arg-type]
                  scenario=scen.value if scen and scen.value else None,  # type: ignore[arg-type]
                  scenario_confidence=scen.confidence if scen else 0.0, quality_note=reg.quality_note(),
                  handwriting_ok=hw_ok, sharpness=round(reg.sharpness, 1))
