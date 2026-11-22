# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Verifier agent: a second, independent reading of the identifiers before a case goes through unchecked.

The holdout eval showed the one error that slips past every rule: a digit read as a similar digit (3 for 8)
that still gives a well-formed plate or policy number. Only a second look catches that. So, when a case is
about to be routed «automatico», the identifiers are cut again at a larger scale and read with a prompt
that asks for character-by-character care. Any disagreement sends the field to a person. Cases already in
review are not re-read: nothing would change for them, and it saves the call.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw

from cailens import form, formats, llm
from cailens.agents.handwriting import strip
from cailens.agents.registrar import Registration
from cailens.schema import Reading, StepLog

AGENT = "verifier_vlm"
PROMPT_VERSION = "ver-v1"
FIELDS = ("data", "A.targa", "B.targa", "A.polizza", "B.polizza", "A.cf", "B.cf")
SYSTEM = """Sei un secondo controllore. Ricevi strisce di un modulo CAI con identificativi scritti a mano.
Leggi CARATTERE PER CARATTERE, senza indovinare dal contesto. Attenzione alle coppie che si confondono:
3/8, 0/O, 1/I, 5/S, 2/Z, 6/G, 7/1, B/8. Trascrivi esattamente cio' che vedi; campo vuoto o illeggibile: ""."""


def mosaic(reg: Registration) -> bytes:
    strips = []
    for fid in FIELDS:
        s = strip(reg, fid)
        strips.append((fid, s.resize((int(s.width * 1.5), int(s.height * 1.5)), Image.Resampling.LANCZOS)))
    label_w, gap = 200, 10
    out = Image.new("L", (label_w + max(s.width for _, s in strips), sum(s.height + gap for _, s in strips)), 255)
    d = ImageDraw.Draw(out)
    y = 0
    for fid, s in strips:
        d.text((8, y + s.height // 2), fid, fill=0, font=form.font(24, "geist-mono.woff2"), anchor="lm")
        out.paste(s, (label_w, y))
        y += s.height + gap
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def schema() -> dict[str, Any]:
    return {"type": "object", "properties": {fid: {"type": "string"} for fid in FIELDS}}


def verify(reg: Registration, readings: dict[str, Reading], cache_id: str | None,
           replay_only: bool = False) -> tuple[dict[str, Reading], list[str], StepLog]:
    """Return updated readings, the fields where the two readings disagree, and the step log."""
    try:
        answer, meta = llm.call_vision(purpose="verifier", prompt_version=PROMPT_VERSION, system=SYSTEM,
                                       text="Trascrivi gli identificativi secondo lo schema.", images=[mosaic(reg)],
                                       schema=schema(), cache_id=cache_id, replay_only=replay_only, max_tokens=1500)
    except llm.LLMUnavailable as exc:
        # no second look possible: the identifiers cannot be trusted blind
        out = {f: r.model_copy(update={"confidence": min(r.confidence, 0.5)}) for f, r in readings.items() if f in FIELDS}
        return out, [], StepLog(step="verifier", ms=0, ok=False, note=str(exc))
    out, disagree = {}, []
    for fid in FIELDS:
        first = readings.get(fid)
        if first is None:
            continue
        second = formats.normalize(answer.get(fid) or None)
        if formats.compact(fid, second) != formats.compact(fid, first.value):
            disagree.append(fid)
            out[fid] = first.model_copy(update={"confidence": min(first.confidence, 0.45),
                                                "evidence": {**first.evidence, "seconda_lettura": second}})
    log = StepLog(step="verifier", ms=meta["ms"], ok=True, input_tokens=meta["input_tokens"],
                  output_tokens=meta["output_tokens"], cost_usd=meta["cost_usd"],
                  note=("cache" if meta.get("cached") else "") + (f" discordanti: {', '.join(disagree)}" if disagree else ""))
    return out, disagree, log
