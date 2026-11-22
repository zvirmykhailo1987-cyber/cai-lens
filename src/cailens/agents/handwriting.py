# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Handwriting agent: the vision model transcribes the text fields, nothing else.

The registered page is cut into one strip per field and the strips are stacked into a single image,
each labelled with its field id. The model sees large, isolated handwriting instead of a whole page,
and answers in a strict schema: one value per field, null when the field is empty, and a separate list
of fields it could not read. Confidence is not the model's opinion: it comes from whether the value
was readable and whether it has the shape the field requires (plate, codice fiscale, date...).
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw

from cailens import form, formats, llm
from cailens.agents.registrar import Registration
from cailens.schema import Reading, StepLog

AGENT = "handwriting_vlm"
PROMPT_VERSION = "hw-v2"
FIELDS = form.TEXT_FIELDS
SYSTEM = """Sei un operatore che trascrive moduli CAI (constatazione amichevole) compilati a mano in stampatello.
Ricevi un'immagine con una striscia per campo; a sinistra di ogni striscia c'e' l'identificativo del campo.
Regole:
1. Trascrivi ESATTAMENTE cio' che e' scritto, in maiuscolo, anche se ti sembra un errore (non correggere codici
   fiscali, targhe o date: un errore di scrittura va riportato com'e').
2. Campo vuoto: stringa vuota "".
3. Se c'e' scrittura ma non riesci a leggerla con sicurezza: "" e aggiungi l'identificativo a "illeggibili".
4. Non usare conoscenze esterne e non completare parole tagliate."""


def strip(reg: Registration, fid: str) -> Image.Image:
    box = form.fields()[fid].box
    pad_y = 8
    crop = reg.gray[max(0, box.y0 - pad_y):box.y1 + pad_y // 2, max(0, box.x0 - 6):min(form.W, box.x1 + 6)]
    img = Image.fromarray(crop)
    scale = 1.6
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)


def mosaic(reg: Registration) -> bytes:
    strips = [(fid, strip(reg, fid)) for fid in FIELDS]
    label_w, gap = 190, 6
    width = label_w + max(s.width for _, s in strips)
    height = sum(s.height + gap for _, s in strips)
    out = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(out)
    y = 0
    for fid, s in strips:
        d.rectangle((0, y, label_w - 8, y + s.height), fill=232)
        d.text((8, y + s.height // 2), fid, fill=0, font=form.font(20, "geist-mono.woff2"), anchor="lm")
        out.paste(s, (label_w, y))
        y += s.height + gap
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def schema() -> dict[str, Any]:
    props: dict[str, Any] = {fid: {"type": "string"} for fid in FIELDS}  # "" = empty: the API caps union types
    props["illeggibili"] = {"type": "array", "items": {"type": "string", "enum": list(FIELDS)}}
    return {"type": "object", "properties": props}


def confidence(fid: str, value: str | None, illegible: bool) -> tuple[str | None, float]:
    if illegible:
        return None, 0.2
    if value is None:
        return None, 0.8  # read as empty
    shape = formats.shape_ok(fid, value)
    if shape is False:
        return value, 0.4
    return value, 0.92 if shape else 0.85


def read(reg: Registration, cache_id: str | None, replay_only: bool = False) -> tuple[dict[str, Reading], StepLog]:
    image = mosaic(reg)
    try:
        answer, meta = llm.call_vision(
            purpose="handwriting", prompt_version=PROMPT_VERSION, system=SYSTEM,
            text="Trascrivi i campi secondo le regole e lo schema.", images=[image], schema=schema(),
            cache_id=cache_id, replay_only=replay_only,
        )
    except llm.LLMUnavailable as exc:
        readings = {fid: Reading(field=fid, value=None, confidence=0.0, agent=AGENT, evidence={"errore": str(exc)})
                    for fid in FIELDS}
        return readings, StepLog(step="handwriting", ms=0, ok=False, note=str(exc))
    bad = set(answer.get("illeggibili") or [])
    readings = {}
    for fid in FIELDS:
        raw = answer.get(fid)
        value, conf = confidence(fid, formats.normalize(raw) if isinstance(raw, str) else None, fid in bad)  # "" -> None
        readings[fid] = Reading(field=fid, value=value, confidence=conf, agent=AGENT,
                                evidence={"forma": None if value is None else str(formats.shape_ok(fid, value))})
    log = StepLog(step="handwriting", ms=meta["ms"], ok=True, input_tokens=meta["input_tokens"],
                  output_tokens=meta["output_tokens"], cost_usd=meta["cost_usd"],
                  note="cache" if meta.get("cached") else "")
    return readings, log
