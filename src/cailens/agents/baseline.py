# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Baseline for the evals only: the whole page to the vision model, every field from one answer.

It exists to measure the design choice. If one model call read crosses and signatures as well as the
pixel counters do, the hybrid pipeline would be extra machinery; the eval says whether it does.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from cailens import form, formats, llm
from cailens.agents.registrar import Registration
from cailens.schema import Reading, StepLog

AGENT = "baseline_vlm"
PROMPT_VERSION = "base-v2"
SYSTEM = """Sei un operatore che legge moduli CAI (constatazione amichevole di incidente) compilati a mano.
Leggi l'intera pagina e compila lo schema:
- caselle_A / caselle_B: i numeri (1-17) delle circostanze della sezione 12 barrate per il veicolo A (colonna
  di caselle a sinistra) e B (a destra). Una casella annerita o scarabocchiata va in "caselle_incerte".
- feriti, veicoli, oggetti: "si", "no" o "nessuna" se nessuna casella e' barrata.
- firma_A / firma_B: true se nel riquadro 15 c'e' una firma.
- campi di testo: trascrivi esattamente, in maiuscolo; "" se vuoti o illeggibili."""


def page_jpeg(reg: Registration) -> bytes:
    img = Image.fromarray(reg.gray).resize((1100, 1556), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def schema() -> dict[str, Any]:
    text = {fid.replace(".", "_"): {"type": "string"} for fid in form.TEXT_FIELDS}
    yn = {"type": "string", "enum": ["si", "no", "nessuna"]}
    boxes = {"type": "array", "items": {"type": "integer"}}
    return {"type": "object", "properties": {
        "caselle_A": boxes, "caselle_B": boxes,
        "caselle_incerte": {"type": "array", "items": {"type": "string"}},
        "feriti": yn, "veicoli": yn, "oggetti": yn,
        "firma_A": {"type": "boolean"}, "firma_B": {"type": "boolean"}, **text,
    }}


def read(reg: Registration, cache_id: str | None, replay_only: bool = False) -> tuple[dict[str, Reading], StepLog]:
    try:
        ans, meta = llm.call_vision(purpose="baseline", prompt_version=PROMPT_VERSION, system=SYSTEM,
                                    text="Compila lo schema leggendo la pagina.", images=[page_jpeg(reg)],
                                    schema=schema(), cache_id=cache_id, replay_only=replay_only)
    except llm.LLMUnavailable as exc:
        return {}, StepLog(step="baseline", ms=0, ok=False, note=str(exc))
    out: dict[str, Reading] = {}

    def put(fid: str, value: str | None) -> None:
        out[fid] = Reading(field=fid, value=value, confidence=0.9, agent=AGENT)

    for side in form.SIDES:
        chosen = {int(n) for n in ans.get(f"caselle_{side}") or [] if 1 <= int(n) <= 17}
        for n in range(1, 18):
            put(f"{side}.c{n:02d}", "si" if n in chosen else "no")
        put(f"{side}.firma", "si" if ans.get(f"firma_{side}") else "no")
    for name in ("feriti", "veicoli", "oggetti"):
        v = ans.get(name)
        put(f"{name}_si", "si" if v == "si" else "no")
        put(f"{name}_no", "si" if v == "no" else "no")
    for fid in form.TEXT_FIELDS:
        raw = ans.get(fid.replace(".", "_"))
        put(fid, formats.normalize(raw) if isinstance(raw, str) else None)
    return out, StepLog(step="baseline", ms=meta["ms"], ok=True, input_tokens=meta["input_tokens"],
                        output_tokens=meta["output_tokens"], cost_usd=meta["cost_usd"],
                        note="cache" if meta.get("cached") else "")
