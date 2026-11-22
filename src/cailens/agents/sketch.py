# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Sketch agent: what kind of accident does section 13 show, and where did each car get hit (section 10)?

The vision model gets three crops - the sketch and the two car outlines with the drivers' arrows - and
chooses from closed lists. Its answers are low-stakes on their own (the liquidator reads the sketch
anyway) but useful as a cross-check: the validator compares them with the boxes ticked in section 12.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw

from cailens import form, llm
from cailens.agents.registrar import Registration
from cailens.schema import SCENARIOS, ZONES, Reading, StepLog

AGENT = "sketch_vlm"
PROMPT_VERSION = "sk-v2"
SYSTEM = """Sei un liquidatore sinistri RC auto. Ricevi un'immagine con tre parti di un modulo CAI:
in alto il grafico dell'incidente (sezione 13), in basso a sinistra la sagoma del veicolo A e a destra quella
del veicolo B (sezione 10), entrambe viste dall'alto con il fronte in alto, e una freccia disegnata a mano
che indica il punto d'urto iniziale.
Scenari:
- tamponamento: stessa strada e stessa direzione, A urta il retro di B;
- incrocio: due strade che si incrociano, un veicolo urta il fianco dell'altro;
- cambio_corsia: strada a piu' corsie, A si sposta nella corsia di B;
- parcheggio: A esce da un parcheggio (P) e urta B che circola;
- retromarcia: A procede all'indietro e urta B;
- rotatoria: A entra nella rotatoria e urta B che vi circola.
Zone d'urto sulla sagoma: anteriore, posteriore, lato_sx, lato_dx, ant_sx, ant_dx, post_sx, post_dx
(sinistra e destra del veicolo, con il fronte in alto).
Se una parte e' vuota o non leggibile usa "non_leggibile". "certezza": quanto e' chiaro il grafico."""


def composite(reg: Registration) -> bytes:
    def crop(box: form.Box, pad: int = 10) -> Image.Image:
        return Image.fromarray(reg.gray[box.y0 - pad:box.y1 + pad, box.x0 - pad:box.x1 + pad])

    sketch = crop(form.SKETCH_BOX)
    a, b = crop(form.IMPACT_BOX["A"], 30), crop(form.IMPACT_BOX["B"], 30)
    width = max(sketch.width, a.width + b.width + 40)
    out = Image.new("L", (width, sketch.height + max(a.height, b.height) + 40), 255)
    out.paste(sketch, (0, 0))
    y = sketch.height + 40
    out.paste(a, (0, y))
    out.paste(b, (a.width + 40, y))
    d = ImageDraw.Draw(out)
    d.text((6, y - 30), "veicolo A", fill=0, font=form.font(22))
    d.text((a.width + 46, y - 30), "veicolo B", fill=0, font=form.font(22))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def schema() -> dict[str, Any]:
    zone = {"type": "string", "enum": [*ZONES, "non_leggibile"]}
    return {"type": "object", "properties": {
        "scenario": {"type": "string", "enum": [*SCENARIOS, "non_leggibile"]},
        "impatto_A": zone, "impatto_B": zone,
        "certezza": {"type": "string", "enum": ["alta", "media", "bassa"]},
        "descrizione": {"type": "string"},
    }}


CERTAINTY = {"alta": 0.9, "media": 0.6, "bassa": 0.3}


def read(reg: Registration, cache_id: str | None, replay_only: bool = False) -> tuple[dict[str, Reading], StepLog]:
    try:
        answer, meta = llm.call_vision(
            purpose="sketch", prompt_version=PROMPT_VERSION, system=SYSTEM,
            text="Classifica il grafico e le zone d'urto secondo lo schema. Descrizione: massimo 25 parole.",
            images=[composite(reg)], schema=schema(), cache_id=cache_id, replay_only=replay_only, max_tokens=2000,
        )
    except llm.LLMUnavailable as exc:
        empty = {k: Reading(field=k, value=None, confidence=0.0, agent=AGENT, evidence={"errore": str(exc)})
                 for k in ("scenario", "A.impatto", "B.impatto")}
        return empty, StepLog(step="sketch", ms=0, ok=False, note=str(exc))
    conf = CERTAINTY.get(answer.get("certezza", "bassa"), 0.3)
    answer = {k: (None if v == "non_leggibile" else v) for k, v in answer.items()}
    out = {
        "scenario": Reading(field="scenario", value=answer.get("scenario"), confidence=conf, agent=AGENT,
                            evidence={"descrizione": str(answer.get("descrizione", ""))[:300]}),
        "A.impatto": Reading(field="A.impatto", value=answer.get("impatto_A"), confidence=conf, agent=AGENT),
        "B.impatto": Reading(field="B.impatto", value=answer.get("impatto_B"), confidence=conf, agent=AGENT),
    }
    log = StepLog(step="sketch", ms=meta["ms"], ok=True, input_tokens=meta["input_tokens"],
                  output_tokens=meta["output_tokens"], cost_usd=meta["cost_usd"],
                  note="cache" if meta.get("cached") else "")
    return out, log
