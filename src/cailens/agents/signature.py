# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Signature agent: is the signature box signed? Presence only - this never verifies who signed.

It matters because art. 148 CAP gives the insurer 30 days instead of 60 when both drivers signed. The
measure is the ink inside the box and how wide the longest stroke runs: a stray dot is not a
signature, initials are small but wide enough, and anything in between goes to a person.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cailens import form
from cailens.agents.registrar import Registration, best_offset
from cailens.schema import Reading

AGENT = "signature_cv"


@dataclass(frozen=True)
class Thresholds:
    absent_px: int = 60  # ink pixels at or below: no signature
    present_px: int = 260  # at or above (and wide enough): signed
    present_width: int = 70  # horizontal extent of the inked area, px


DEFAULT = Thresholds()


def measure(reg: Registration, box: form.Box) -> tuple[int, int]:
    dx, dy = best_offset(reg.ink, box, radius=6)
    inner = form.Box(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy).inset(6)
    patch = reg.ink[inner.y0:inner.y1, inner.x0:inner.x1]
    patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop JPEG speckle
    pixels = int(patch.sum())
    cols = np.where(patch.any(axis=0))[0]
    width = int(cols[-1] - cols[0]) if cols.size else 0
    return pixels, width


def classify(pixels: int, width: int, t: Thresholds = DEFAULT) -> tuple[str, float]:
    if pixels <= t.absent_px:
        return "no", float(min(1.0, 0.8 + (t.absent_px - pixels) / t.absent_px * 0.2))
    if pixels >= t.present_px and width >= t.present_width:
        return "si", float(min(1.0, 0.8 + (pixels - t.present_px) / 2000))
    return "incerto", 0.3


def read(reg: Registration, t: Thresholds = DEFAULT) -> dict[str, Reading]:
    out = {}
    for side in form.SIDES:
        fid = f"{side}.firma"
        pixels, width = measure(reg, form.fields()[fid].box)
        state, conf = classify(pixels, width, t)
        if not reg.quality_ok:  # a poor photo: keep the reading, but a person confirms it
            conf = min(conf, 0.5)
        out[fid] = Reading(field=fid, value=state, confidence=round(conf, 3), agent=AGENT,
                           evidence={"ink_px": float(pixels), "width_px": float(width)})
    return out
