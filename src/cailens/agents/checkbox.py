# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Checkbox agent: is there a cross in the box? Deterministic, measured, no model.

The reading is the share of ink inside the box (border excluded), after a local search that re-centres
the box on its printed border. Three bands, calibrated on the dev split (scripts/calibrate.py):
little ink = "no", a clear stroke = "si", in between = "incerto". A box almost full of ink is also
"incerto": that is how a cross scratched out looks, and only a person can say what was meant.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cailens import form
from cailens.agents.registrar import Registration, best_offset
from cailens.schema import Reading

AGENT = "checkbox_cv"

@dataclass(frozen=True)
class Thresholds:
    empty: float = 0.03  # at or below: no
    marked: float = 0.075  # at or above: si
    scratched: float = 0.9  # at or above: incerto (possible correction)

DEFAULT = Thresholds()

MIN_BLOB = 20  # ink components smaller than this (px) are noise, not a stroke
MIN_SPAN = 7  # ...and so are compact ones no longer than this (px): a pen dot, however blurred


def strokes(reg: Registration) -> np.ndarray:
    """The ink map without long straight lines (printed rules, a fold crease across the page)."""
    if reg.strokes is None:
        ink = reg.ink
        long_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 1)))
        long_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 61)))
        lines = cv2.dilate(np.bitwise_or(long_h, long_v), np.ones((3, 3), np.uint8))
        reg.strokes = (ink & (1 - lines)).astype(np.uint8)
    return reg.strokes


def ink_ratio(reg: Registration, box: form.Box) -> tuple[float, tuple[int, int]]:
    dx, dy = best_offset(reg.ink, box, radius=10)
    inner = form.Box(box.x0 + dx, box.y0 + dy, box.x1 + dx, box.y1 + dy).inset(5)
    patch = strokes(reg)[inner.y0:inner.y1, inner.x0:inner.x1]
    if not patch.size:
        return 0.0, (dx, dy)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB
            and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) > MIN_SPAN]
    inked = int(sum(stats[i, cv2.CC_STAT_AREA] for i in keep))
    return inked / patch.size, (dx, dy)

def classify(ratio: float, t: Thresholds = DEFAULT) -> tuple[str, float]:
    if ratio >= t.scratched:
        return "incerto", 0.35
    if ratio >= t.marked:
        return "si", float(min(1.0, 0.75 + (ratio - t.marked) / 0.2))
    if ratio <= t.empty:
        return "no", float(min(1.0, 0.75 + (t.empty - ratio) / t.empty * 0.25))
    return "incerto", 0.3

def read(reg: Registration, t: Thresholds = DEFAULT) -> dict[str, Reading]:
    out = {}
    for fid in form.CHECKBOX_FIELDS:
        ratio, (dx, dy) = ink_ratio(reg, form.fields()[fid].box)
        state, conf = classify(ratio, t)
        if not reg.quality_ok:  # a poor photo: keep the reading, but a person confirms it
            conf = min(conf, 0.5)
        out[fid] = Reading(field=fid, value=state, confidence=round(conf, 3), agent=AGENT,
                           evidence={"ink": round(ratio, 4), "dx": float(dx), "dy": float(dy)})
    return out

def ratios(reg: Registration) -> dict[str, float]:
    return {fid: ink_ratio(reg, form.fields()[fid].box)[0] for fid in form.CHECKBOX_FIELDS}

__all__ = ["DEFAULT", "Thresholds", "classify", "ink_ratio", "np", "ratios", "read"]
