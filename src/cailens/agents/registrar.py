# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Registrar agent: finds the form in a scan or photo and maps it onto the template.

Two stages, both plain OpenCV. (1) The page: the largest bright quadrilateral (on a scan it is the
whole image). (2) The printed frame: inside the straightened page, the outer contour of the ink is
the thick border of the form; its four corners give the final homography. The result is scored by
how many printed rules of the template land on ink: a low score sends the document to a person
instead of letting every later agent read the wrong pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np
from PIL import Image

from cailens import form

MIN_SCORE = 0.62  # below this the registration is not trusted at all: nothing is read
GOOD_SCORE = 0.95  # below this the page is read, but every mark goes to a person (dev/holdout: >= 0.999)
MAX_INK = 0.35  # a filled form is ~5-15% ink; more means noise, a photo of something else, or a dark page
MIN_SHARPNESS = 40.0  # variance of the Laplacian; holdout photos >= 51, the «scarsa» stress level <= 38


@dataclass
class Registration:
    gray: np.ndarray  # the page in template coordinates, uint8, H x W
    ink: np.ndarray  # binary ink map of ``gray`` (1 = ink)
    score: float  # share of template rules that land on ink, 0-1
    ok: bool
    homography: np.ndarray
    note: str = ""
    strokes: np.ndarray | None = None  # cache for the checkbox agent
    sharpness: float = 0.0

    @property
    def quality_ok(self) -> bool:
        """Sharp enough and aligned well enough to trust a mark read by the pixels."""
        return self.ok and self.score >= GOOD_SCORE and self.sharpness >= MIN_SHARPNESS

    def quality_note(self) -> str:
        bits = []
        if self.sharpness < MIN_SHARPNESS:
            bits.append(f"immagine sfocata (nitidezza {self.sharpness:.0f}, minimo {MIN_SHARPNESS:.0f})")
        if self.score < GOOD_SCORE:
            bits.append(f"allineamento imperfetto ({self.score:.2f})")
        return "; ".join(bits)


def _order(pts: np.ndarray) -> np.ndarray:
    """tl, tr, br, bl from any four (or more) points: the extreme corners."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def ink_map(gray: np.ndarray) -> np.ndarray:
    """Dark strokes against the local paper level (robust to tints and uneven light)."""
    binary = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 18)
    return binary.astype(np.uint8)


def _page_quad(gray: np.ndarray) -> np.ndarray | None:
    h, w = gray.shape
    scale = 900 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.3 * small.shape[0] * small.shape[1]:
        return None
    quad: np.ndarray = _order(cv2.convexHull(c)) / scale
    return quad


@cache
def _frame_target() -> np.ndarray:
    x0, y0, x1, y1 = form.FRAME
    half = form.FRAME_WIDTH / 2
    return np.array([[x0 - half, y0 - half], [x1 + half, y0 - half], [x1 + half, y1 + half], [x0 - half, y1 + half]],
                    dtype=np.float32)


@cache
def _template_rules() -> tuple[np.ndarray, int]:
    lines = (np.asarray(form.template_lines()) > 0).astype(np.uint8)
    return lines, int(lines.sum())


def score(ink: np.ndarray) -> float:
    """Share of the template's printed rules that have ink within 2 px in the registered page."""
    rules, total = _template_rules()
    near = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    return float((near & rules).sum() / total)


def register(img: Image.Image) -> Registration:
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    size = (form.W, form.H)
    page = _page_quad(gray)
    if page is None:
        page = np.array([[0, 0], [gray.shape[1], 0], [gray.shape[1], gray.shape[0]], [0, gray.shape[0]]], np.float32)
    corners = np.array([[0, 0], [form.W, 0], [form.W, form.H], [0, form.H]], dtype=np.float32)
    h1 = cv2.getPerspectiveTransform(page, corners)
    warped = cv2.warpPerspective(gray, h1, size, flags=cv2.INTER_LINEAR, borderValue=255)
    ink = ink_map(warped)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = max(contours, key=cv2.contourArea) if contours else None
    note = ""
    homography = h1
    if best is not None and cv2.contourArea(cv2.convexHull(best)) > 0.5 * form.W * form.H:
        frame = _order(best)
        h2 = cv2.getPerspectiveTransform(frame, _frame_target())
        homography = h2 @ h1
        warped = cv2.warpPerspective(gray, homography, size, flags=cv2.INTER_LINEAR, borderValue=255)
        ink = ink_map(warped)
    else:
        note = "cornice del modulo non trovata"
    s = score(ink)
    density = float(ink.mean())
    ok = s >= MIN_SCORE and not note and density <= MAX_INK  # fail closed: no frame, no reading
    if s < MIN_SCORE:
        note = (note + "; " if note else "") + f"allineamento insufficiente ({s:.2f})"
    if density > MAX_INK:
        note = (note + "; " if note else "") + f"troppo inchiostro ({density:.0%}): non sembra un modulo"
    sharp = float(cv2.Laplacian(warped, cv2.CV_64F).var())
    return Registration(warped, ink, s, ok, homography, note, sharpness=sharp)


def best_offset(ink: np.ndarray, box: form.Box, radius: int = 7) -> tuple[int, int]:
    """Shift (dx, dy) that puts the most ink on the printed border of ``box``: absorbs the few pixels
    a phone photo bends away from a perfect homography."""
    pad = radius + 3
    y0, y1 = max(0, box.y0 - pad), min(ink.shape[0], box.y1 + pad)
    x0, x1 = max(0, box.x0 - pad), min(ink.shape[1], box.x1 + pad)
    region = ink[y0:y1, x0:x1].astype(np.int32)
    ring = np.zeros((box.h + 1, box.w + 1), np.int32)
    ring[:2, :] = ring[-2:, :] = 1
    ring[:, :2] = ring[:, -2:] = 1
    best, best_val = (0, 0), -1
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ry, rx = box.y0 + dy - y0, box.x0 + dx - x0
            if ry < 0 or rx < 0 or ry + ring.shape[0] > region.shape[0] or rx + ring.shape[1] > region.shape[1]:
                continue
            val = int((region[ry:ry + ring.shape[0], rx:rx + ring.shape[1]] * ring).sum())
            if val > best_val or (val == best_val and abs(dx) + abs(dy) < abs(best[0]) + abs(best[1])):
                best, best_val = (dx, dy), val
    return best
