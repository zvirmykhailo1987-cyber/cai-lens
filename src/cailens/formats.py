# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Shapes of Italian identifiers and dates: deterministic checks on what the model transcribed."""

from __future__ import annotations

import re
from datetime import date, datetime

from cailens.synth import cf_check

PATTERNS = {
    "targa": re.compile(r"[A-HJ-NPR-TV-Z]{2}\d{3}[A-HJ-NPR-TV-Z]{2}"),
    "cf": re.compile(r"[A-Z]{6}\d{2}[ABCDEHLMPRST]\d{2}[A-Z]\d{3}[A-Z]"),
    "polizza": re.compile(r"[A-Z0-9]{6,14}"),
    "patente": re.compile(r"[A-Z]{2}\d{7}[A-Z]"),
    "totale": re.compile(r"\d{1,2}"),
    "ora": re.compile(r"([01]\d|2[0-3]):[0-5]\d"),
    "data": re.compile(r"\d{2}/\d{2}/\d{4}"),
}


def kind(field: str) -> str:
    return field.split(".")[-1]


def normalize(value: str | None) -> str | None:
    if value is None:
        return None
    v = " ".join(value.upper().replace("’", "'").split())
    return v or None


NO_WITNESS = {"NESSUNO", "NESSUN TESTIMONE", "NESSUNA", "-", "--"}


def compact(field: str, value: str | None) -> str | None:
    """Canonical form for comparisons: «AB 123 CD» is the plate AB123CD, «ASS.NI» and «ASSNI» name the same
    insurer, «NESSUNO» in the witnesses box means no witnesses."""
    v = normalize(value)
    if v is None:
        return None
    k = kind(field)
    if k in ("targa", "cf", "polizza", "patente", "totale"):
        v = v.replace(" ", "")
    elif k == "compagnia":
        v = " ".join(v.replace(".", " ").split()).replace(" ", "")
    elif k == "testimoni" and v in NO_WITNESS:
        return None
    return v


def shape_ok(field: str, value: str | None) -> bool | None:
    """True/False if the field has a known shape, None if it is free text."""
    pat = PATTERNS.get(kind(field))
    if pat is None or value is None:
        return None
    v = compact(field, value) or ""
    if not pat.fullmatch(v):
        return False
    if kind(field) == "data":
        return parse_date(v) is not None
    return True


def cf_valid(value: str | None) -> bool:
    v = compact("cf", value) or ""
    return bool(PATTERNS["cf"].fullmatch(v)) and cf_check(v[:15]) == v[15]


def parse_date(value: str | None) -> date | None:
    try:
        return datetime.strptime((value or "").strip(), "%d/%m/%Y").date()
    except ValueError:
        return None
