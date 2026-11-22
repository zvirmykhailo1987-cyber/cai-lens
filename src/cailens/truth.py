# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""What each field of a synthetic form should read as: the answer key for the evals."""

from __future__ import annotations

from cailens.schema import CaseTruth

AMBIGUOUS = "annullata"  # a scratched-out cross: the only right answer is "a person must look"


def expected(t: CaseTruth) -> dict[str, str | None]:
    """Field id -> expected value. Marks and signatures: "si"/"no"/"annullata"; text: as written."""
    out: dict[str, str | None] = {
        "data": t.data, "ora": t.ora, "luogo": t.luogo, "testimoni": t.testimoni,
    }
    for name, val in (("feriti", t.feriti), ("veicoli", t.veicoli), ("oggetti", t.oggetti)):
        out[f"{name}_si"] = "si" if val is True else "no"
        out[f"{name}_no"] = "si" if val is False else "no"
    for side, p in (("A", t.A), ("B", t.B)):
        for key in ("cognome", "nome", "cf", "targa", "compagnia", "polizza", "cond_cognome", "cond_nome",
                    "patente", "danni", "osservazioni", "totale"):
            out[f"{side}.{key}"] = getattr(p, key)
        for n in range(1, 18):
            fid = f"{side}.c{n:02d}"
            out[fid] = AMBIGUOUS if n in p.annullate else ("si" if n in p.circostanze else "no")
        out[f"{side}.firma"] = "si" if p.firma else "no"
        out[f"{side}.impatto"] = p.impatto
    return out
