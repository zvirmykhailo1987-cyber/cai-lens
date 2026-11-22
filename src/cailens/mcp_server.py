# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""MCP server: CAI-Lens as tools for another agent (a claims assistant, Claude Code, a workflow).

Run:  python -m cailens.mcp_server            (stdio)
      python -m cailens.mcp_server --http     (streamable HTTP on 127.0.0.1:8766, no auth: loopback only)

``read_cai`` only opens images inside CAI_MCP_ROOT (default: data/inbox), never an arbitrary path.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from PIL import Image, UnidentifiedImageError

from cailens import formats, synth
from cailens.agents import rules
from cailens.paths import DATA
from cailens.pipeline import process
from cailens.schema import Reading

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
MAX_PIXELS = 40_000_000
server = MCPServer(
    name="cai-lens",
    instructions=(
        "Reads the Italian accident report form (CAI). Each field comes with a confidence; anything uncertain "
        "is listed in 'reasons' and must be checked by a person. deadline_148 applies art. 148 CAP. "
        "Never present a reading as a decision on liability or on the offer."
    ),
)


def inbox() -> Path:
    return Path(os.environ.get("CAI_MCP_ROOT", DATA / "inbox")).resolve()


def _open(image_path: str) -> Image.Image | str:
    root = inbox()
    path = (root / image_path).resolve() if not Path(image_path).is_absolute() else Path(image_path).resolve()
    if not path.is_relative_to(root):
        return f"il file deve stare in {root}"
    if not path.is_file():
        return "file non trovato"
    try:
        with Image.open(path) as probe:
            if probe.width * probe.height > MAX_PIXELS:
                return "immagine troppo grande"
            probe.load()
            return probe.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        return f"immagine non leggibile: {type(exc).__name__}"


@server.tool(annotations=READ_ONLY)
def read_cai(image_path: str, received_on: str) -> dict[str, Any]:
    """Read one CAI form (JPEG/PNG inside the inbox folder). received_on: YYYY-MM-DD, when the insurer got it."""
    try:
        received = date.fromisoformat(received_on)
    except ValueError:
        return {"error": "received_on deve essere YYYY-MM-DD"}
    img = _open(image_path)
    if isinstance(img, str):
        return {"error": img}
    res = process(img, Path(image_path).stem, received)
    return {
        "route": res.route, "reasons": res.reasons,
        "deadline": res.deadline.model_dump(mode="json"),
        "issues": [i.model_dump() for i in res.issues],
        "fields": {k: {"value": r.value, "confidence": r.confidence, "agent": r.agent} for k, r in res.readings.items()},
    }


@server.tool(annotations=READ_ONLY)
def deadline_148(firma_A: Literal["si", "no", "incerto"], firma_B: Literal["si", "no", "incerto"],
                 feriti: Literal["si", "no", "incerto"], received_on: str) -> dict[str, Any]:
    """Term for the insurer's offer under art. 148 CAP from the facts of the form (uncertain -> shortest term)."""
    try:
        received = date.fromisoformat(received_on)
    except ValueError:
        return {"error": "received_on deve essere YYYY-MM-DD"}
    readings = {
        "A.firma": Reading(field="A.firma", value=firma_A, confidence=1, agent="mcp"),
        "B.firma": Reading(field="B.firma", value=firma_B, confidence=1, agent="mcp"),
        "feriti_si": Reading(field="feriti_si", value="si" if feriti == "si" else ("incerto" if feriti == "incerto" else "no"),
                             confidence=1, agent="mcp"),
        "feriti_no": Reading(field="feriti_no", value="si" if feriti == "no" else "no", confidence=1, agent="mcp"),
    }
    return rules.deadline(readings, received).model_dump(mode="json")


@server.tool(annotations=READ_ONLY)
def check_codice_fiscale(value: str) -> dict[str, Any]:
    """Shape and check character of an Italian codice fiscale (says nothing about whether the person exists)."""
    v = formats.compact("cf", value) or ""
    return {"value": v, "shape_ok": bool(formats.PATTERNS["cf"].fullmatch(v)), "check_char_ok": formats.cf_valid(v)}


@server.tool()
def make_synthetic_case(seed: int) -> dict[str, Any]:
    """Write a synthetic filled-in CAI form into the inbox and return its file name and ground truth."""
    truth, img = synth.generate(int(seed) % 10_000_000, "demo")
    root = inbox()
    root.mkdir(parents=True, exist_ok=True)
    name = f"{truth.case_id}.jpg"
    img.save(root / name, "JPEG", quality=88)
    return {"image_path": name, "received_on": truth.received_on.isoformat(), "truth": truth.model_dump(mode="json")}


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    args = ap.parse_args()
    if args.http:
        server.run("streamable-http", host="127.0.0.1", port=8766)
    else:
        server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
