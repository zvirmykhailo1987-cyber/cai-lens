# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Project home: evals/ and data/ live here. An installed package (Docker) sets CAI_HOME."""

from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
HOME = Path(os.environ.get("CAI_HOME", PKG.parents[1]))
DATA = Path(os.environ.get("CAI_DATA", HOME / "data"))
FONTS = PKG / "static" / "fonts"
