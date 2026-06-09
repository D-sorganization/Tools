"""Movement Optimizer GUI for standalone use.

The source package lives under ``src/optimizer_gui/python`` so direct
launcher execution extends this package path at import time.
"""

from __future__ import annotations

from pathlib import Path

_NESTED_PACKAGE = Path(__file__).resolve().parent / "python" / "optimizer_gui"
if _NESTED_PACKAGE.exists():
    __path__.append(str(_NESTED_PACKAGE))
