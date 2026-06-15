"""Signal Processing Studio - Unified signal processing application.

This bridge extends only this package's search path so submodules resolve from
the canonical ``python/signal_processing_studio`` tree without mutating the
process-global ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path

_CANONICAL_PKG = Path(__file__).resolve().parent / "python" / "signal_processing_studio"
_canonical_str = str(_CANONICAL_PKG)
if _canonical_str not in __path__:
    __path__.insert(0, _canonical_str)
