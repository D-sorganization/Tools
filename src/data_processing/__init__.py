"""Data processing tools namespace package.

Contains:
  - data_processor: Signal processing and time-series analysis tool
  - processor: Shared facade for UI-agnostic data processing
"""

from __future__ import annotations

from pathlib import Path

_SHARED_DATA_PROCESSING = (
    Path(__file__).resolve().parents[1] / "shared" / "python" / "data_processing"
)
if _SHARED_DATA_PROCESSING.is_dir():
    __path__.append(str(_SHARED_DATA_PROCESSING))
