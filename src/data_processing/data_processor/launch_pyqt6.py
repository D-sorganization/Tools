#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Data Processor."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap  # noqa: E402

bootstrap(__file__)

# The Data Processor package is nested under ``src/data_processing/data_processor/
# python`` (not directly under ``src``), so it needs its own roots on sys.path.
# The shared bridge is the single source of truth for that path setup.
from shared.python.sidekick.data_processing.embedding import (  # noqa: E402
    ensure_full_data_processor_on_path,
)

ensure_full_data_processor_on_path()

from shared.python.gui_launcher import make_pyqt6_launcher  # noqa: E402

if __name__ == "__main__":
    sys.exit(make_pyqt6_launcher("data_processing.data_processor.gui_registration"))
