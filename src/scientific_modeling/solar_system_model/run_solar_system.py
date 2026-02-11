#!/usr/bin/env python3
"""
Convenience script to run the Solar System Simulation.

Usage:
    python run_solar_system.py [options]

For full options, run:
    python run_solar_system.py --help
"""

import os
import sys
from pathlib import Path

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)


from solar_system.main import main

if __name__ == "__main__":
    sys.exit(main())
