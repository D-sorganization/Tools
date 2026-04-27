#!/usr/bin/env python3
"""
Convenience script to run the Solar System Simulation.

Usage:
    python run_solar_system.py [options]

For full options, run:
    python run_solar_system.py --help
"""

import sys

from _bootstrap import bootstrap

_REPO_ROOT = bootstrap(__file__)


from solar_system.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
