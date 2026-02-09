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

# Bootstrap imports for development mode (before pip install -e .)
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent))


from solar_system.main import main

if __name__ == "__main__":
    sys.exit(main())
