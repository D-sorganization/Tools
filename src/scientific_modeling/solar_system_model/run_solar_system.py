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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))


from solar_system.main import main

if __name__ == "__main__":
    sys.exit(main())
