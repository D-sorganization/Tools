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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solar_system.main import main

if __name__ == "__main__":
    sys.exit(main())
