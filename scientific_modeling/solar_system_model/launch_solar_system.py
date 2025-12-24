#!/usr/bin/env python3
"""One-click launcher for the Solar System Simulation.

Double-click this file (or run ``python launch_solar_system.py``) to start the
simulation with sensible defaults and preflight dependency checks. Use the
command-line flags on ``solar_system.main`` if you want deeper customization.
"""

from solar_system.launcher import launch_quickstart

if __name__ == "__main__":
    raise SystemExit(launch_quickstart())
