#!/usr/bin/env python3
"""P&ID Generator — CLI launcher for the Tools monorepo.

Usage:
    python launch_cli.py --spec path/to/spec.yml --out output.dxf [--svg output.svg]
"""

from programmatic_pid.cli import main

if __name__ == "__main__":
    main()
