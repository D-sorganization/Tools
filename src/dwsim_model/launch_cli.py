#!/usr/bin/env python3
"""Standalone CLI launcher for DWSIM Gasification Model."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from dwsim_model.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
