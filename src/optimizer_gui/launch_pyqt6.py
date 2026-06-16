#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Adam Optimizer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast


class _BootstrapModule(Protocol):
    """Protocol for the dynamically loaded repository bootstrap module."""

    def bootstrap(self, caller_file: str) -> None:
        """Install repository-local import paths for a script entry point."""


def _load_bootstrap() -> _BootstrapModule:
    """Load the repository bootstrap module when launched as a script."""
    repo_root = Path(__file__).resolve().parents[2]
    bootstrap_path = repo_root / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_tools_bootstrap", bootstrap_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bootstrap module from {bootstrap_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_BootstrapModule, module)


bootstrap = _load_bootstrap().bootstrap

bootstrap(__file__)

from shared.python.gui_launcher import make_pyqt6_launcher  # noqa: E402


def main() -> int:
    """Launch the Movement Optimizer PyQt6 application."""
    return int(make_pyqt6_launcher("optimizer_gui.gui_registration"))


if __name__ == "__main__":
    sys.exit(main())
