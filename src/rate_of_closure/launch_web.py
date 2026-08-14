#!/usr/bin/env python3
"""Launch the Rate of Closure Python-only production web companion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast


class _BootstrapModule(Protocol):
    """Minimum repository-bootstrap contract needed by this launcher."""

    def bootstrap(self, caller_file: str) -> Path:
        """Install repository-local import paths and return the root."""


def _load_bootstrap() -> _BootstrapModule:
    """Load the root bootstrap without mutating import paths in this script."""
    bootstrap_path = Path(__file__).resolve().parents[2] / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_tools_bootstrap", bootstrap_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bootstrap module from {bootstrap_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_BootstrapModule, module)


_REPO_ROOT = _load_bootstrap().bootstrap(__file__)

from rate_of_closure.web_companion.cli import main as companion_main  # noqa: E402


def main() -> int:
    """Launch the verified packaged bundle without Node or Vite at runtime."""
    return int(companion_main())


if __name__ == "__main__":
    sys.exit(main())
