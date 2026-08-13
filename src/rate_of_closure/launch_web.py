#!/usr/bin/env python3
"""Launch the Rate of Closure Impact Explorer React web application."""

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

from rate_of_closure.gui_registration import GUI_INFO  # noqa: E402
from rate_of_closure.web_authority.runtime import start_authority  # noqa: E402
from shared.python.gui_launcher import launch_web_from_gui_info  # noqa: E402


def main() -> int:
    """Launch the isolated local authority and its proxied React client."""
    runtime = start_authority(source_root=_REPO_ROOT / "src")
    try:
        return int(
            launch_web_from_gui_info(
                GUI_INFO,
                __file__,
                env_vars=runtime.vite_environment,
            )
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    sys.exit(main())
