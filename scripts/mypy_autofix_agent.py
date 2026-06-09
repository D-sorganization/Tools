"""Compatibility wrapper for the canonical mypy autofix agent.

The implementation lives in ``src.tools.mypy_autofix_agent`` so the installed
``mypy-autofix`` console script and the historical ``python scripts/...`` entry
point execute the same code path.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast


def _repo_root() -> Path:
    """Return the repository root for this checked-out compatibility script."""
    return Path(__file__).resolve().parents[1]


def _canonical_agent_path(repo_root: Path) -> Path:
    """Return the canonical implementation path after validating preconditions."""
    if not repo_root.is_dir():
        raise ValueError(f"Repository root is not a directory: {repo_root}")
    agent_path = repo_root / "src" / "tools" / "mypy_autofix_agent.py"
    if not agent_path.is_file():
        raise ValueError(f"Canonical mypy autofix agent is missing below: {repo_root}")
    return agent_path


def _load_canonical_main(agent_path: Path) -> Callable[[], int]:
    """Load the canonical CLI entrypoint without mutating ``sys.path``."""
    spec = importlib.util.spec_from_file_location(
        "_tools_mypy_autofix_agent_canonical", agent_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load canonical mypy autofix agent: {agent_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    package_main = getattr(module, "main", None)
    if not callable(package_main):
        raise TypeError(
            f"Canonical mypy autofix agent lacks a callable main: {agent_path}"
        )
    return cast(Callable[[], int], package_main)


def _canonical_main() -> int:
    """Load and run the canonical package entrypoint."""
    package_main = _load_canonical_main(_canonical_agent_path(_repo_root()))
    return package_main()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical mypy autofix CLI.

    Args:
        argv: Reserved for future programmatic invocation. The canonical CLI
            currently reads ``sys.argv`` directly.

    Raises:
        ValueError: If a custom argument sequence is supplied before the
            canonical CLI supports dependency-injected arguments.
    """
    if argv is not None:
        raise ValueError("Custom argv is not supported by the canonical CLI yet")
    return _canonical_main()


if __name__ == "__main__":
    sys.exit(main())
