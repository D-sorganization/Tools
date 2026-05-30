"""Regression coverage for Sidekick standard response imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from pytest import MonkeyPatch


def test_standard_response_imports_from_repo_package_path(
    monkeypatch: MonkeyPatch,
) -> None:
    """Importing via the repo package path does not need top-level path shims."""
    shared_python_path = str(Path.cwd() / "src" / "shared" / "python")
    monkeypatch.setattr(
        sys,
        "path",
        [path for path in sys.path if path != shared_python_path],
    )
    for module_name in list(sys.modules):
        if module_name == "compatibility" or module_name.startswith(
            "src.shared.python.sidekick"
        ):
            sys.modules.pop(module_name)

    module = importlib.import_module("src.shared.python.sidekick.api.standard_response")

    assert module.ErrorCode.INVALID_INPUT == "INVALID_INPUT"
