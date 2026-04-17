"""Regression tests for the shared mypy autofix helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_mypy_autofix_agent():
    module_path = (
        Path(__file__).resolve().parents[2] / "src/tools/mypy_autofix_agent.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mypy_autofix_agent_under_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_file_lines_raises_when_safety_read_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mypy_autofix_agent()
    target = tmp_path / "target.py"
    target.write_text("print('old')\n", encoding="utf-8")

    original_read_text = Path.read_text

    def fail_target_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == target:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_target_read)

    with pytest.raises(OSError, match="Unable to verify safe write target") as exc_info:
        module.write_file_lines(str(target), ["print('new')\n"])

    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)
    assert original_read_text(target, encoding="utf-8") == "print('old')\n"
