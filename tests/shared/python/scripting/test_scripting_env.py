"""Tests for the interactive scripting environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.python.scripting.scripting_env import ConsoleEnvironment


def test_refresh_user_functions_propagates_system_level_errors(tmp_path: Path) -> None:
    """System-level failures in saved user code should not be hidden."""
    user_library = tmp_path / "user_library.py"
    user_library.write_text(
        "raise MemoryError('simulated exhaustion')\n", encoding="utf-8"
    )

    with pytest.raises(MemoryError, match="simulated exhaustion"):
        ConsoleEnvironment(user_lib_path=str(user_library))


def test_refresh_user_functions_reports_expected_user_code_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected user-code failures are reported without crashing the host."""
    user_library = tmp_path / "user_library.py"
    user_library.write_text(
        "raise ValueError('bad saved function')\n", encoding="utf-8"
    )

    ConsoleEnvironment(user_lib_path=str(user_library))

    captured = capsys.readouterr()
    assert "Error loading user library: bad saved function" in captured.err
