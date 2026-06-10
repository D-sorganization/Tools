"""Tests for ShellTool allowlisted command execution (issue #3176).

Verifies that:
- An allowlisted command runs end-to-end and returns real output with
  ``return_code == 0`` (regression for the empty-executable bug that made
  every invocation raise FileNotFoundError).
- The allow/deny matrix is enforced: allowlisted commands pass; denied
  commands and any command containing a shell operator are rejected with
  ``success=False`` and are never executed.
- ShellTool runs the argv directly (``shell=False``) without an empty
  executable or a ``-c`` wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.shared.python.ai.tools.cli_tools import ShellTool  # noqa: E402


@pytest.fixture()
def tool(tmp_path: Path) -> ShellTool:
    """ShellTool whose working dir is a temp dir, default allowlist."""
    return ShellTool(working_dir=tmp_path)


# ---------------------------------------------------------------------------
# End-to-end: a real allowlisted command actually runs
# ---------------------------------------------------------------------------


class TestRealExecution:
    """A real allowlisted command must run and return output."""

    @pytest.mark.unit
    def test_allowlisted_command_runs_and_returns_output(self) -> None:
        """A custom-allowlisted real command produces output, rc==0.

        Uses the running Python interpreter (cross-platform) rather than a
        POSIX-only binary so the test runs on Windows CI too. The path is
        normalized to forward slashes so ``shlex.split`` (posix mode) does
        not strip backslashes on Windows.
        """
        exe = sys.executable.replace("\\", "/")
        tool = ShellTool(allowed_commands=[exe])
        result = tool.execute(f"{exe} -c 'print(42)'")

        assert result.success is True, result.error
        assert result.return_code == 0
        assert "42" in result.output

    @pytest.mark.unit
    def test_not_empty_executable_regression(self) -> None:
        """Regression: execute must not invoke an empty-string executable."""
        import subprocess as _sp

        exe = sys.executable.replace("\\", "/")
        tool = ShellTool(allowed_commands=[exe])
        with patch("subprocess.run", wraps=_sp.run) as run:
            tool.execute(f"{exe} --version")
        argv = run.call_args[0][0]
        assert argv[0] == exe  # not "" and not "-c"
        assert "-c" not in argv  # no shell -c wrapper


# ---------------------------------------------------------------------------
# Allow / deny matrix
# ---------------------------------------------------------------------------


class TestAllowDenyMatrix:
    """Parametrized allow/deny matrix for _is_command_allowed/execute."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "command",
        ["ls", "ls -la", "pwd", "cat file.txt", "head -n 5 x", "wc -l y"],
    )
    def test_allowlisted_commands_pass_gate(
        self, tool: ShellTool, command: str
    ) -> None:
        """Allowlisted base commands pass the allow gate."""
        assert tool._is_command_allowed(command) is True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",  # dangerous + not allowlisted
            "sudo ls",  # dangerous
            "ls; rm x",  # operator ;
            "ls && cat y",  # operator &&
            "ls | grep x",  # operator |
            "cat < f",  # redirection
            "echo $HOME",  # substitution
            "git status",  # not in allowlist
            "",  # empty
        ],
    )
    def test_denied_commands_rejected(self, tool: ShellTool, command: str) -> None:
        """Denied/operator/unknown commands are rejected and not executed."""
        assert tool._is_command_allowed(command) is False

        with patch("subprocess.run") as run:
            result = tool.execute(command)

        assert result.success is False
        run.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "command",
        ["rm -rf /", "ls; rm x", "ls && cat y"],
    )
    def test_acceptance_criteria_rejections(
        self, tool: ShellTool, command: str
    ) -> None:
        """Explicit AC examples from issue #3176 are rejected."""
        result = tool.execute(command)
        assert result.success is False
        assert "not allowed" in result.error.lower()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "command",
        [
            "ls /bin/rm",
            "ls ./rm",
            "ls --use-compress-program=rm",
            "ls --exec=/usr/bin/sudo",
        ],
    )
    def test_bypasses_rejected(self, tool: ShellTool, command: str) -> None:
        """Bypass attempts with paths or embedded dangerous commands are rejected."""
        assert tool._is_command_allowed(command) is False
