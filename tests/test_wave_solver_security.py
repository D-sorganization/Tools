"""Security tests for wave_solver.py (issue #3143).

Verifies:
- Command construction yields structured argv lists (no shell strings),
  so malicious issue text containing shell metacharacters is inert.
- Destructive/mutating actions are gated behind explicit opt-in and do not
  execute in the default dry-run mode.
- ``--dangerously-skip-permissions`` is off by default.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "wave_solver.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "wave_solver_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ws = _load_module()

MALICIOUS = '"; rm -rf / & whoami `id` $(touch pwned)\nnewline'


def test_claude_argv_is_list_not_string() -> None:
    """The Claude command is an argv list, not a shell string."""
    argv = ws.build_claude_argv("hello")
    assert isinstance(argv, list)
    assert argv[0] == "claude"
    assert "-p" in argv


def test_claude_argv_treats_malicious_prompt_as_inert() -> None:
    """Shell metacharacters in the prompt land in a single argv element."""
    argv = ws.build_claude_argv(MALICIOUS)
    assert MALICIOUS in argv  # exact element, unmodified, unquoted
    # No element is a concatenated shell command string.
    assert not any(" rm -rf " in a and a != MALICIOUS for a in argv)


def test_skip_permissions_off_by_default() -> None:
    """--dangerously-skip-permissions is opt-in only."""
    assert "--dangerously-skip-permissions" not in ws.build_claude_argv("x")
    optd = ws.build_claude_argv("x", skip_permissions=True)
    assert "--dangerously-skip-permissions" in optd


def test_pr_create_argv_keeps_title_inert() -> None:
    """A malicious issue title stays inside argv elements, not a shell str."""
    argv = ws.build_pr_create_argv(42, MALICIOUS)
    assert isinstance(argv, list)
    assert argv[:3] == ["gh", "pr", "create"]
    # The title appears verbatim as the element after --title.
    title_value = argv[argv.index("--title") + 1]
    assert MALICIOUS in title_value


def test_run_cmd_rejects_non_string_argv() -> None:
    """DbC: non-string argv entries raise TypeError."""
    with pytest.raises(TypeError):
        ws.run_cmd(["git", 123])  # type: ignore[list-item]


def test_run_cmd_rejects_empty_argv() -> None:
    """DbC: empty argv raises ValueError."""
    with pytest.raises(ValueError):
        ws.run_cmd([])


def test_mutating_command_is_blocked_in_dry_run() -> None:
    """A mutating command is NOT executed when allow_mutations is False."""
    config = ws.WaveConfig(allow_mutations=False)
    with patch.object(ws.subprocess, "run") as mock_run:
        result = ws.run_cmd(["git", "reset", "--hard"], mutating=True, config=config)
    mock_run.assert_not_called()
    assert result is None


def test_mutating_command_runs_when_allowed() -> None:
    """A mutating command executes via argv list when explicitly allowed."""
    config = ws.WaveConfig(allow_mutations=True)
    with patch.object(ws.subprocess, "run") as mock_run:
        mock_run.return_value.stdout = "ok"
        ws.run_cmd(["git", "reset", "--hard"], mutating=True, config=config)
    mock_run.assert_called_once()
    called_args, called_kwargs = mock_run.call_args
    # First positional arg is the argv list; shell must be False.
    assert called_args[0] == ["git", "reset", "--hard"]
    assert called_kwargs.get("shell") is False


def test_non_mutating_read_runs_in_dry_run() -> None:
    """A read-only command still executes in dry-run mode."""
    config = ws.WaveConfig(allow_mutations=False)
    with patch.object(ws.subprocess, "run") as mock_run:
        mock_run.return_value.stdout = "data"
        result = ws.run_cmd(["gh", "issue", "list"], config=config)
    mock_run.assert_called_once()
    assert result == "data"


def test_no_shell_true_anywhere_in_source() -> None:
    """Assert no live shell=True remains in wave_solver.py."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "shell=False" in source
