import json
import sys
from unittest.mock import MagicMock, patch

from wave_solver import WaveConfig, main, run_cmd


def test_run_cmd_success():
    # Test successful command execution using portable sys.executable
    res = run_cmd([sys.executable, "-c", "print('Hello')"])
    assert res == "Hello"


def test_run_cmd_error():
    # Test failed command with ignore_err=True (non-zero exit code, check=False)
    res = run_cmd([sys.executable, "-c", "import sys; sys.exit(1)"], ignore_err=True)
    assert res is None or res == ""


def test_run_cmd_error_raises():
    import subprocess

    import pytest

    # Test failed command with ignore_err=False raises CalledProcessError
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd([sys.executable, "-c", "import sys; sys.exit(1)"], ignore_err=False)


@patch("subprocess.run")
def test_main_no_issues(mock_run):
    # Mock run_cmd to return None (no issues)
    mock_run.return_value = MagicMock(stdout="", returncode=0)
    # main should return early without doing anything else
    main()
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_main_invalid_json(mock_run):
    # Mock run_cmd to return invalid JSON
    mock_run.return_value = MagicMock(stdout="{invalid_json}", returncode=0)
    main()
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_main_with_issues(mock_run):
    # Mock run_cmd to simulate:
    # 1. gh issue list (returns JSON list)
    # 2. git status (returns "" to avoid further Git/Claude actions)
    # We want a mock side_effect for subprocess.run
    def run_side_effect(cmd, **kwargs):
        mock_res = MagicMock(returncode=0)
        cmd_list = list(cmd) if isinstance(cmd, (list, tuple)) else [str(cmd)]
        if len(cmd_list) >= 3 and cmd_list[:3] == ["gh", "issue", "list"]:
            mock_res.stdout = json.dumps(
                [
                    {
                        "number": 1,
                        "title": "[A-N Assessment] Fix LOD",
                        "body": "Issue body",
                    },
                    {"number": 2, "title": "Regular bug", "body": "Other body"},
                    {
                        "number": 1,
                        "title": "[A-N Assessment] Fix LOD",
                        "body": "Duplicate issue",
                    },
                ]
            )
        elif len(cmd_list) >= 2 and cmd_list[:2] == ["git", "status"]:
            mock_res.stdout = ""
        else:
            mock_res.stdout = ""
        return mock_res

    mock_run.side_effect = run_side_effect

    # Run main with allow_mutations=True so mutating git/gh commands execute
    config = WaveConfig(allow_mutations=True)
    main(config=config)

    called_cmds = [call[0][0] for call in mock_run.call_args_list]

    # Duplicate issue with same title should be closed
    assert any(cmd == ["gh", "issue", "close", "1"] for cmd in called_cmds)

    # We checkout branch
    expected_branch = ["git", "checkout", "-b", "fix/a-n-issue-1"]
    assert any(cmd == expected_branch for cmd in called_cmds)

    # We run claude
    assert any(len(cmd) >= 2 and cmd[:2] == ["claude", "-p"] for cmd in called_cmds)
