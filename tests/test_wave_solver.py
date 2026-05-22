import json
from unittest.mock import MagicMock, patch

from wave_solver import main, run_cmd


def test_run_cmd_success():
    # Test successful command execution
    res = run_cmd("echo Hello")
    assert res == "Hello"


def test_run_cmd_error():
    # Test failed command with ignore_err=True (non-zero exit code, check=False)
    res = run_cmd('python -c "import sys; sys.exit(1)"', ignore_err=True)
    assert res is None or res == ""


def test_run_cmd_error_raises():
    import subprocess

    import pytest

    # Test failed command with ignore_err=False raises CalledProcessError
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd('python -c "import sys; sys.exit(1)"', ignore_err=False)


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
        if isinstance(cmd, str) and "gh issue list" in cmd:
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
        elif isinstance(cmd, str) and "git status" in cmd:
            mock_res.stdout = ""
        else:
            mock_res.stdout = ""
        return mock_res

    mock_run.side_effect = run_side_effect

    main()

    # Verify that it tried to run Claude/Git for issue #1 and closed the duplicate
    called_cmds = [call[0][0] for call in mock_run.call_args_list]

    # Duplicate issue with same title should be closed
    assert any("gh issue close 1" in cmd for cmd in called_cmds)

    # We checkout branch
    assert any("checkout -b fix/a-n-issue-1" in cmd for cmd in called_cmds)

    # We run claude
    assert any("claude -p" in cmd for cmd in called_cmds)
