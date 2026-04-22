"""Tests for web-launch helper extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from gui_launcher.launcher_web import launch_web_app, launch_web_from_gui_info


def test_launch_web_from_gui_info_uses_web_directory_next_to_caller(
    tmp_path: Path,
) -> None:
    """GUI_INFO web launch resolves the sibling web directory."""
    caller = tmp_path / "launch_web.py"
    caller.write_text("", encoding="utf-8")

    with patch("gui_launcher.launcher_web.launch_web_app") as mock_launch:
        mock_launch.return_value = 0
        result = launch_web_from_gui_info(
            {"name": "Data Tool", "web": {"port": 8080}},
            str(caller),
        )

    assert result == 0
    mock_launch.assert_called_once_with(
        tool_name="Data Tool",
        web_dir=tmp_path / "web",
        port=8080,
        auto_open_browser=True,
    )


@patch("gui_launcher.launcher_web._npm_executable")
def test_launch_web_app_fails_when_npm_is_unavailable(
    mock_npm: MagicMock,
    tmp_path: Path,
) -> None:
    """Unavailable Node/npm fails before any process is spawned."""
    mock_npm.return_value = None

    assert launch_web_app("Tool", tmp_path, auto_open_browser=False) == 1


@patch("gui_launcher.launcher_web.subprocess.Popen")
@patch("gui_launcher.launcher_web.subprocess.run")
@patch("gui_launcher.launcher_web._npm_executable")
def test_launch_web_app_starts_resolved_npm_command(
    mock_npm: MagicMock,
    mock_run: MagicMock,
    mock_popen: MagicMock,
    tmp_path: Path,
) -> None:
    """The dev server uses the resolved npm executable without a shell."""
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "node_modules").mkdir()
    mock_npm.return_value = "npm.cmd"
    process = MagicMock()
    process.wait.return_value = 0
    mock_popen.return_value = process
    process_started = MagicMock()

    result = launch_web_app(
        "Tool",
        web_dir,
        port=4321,
        auto_open_browser=False,
        env_vars={"EXTRA": "1"},
        process_started=process_started,
    )

    assert result == 0
    mock_run.assert_not_called()
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert args[0] == ["npm.cmd", "run", "dev"]
    assert kwargs["cwd"] == str(web_dir)
    assert kwargs["shell"] is False
    assert kwargs["env"]["PORT"] == "4321"
    assert kwargs["env"]["EXTRA"] == "1"
    process_started.assert_called_once_with(process)
