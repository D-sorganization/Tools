from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "tools" / "launch_utils.py"
SPEC = importlib.util.spec_from_file_location("launch_utils_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launch_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch_utils)


def test_launch_octave_uses_configured_executable(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "run_me.m"
    script.write_text("disp('ok')\n", encoding="utf-8")
    monkeypatch.setenv("OCTAVE_EXECUTABLE", "octave-cli")

    with patch.object(launch_utils.subprocess, "Popen") as mock_popen:
        process = Mock()
        process.pid = 42
        mock_popen.return_value = process

        launch_utils.launch_octave_tool(script, "Octave Tool")

    called_args, called_kwargs = mock_popen.call_args
    assert called_args[0][0] == "octave-cli"
    assert called_args[0][1:3] == ["--quiet", "--eval"]
    assert "run('" in called_args[0][3]
    assert called_kwargs["cwd"] == script.parent
