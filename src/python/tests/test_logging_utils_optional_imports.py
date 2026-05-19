"""Regression tests for optional native dependency imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_logging_utils_import_does_not_import_torch() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join(
        [
            str(repo_root / "src" / "python" / "src"),
            str(repo_root / "src" / "data_processing" / "data_processor" / "python"),
        ]
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([pythonpath, env.get("PYTHONPATH", "")])
    script = (
        "import sys; "
        "import utils.logging_utils; "
        "raise SystemExit('torch' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=env,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_set_seeds_does_not_import_torch_by_default() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src" / "python" / "src"), env.get("PYTHONPATH", "")]
    )
    env.pop("TOOLS_ENABLE_TORCH_SEEDING", None)
    script = (
        "import sys; "
        "from utils.logging_utils import set_seeds; "
        "set_seeds(123); "
        "raise SystemExit('torch' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=env,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
