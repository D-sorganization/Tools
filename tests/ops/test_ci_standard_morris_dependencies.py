from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"


def test_protected_python_lanes_install_morris_server_runtime() -> None:
    """Morris authority tests must collect in each protected Python lane."""
    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))

    for job_name in ("quality-gate", "tests"):
        install_step = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("name") == "Install Dependencies"
        )
        install_commands = install_step["run"]

        assert (
            "python -m pip install fastapi python-multipart uvicorn" in install_commands
        )
