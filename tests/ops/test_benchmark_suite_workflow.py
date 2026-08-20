from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "benchmark-suite.yml"


def test_benchmark_suite_uses_an_isolated_virtual_environment() -> None:
    """A damaged shared pip installation must not contaminate benchmark jobs."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["benchmark-foundation"]["steps"]
    setup_index = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    )

    verify_step = steps[setup_index + 1]
    assert verify_step["name"] == "Verify Python runtime"
    assert "sys.version_info" in verify_step["run"]

    venv_step = steps[setup_index + 2]
    assert venv_step["name"] == "Create isolated benchmark virtual environment"
    assert 'python -m venv "$RUNNER_TEMP/benchmark-venv"' in venv_step["run"]
    assert (
        'echo "$RUNNER_TEMP/benchmark-venv/bin" >> "$GITHUB_PATH"' in venv_step["run"]
    )
    assert (
        'echo "VIRTUAL_ENV=$RUNNER_TEMP/benchmark-venv" >> "$GITHUB_ENV"'
        in venv_step["run"]
    )
    assert (
        '"$RUNNER_TEMP/benchmark-venv/bin/python" -m pip --version' in venv_step["run"]
    )


def test_benchmark_suite_does_not_repair_the_shared_interpreter_in_place() -> None:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m ensurepip --upgrade" not in workflow_text
    assert 'python -m pip install "numpy>=2.0.1"' in workflow_text
