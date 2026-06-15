from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAVY_WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "heavy-integration-tests.yml",
    REPO_ROOT / ".github" / "workflows" / "heavy-tests-opt-in.yml",
)


def _workflow_steps(workflow_path: Path) -> list[dict[str, object]]:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    return workflow["jobs"]["run-heavy-tests"]["steps"]


def _step_script(workflow_path: Path, step_name: str) -> str:
    steps = _workflow_steps(workflow_path)
    return next(step["run"] for step in steps if step.get("name") == step_name)


def test_heavy_workflows_install_pytest_asyncio_for_strict_config() -> None:
    for workflow_path in HEAVY_WORKFLOWS:
        install_script = _step_script(workflow_path, "Install Python dependencies")

        assert "pytest-asyncio" in install_script, workflow_path.name


def test_heavy_workflows_collect_only_explicit_heavy_paths() -> None:
    for workflow_path in HEAVY_WORKFLOWS:
        test_script = _step_script(workflow_path, "Run heavy integration tests")

        assert '-m "live_simulation or e2e"' in test_script, workflow_path.name
        assert "tests/heavy_integration/" in test_script, workflow_path.name
        assert "tests/e2e/" in test_script, workflow_path.name
        assert "    tests/ \\" not in test_script, workflow_path.name
