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


def test_heavy_workflows_report_coverage_without_global_floor() -> None:
    for workflow_path in HEAVY_WORKFLOWS:
        test_script = _step_script(workflow_path, "Run heavy integration tests")

        assert "--cov=." in test_script, workflow_path.name
        assert "--cov-report=xml:heavy_coverage.xml" in test_script, workflow_path.name
        assert "--cov-report=term-missing" in test_script, workflow_path.name
        assert "--cov-fail-under=0" in test_script, workflow_path.name


def test_heavy_workflows_preserve_pip_cache() -> None:
    for workflow_path in HEAVY_WORKFLOWS:
        steps = _workflow_steps(workflow_path)
        step_names = [step.get("name") for step in steps]
        assert "Purge pip cache" not in step_names, (
            f"{workflow_path.name} must not purge pip cache (Tools #5858)"
        )
        for step in steps:
            run_cmd = str(step.get("run", ""))
            assert "pip cache purge" not in run_cmd, (
                f"{workflow_path.name} step {step.get('name')} runs pip cache purge"
            )


def test_heavy_workflows_install_timeout_raised() -> None:
    for workflow_path in HEAVY_WORKFLOWS:
        steps = _workflow_steps(workflow_path)
        install_step = next(
            step for step in steps if step.get("name") == "Install Python dependencies"
        )
        timeout = int(install_step.get("timeout-minutes", 0))
        msg = (
            f"{workflow_path.name} Install Python dependencies timeout "
            f"must be >= 30 min (was {timeout})"
        )
        assert timeout >= 30, msg
