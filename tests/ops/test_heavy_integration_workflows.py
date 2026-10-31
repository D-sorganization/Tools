from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAVY_WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "heavy-integration-tests.yml",
    REPO_ROOT / ".github" / "workflows" / "heavy-tests-opt-in.yml",
)
HEAVY_REQUIREMENTS = REPO_ROOT / "requirements-heavy.txt"
HEAVY_TEST_DIRS = (
    REPO_ROOT / "tests" / "heavy_integration",
    REPO_ROOT / "tests" / "e2e",
)
# Import name -> distribution name where they differ.
_DIST_FOR_IMPORT = {"PyQt6": "pyqt6", "yaml": "pyyaml"}
# Pytest plugins the pyproject addopts / ini options need under --strict-config.
_REQUIRED_PLUGINS = (
    "pytest-asyncio",
    "pytest-benchmark",
    "pytest-cov",
    "pytest-qt",
    "pytest-timeout",
    "pytest-xdist",
)


def _workflow_steps(workflow_path: Path) -> list[dict[str, object]]:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    return workflow["jobs"]["run-heavy-tests"]["steps"]


def _step_script(workflow_path: Path, step_name: str) -> str:
    steps = _workflow_steps(workflow_path)
    return next(step["run"] for step in steps if step.get("name") == step_name)


def _heavy_requirement_names() -> set[str]:
    names = set()
    for line in HEAVY_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if requirement:
            names.add(re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].lower())
    return names


def _third_party_imports(test_dir: Path) -> set[str]:
    local = {"tests", "shared", "conftest"}
    found: set[str] = set()
    for source in test_dir.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                found.add(node.module.split(".")[0])
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "importorskip"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                found.add(str(node.args[0].value).split(".")[0])
    return found - set(sys.stdlib_module_names) - local - {"__future__"}


def test_heavy_requirements_cover_heavy_test_imports() -> None:
    """Every third-party import in the heavy dirs is in requirements-heavy.txt."""
    names = _heavy_requirement_names()
    for test_dir in HEAVY_TEST_DIRS:
        for module in sorted(_third_party_imports(test_dir)):
            dist = _DIST_FOR_IMPORT.get(module, module).lower()
            assert dist in names, f"{module} ({test_dir.name}) not in {names}"


def test_heavy_requirements_carry_strict_config_plugins() -> None:
    names = _heavy_requirement_names()
    for plugin in _REQUIRED_PLUGINS:
        assert plugin in names, plugin


def test_heavy_workflows_install_only_the_heavy_requirements() -> None:
    """The lane must not pull .[dev,test] or requirements.txt (~1 GB, #5881)."""
    for workflow_path in HEAVY_WORKFLOWS:
        install_script = _step_script(workflow_path, "Install Python dependencies")

        assert "-r requirements-heavy.txt" in install_script, workflow_path.name
        assert ".[dev" not in install_script, workflow_path.name
        assert "-r requirements.txt" not in install_script, workflow_path.name


def test_heavy_workflows_key_pip_cache_on_heavy_requirements() -> None:
    for workflow_path in HEAVY_WORKFLOWS:
        setup = next(
            step
            for step in _workflow_steps(workflow_path)
            if str(step.get("uses", "")).startswith("actions/setup-python")
        )
        with_block = setup["with"]

        assert with_block.get("cache") == "pip", workflow_path.name
        assert with_block.get("cache-dependency-path") == "requirements-heavy.txt", (
            workflow_path.name
        )


def test_heavy_workflows_verify_the_lane_imports() -> None:
    """importorskip deps must fail loudly when missing, not skip silently."""
    for workflow_path in HEAVY_WORKFLOWS:
        verify_script = _step_script(workflow_path, "Verify heavy dependencies")

        for module in ("trimesh", "ezdxf", "PyQt6", "pandas", "scipy"):
            assert f"'{module}" in verify_script, (workflow_path.name, module)


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
