from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from tests.rust_bindings import _tools_core_import as tools_core_import

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
RUST_BINDING_CONTRACT = REPO_ROOT / "tests" / "rust_bindings" / "test_rust_bindings.py"


def _ci_standard_workflow() -> dict[str, Any]:
    return yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))


def _step_by_name(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(step for step in steps if step.get("name") == name)


def test_ci_tests_job_builds_and_installs_tools_core_wheel_in_required_lane() -> None:
    workflow = _ci_standard_workflow()
    tests_job = workflow["jobs"]["tests"]
    steps = tests_job["steps"]

    assert "3.11" in tests_job["strategy"]["matrix"]["python-version"]

    wheel_step = _step_by_name(steps, "Build and install tools_core Rust wheel")
    assert "matrix.python-version == '3.11'" in wheel_step["if"]

    wheel_script = wheel_step["run"]
    assert "maturin build" in wheel_script
    assert "rust_core/tools-core/Cargo.toml" in wheel_script
    assert "python -m pip install" in wheel_script
    assert "import tools_core" in wheel_script


def test_ci_tests_job_runs_non_skippable_tools_core_contract_in_required_lane() -> None:
    workflow = _ci_standard_workflow()
    run_tests = _step_by_name(
        workflow["jobs"]["tests"]["steps"],
        "Run Tests with Coverage (Python ${{ matrix.python-version }})",
    )

    test_script = run_tests["run"]
    assert "TOOLS_CORE_REQUIRED=1" in test_script
    assert "tests/rust_bindings/test_rust_bindings.py" in test_script


def test_rust_binding_contract_can_hard_fail_when_tools_core_is_required() -> None:
    source = RUST_BINDING_CONTRACT.read_text(encoding="utf-8")

    assert "import_required_tools_core" in source


def test_tools_core_import_helper_hard_imports_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("tools_core")

    def fake_import_module(name: str) -> ModuleType:
        assert name == "tools_core"
        return module

    monkeypatch.setenv(tools_core_import.TOOLS_CORE_REQUIRED_ENV, "1")
    monkeypatch.setattr(
        tools_core_import.importlib, "import_module", fake_import_module
    )

    assert tools_core_import.import_required_tools_core() is module


def test_tools_core_import_helper_importorskips_when_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("tools_core")

    def fake_importorskip(name: str, *, reason: str) -> ModuleType:
        assert name == "tools_core"
        assert reason == tools_core_import.TOOLS_CORE_MISSING_REASON
        return module

    monkeypatch.delenv(tools_core_import.TOOLS_CORE_REQUIRED_ENV, raising=False)
    monkeypatch.setattr(tools_core_import.pytest, "importorskip", fake_importorskip)

    assert tools_core_import.import_required_tools_core() is module
