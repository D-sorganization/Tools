from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import yaml

from tests.rust_bindings import _tools_core_import as tools_core_import

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
RUST_BINDING_CONTRACT = REPO_ROOT / "tests" / "rust_bindings" / "test_rust_bindings.py"


def _ci_standard_workflow() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8")))


def _step_by_name(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(step for step in steps if step.get("name") == name)


def test_ci_tests_job_caches_builds_and_verifies_tools_core_required_lane() -> None:
    workflow = _ci_standard_workflow()
    tests_job = workflow["jobs"]["tests"]
    steps = tests_job["steps"]

    assert "3.11" in tests_job["strategy"]["matrix"]["python-version"]

    restore_step = _step_by_name(steps, "Restore cached tools_core wheel")
    decide_step = _step_by_name(
        steps, "Decide whether the tools_core wheel must be built"
    )
    build_step = _step_by_name(steps, "Build tools_core Rust wheel")
    save_step = _step_by_name(steps, "Save tools_core wheel to cache")
    verify_step = _step_by_name(steps, "Install and verify tools_core wheel")

    assert "matrix.python-version == '3.11'" in restore_step["if"]
    assert restore_step["uses"] == "actions/cache/restore@v6"
    assert restore_step["with"]["path"] == "dist/tools-core-wheels"
    assert "hashFiles(" in restore_step["with"]["key"]
    assert "need_build=true" in decide_step["run"]
    assert "need_build == 'true'" in build_step["if"]
    assert "python -m pip install maturin" in build_step["run"]
    assert "--no-cache-dir maturin" not in build_step["run"]
    assert "maturin build" in build_step["run"]
    assert "rust_core/tools-core/Cargo.toml" in build_step["run"]
    assert save_step["uses"] == "actions/cache/save@v6"
    assert save_step["continue-on-error"] is True
    assert save_step["with"]["key"] == restore_step["with"]["key"]
    assert "matrix.python-version == '3.11'" in verify_step["if"]
    assert "python -m pip install dist/tools-core-wheels/*.whl" in verify_step["run"]
    assert "import tools_core" in verify_step["run"]


def test_ci_tests_job_runs_non_skippable_tools_core_contract_in_required_lane() -> None:
    workflow = _ci_standard_workflow()
    run_tests = _step_by_name(
        workflow["jobs"]["tests"]["steps"],
        "Run Tests with Coverage (Python ${{ matrix.python-version }})",
    )

    test_script = run_tests["run"]
    assert "TOOLS_CORE_REQUIRED=1" in test_script
    # The lane runs the whole tree as shards (Tools #4913); the contract file
    # must be claimed by a shard rather than named in the workflow.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ci_test_shards", REPO_ROOT / "scripts" / "ci_test_shards.py"
    )
    assert spec is not None and spec.loader is not None
    shards = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = shards
    spec.loader.exec_module(shards)
    assert any(
        shard.claims("tests/rust_bindings/test_rust_bindings.py")
        for shard in shards.SHARDS
    )


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
