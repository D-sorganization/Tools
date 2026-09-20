"""Tests for Rate of Closure frozen PyQt6 qualification and scientific parity.

Covers Issue #4382 and Epic #4377:
(a) PyInstaller specification and hooks without repo _bootstrap.py reliance.
(b) Windows one-folder PyQt6 artifact offscreen and interactive qualification.
(c) Qt/Matplotlib/SciPy collection, graceful optional-Rust capability messaging,
    bounded canonical simulation, Ground Study evidence save, clean exit,
    spaces/Unicode/unrelated-cwd behavior, artifact hygiene, and byte-identical
    job/result evidence versus source loopback and qualified companion.
(d) Explicit unsupported status for PyQt direct-worker restart recovery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rate_of_closure.packaging.entrypoint import (
    DIRECT_WORKER_RECOVERY_UNSUPPORTED_REASON,
    execute_ground_study_evidence,
    probe_frozen_capabilities,
    run_canonical_simulation,
    run_offscreen_smoke_test,
)
from rate_of_closure.packaging.qualify_frozen import (
    check_artifact_hygiene,
    qualify_frozen_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).parents[2].resolve()
_PACKAGING_DIR = _REPO_ROOT / "src" / "rate_of_closure" / "packaging"
_GOLDEN_RESULT_PATH = (
    _REPO_ROOT
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "regional_ground_execution_result_golden_v1.json"
)


def test_packaging_spec_file_declares_required_sections() -> None:
    spec_path = _PACKAGING_DIR / "rate_of_closure.spec"
    assert spec_path.is_file(), f"missing spec file: {spec_path}"
    content = spec_path.read_text(encoding="utf-8")

    assert "entrypoint.py" in content
    assert "rate_of_closure" in content
    assert "PyQt6" in content
    assert "matplotlib" in content
    assert "scipy" in content
    assert "numpy" in content
    assert "COLLECT" in content or "onedir" in content.lower()
    assert "RateOfClosureExplorer" in content


def test_packaging_hook_declares_data_and_hidden_imports() -> None:
    hook_path = _PACKAGING_DIR / "hooks" / "hook-rate_of_closure.py"
    assert hook_path.is_file(), f"missing hook file: {hook_path}"
    content = hook_path.read_text(encoding="utf-8")

    assert "collect_data_files" in content
    assert "collect_submodules" in content
    assert "rate_of_closure" in content


def test_probe_capabilities_visibly_unsupported_direct_worker_recovery() -> None:
    caps = probe_frozen_capabilities()
    assert isinstance(caps, dict)

    recovery = caps.get("direct_worker_restart_recovery")
    assert isinstance(recovery, dict)
    assert recovery.get("supported") is False
    assert recovery.get("status") == "unsupported"
    assert recovery.get("reason") == DIRECT_WORKER_RECOVERY_UNSUPPORTED_REASON


def test_probe_capabilities_messages_graceful_optional_rust() -> None:
    caps = probe_frozen_capabilities()
    assert isinstance(caps, dict)

    rust_info = caps.get("rust_capability")
    assert isinstance(rust_info, dict)
    assert "available" in rust_info
    assert isinstance(rust_info["available"], bool)
    assert "detail" in rust_info
    assert "backend" in rust_info
    if not rust_info["available"]:
        assert (
            "fallback" in rust_info["detail"].lower()
            or "reference" in rust_info["detail"].lower()
        )


def test_probe_capabilities_confirms_collected_models_and_packages() -> None:
    caps = probe_frozen_capabilities()
    assert isinstance(caps, dict)

    models = caps.get("collected_models")
    assert isinstance(models, dict)
    assert models.get("locus_capabilities") is True
    assert models.get("neural_capabilities") is True
    assert models.get("example_driver_head") is True

    packages = caps.get("package_versions")
    assert isinstance(packages, dict)
    assert "PyQt6" in packages
    assert "matplotlib" in packages
    assert "scipy" in packages
    assert "numpy" in packages


def test_entrypoint_canonical_simulation_is_bounded_and_consistent() -> None:
    sim_result = run_canonical_simulation()
    assert isinstance(sim_result, dict)
    assert sim_result.get("status") == "success"

    metrics = sim_result.get("metrics", {})
    assert isinstance(metrics, dict)
    assert "scenario_speed_mph" in metrics
    assert metrics["scenario_speed_mph"] == 113.0
    assert metrics["ball_speed_mph"] > 30.0
    assert "carry_distance_m" in metrics
    assert metrics["carry_distance_m"] > 5.0
    assert "impact_outcome" in sim_result
    assert sim_result["impact_outcome"] == "hit"


def test_entrypoint_ground_study_evidence_save_and_parity(tmp_path: Path) -> None:
    result_json_path = tmp_path / "regional-ground-execution-result.json"
    rows_csv_path = tmp_path / "regional-ground-execution-rows.csv"

    evidence = execute_ground_study_evidence(
        result_dest=result_json_path,
        csv_dest=rows_csv_path,
    )
    assert evidence["status"] == "succeeded"
    assert result_json_path.is_file()
    assert rows_csv_path.is_file()

    saved_text = result_json_path.read_text(encoding="utf-8")
    saved_doc = json.loads(saved_text)

    golden_doc = json.loads(_GOLDEN_RESULT_PATH.read_text(encoding="utf-8"))
    assert saved_doc == golden_doc["result"]
    assert evidence["dataset_sha256"] == golden_doc["dataset_sha256"]
    assert evidence["canonical_sha256"] == golden_doc["canonical_sha256"]

    csv_text = rows_csv_path.read_text(encoding="utf-8")
    assert "trial_index" in csv_text
    assert "metric.carry_distance" in csv_text


def test_artifact_hygiene_checker_detects_violations(tmp_path: Path) -> None:
    # Clean directory passes
    clean_dir = tmp_path / "clean_bundle"
    clean_dir.mkdir()
    (clean_dir / "RateOfClosureExplorer.exe").write_bytes(b"MZfake")
    (clean_dir / "_internal").mkdir()
    (clean_dir / "_internal" / "python314.dll").write_bytes(b"fake")

    report = check_artifact_hygiene(clean_dir)
    assert report.is_clean is True
    assert len(report.violations) == 0

    # Directory with leaks fails
    dirty_dir = tmp_path / "dirty_bundle"
    dirty_dir.mkdir()
    (dirty_dir / "RateOfClosureExplorer.exe").write_bytes(b"MZfake")
    (dirty_dir / ".env").write_text("SECRET=123", encoding="utf-8")
    (dirty_dir / "secret.sqlite3").write_bytes(b"sqlite")
    (dirty_dir / "leaked_source.py").write_text("print('leak')", encoding="utf-8")

    dirty_report = check_artifact_hygiene(dirty_dir)
    assert dirty_report.is_clean is False
    assert any(".env" in v.message for v in dirty_report.violations)
    assert any(".sqlite3" in v.message for v in dirty_report.violations)
    assert any("leaked_source.py" in v.message for v in dirty_report.violations)


def test_run_offscreen_smoke_test() -> None:
    exit_code = run_offscreen_smoke_test()
    assert exit_code == 0


def test_built_frozen_artifact_qualification() -> None:
    artifact_dir = _REPO_ROOT / "dist" / "RateOfClosureExplorer"
    exe_name = (
        "RateOfClosureExplorer.exe"
        if sys.platform == "win32"
        else "RateOfClosureExplorer"
    )
    if not (artifact_dir / exe_name).is_file():
        pytest.skip("Frozen artifact not built; skipping live binary qualification")

    report = qualify_frozen_artifact(artifact_dir, timeout_s=45.0)
    assert report["status"] == "qualified"
    assert report["hygiene"] == "passed"
    assert report["ground_study_parity"] == "verified"
    assert report["unicode_and_cwd_relocation"] == "verified"
