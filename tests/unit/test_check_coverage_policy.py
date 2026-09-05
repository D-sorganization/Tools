"""Regression tests for the coverage policy gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_coverage_policy.py"
)


def _load_coverage_policy_module() -> Any:
    spec = importlib.util.spec_from_file_location("check_coverage_policy", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_tracked_packages_limits_package_thresholds(tmp_path: Path) -> None:
    """Changed-file scoping should not enforce unrelated package thresholds."""
    module = _load_coverage_policy_module()
    changed_files = tmp_path / "changed_python_files.txt"
    changed_files.write_text(
        "src/shared/python/sidekick/api/standard_response.py\n",
        encoding="utf-8",
    )

    tracked = {
        "src/shared/python/notes": 95.0,
        "src/shared/python/sidekick/calculators/conversion/service.py": 90.0,
        "src/shared/python/upstream_drift_tools": 100.0,
    }

    assert module._changed_tracked_packages(changed_files, tracked) == set()


def test_changed_tracked_packages_matches_nested_package_paths(tmp_path: Path) -> None:
    """Package thresholds still apply when a tracked package path changes."""
    module = _load_coverage_policy_module()
    changed_files = tmp_path / "changed_python_files.txt"
    changed_files.write_text(
        "src/shared/python/upstream_drift_tools/api/standard_response.py\n",
        encoding="utf-8",
    )

    tracked = {
        "src/shared/python/notes": 95.0,
        "src/shared/python/sidekick/calculators/conversion/service.py": 90.0,
        "src/shared/python/upstream_drift_tools": 100.0,
    }

    assert module._changed_tracked_packages(changed_files, tracked) == {
        "src/shared/python/upstream_drift_tools"
    }


def test_changed_tracked_packages_matches_tracked_file_paths(tmp_path: Path) -> None:
    """File-level coverage ratchets should apply when the exact file changes."""
    module = _load_coverage_policy_module()
    changed_files = tmp_path / "changed_python_files.txt"
    changed_files.write_text(
        "src/shared/python/sidekick/calculators/conversion/service.py\n",
        encoding="utf-8",
    )

    tracked = {
        "src/shared/python/sidekick/calculators/conversion/service.py": 90.0,
        "src/shared/python/safe_pandas_eval.py": 99.0,
    }

    assert module._changed_tracked_packages(changed_files, tracked) == {
        "src/shared/python/sidekick/calculators/conversion/service.py"
    }


def test_parse_coverage_matches_source_relative_windows_paths(tmp_path: Path) -> None:
    """Coverage.py source roots plus short filenames still match policy paths."""
    module = _load_coverage_policy_module()
    source = str(Path.cwd() / "src" / "shared" / "python" / "upstream_drift_tools")
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        f"""<?xml version="1.0" ?>
<coverage line-rate="1">
  <sources>
    <source>{source}</source>
  </sources>
  <packages>
    <package name=".">
      <classes>
        <class name="__init__.py" filename="__init__.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )

    stats = module.parse_coverage(
        coverage_xml,
        ["src/shared/python/upstream_drift_tools"],
    )

    assert stats["package_percent"]["src/shared/python/upstream_drift_tools"] == 100.0


def test_effective_total_floor_does_not_lower_policy_target() -> None:
    """We exclude the 60% target from hard gating until reached."""
    module = _load_coverage_policy_module()

    assert module._effective_total_floor(min_total=60.0, baseline_total=15.0) == 15.0


def test_total_coverage_is_full_suite_gate_only() -> None:
    """Changed-file scoped PR runs should enforce touched packages, not total."""
    module = _load_coverage_policy_module()

    assert module._should_enforce_total_coverage(None) is True
    assert module._should_enforce_total_coverage(set()) is False
    assert (
        module._should_enforce_total_coverage(
            {"src/shared/python/upstream_drift_tools"}
        )
        is False
    )


def test_full_suite_nightly_enforces_repo_wide_coverage_policy() -> None:
    """The nightly full-suite lane must run the total-coverage ratchet."""
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "full-suite-nightly.yml").read_text(
        encoding="utf-8"
    )

    assert "--cov=." in workflow
    assert "--cov-report=xml:coverage.xml" in workflow
    assert "scripts/check_coverage_policy.py" in workflow
    assert "--output-json coverage_trend_full_suite.json" in workflow

    gate_block = workflow.split("name: Coverage Policy Gate", maxsplit=1)[1]
    assert "--changed-files" not in gate_block


def test_committed_baseline_does_not_undercut_policy_target() -> None:
    """The committed baseline should support ratcheting, not redefine the floor."""
    module = _load_coverage_policy_module()
    root = Path(__file__).resolve().parents[2]
    baseline = json.loads((root / "config" / "coverage_baseline.json").read_text())

    assert baseline["total_percent"] >= module.coverage_floor()


def test_coverage_floor_is_declared_once_in_pyproject() -> None:
    """Tools #4913: one floor, in pyproject; nothing else may declare one."""
    module = _load_coverage_policy_module()
    root = Path(__file__).resolve().parents[2]

    assert module.coverage_floor() > 0
    policy = json.loads((root / "config" / "coverage_policy.json").read_text())
    assert "minimum_total_percent" not in policy
    assert not (root / ".coveragerc").exists()
    ci_standard = (root / ".github" / "workflows" / "ci-standard.yml").read_text(
        encoding="utf-8"
    )
    assert "--cov-fail-under" not in ci_standard
    claude_md = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "coverage minimum**" not in claude_md


def test_policy_file_declaring_a_floor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_coverage_policy_module()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"minimum_total_percent": 60.0}), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"total_percent": 60.0}), encoding="utf-8")
    coverage = tmp_path / "coverage.xml"
    coverage.write_text(
        '<coverage line-rate="0.5"><sources><source>src</source></sources></coverage>',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_coverage_policy.py",
            "--coverage-file",
            str(coverage),
            "--policy-file",
            str(policy),
            "--baseline-file",
            str(baseline),
            "--output-json",
            str(tmp_path / "trend.json"),
        ],
    )

    assert module.main() == 1


def test_coverage_policy_tracks_safe_eval_files() -> None:
    policy_path = (
        Path(__file__).resolve().parents[2] / "config" / "coverage_policy.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert policy["tracked_packages"]["src/shared/python/safe_eval.py"] >= 99.0
    assert policy["tracked_packages"]["src/shared/python/safe_pandas_eval.py"] >= 99.0
    assert (
        policy["tracked_packages"][
            "src/shared/python/signal_toolkit/adaptive_filter.py"
        ]
        >= 95.0
    )
    assert (
        policy["tracked_packages"]["src/shared/python/file_watcher/_fallback.py"]
        >= 95.0
    )
    assert policy["tracked_packages"]["src/shared/python/upstream_drift_tools"] >= 100.0
