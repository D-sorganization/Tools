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


def _write_coverage_xml(path: Path, line_rate: float) -> None:
    path.write_text(
        f"""<?xml version="1.0" ?>
<coverage line-rate="{line_rate}">
  <packages>
    <package name=".">
      <classes>
        <class name="m.py" filename="m.py">
          <lines>
            <line number="1" hits="1" />
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )


def _run_main(
    module: Any,
    tmp_path: Path,
    *,
    line_rate: float,
    minimum_total_percent: float,
    baseline_total_percent: float,
    max_total_drop_percent: float = 2.0,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    coverage_xml = tmp_path / "coverage.xml"
    _write_coverage_xml(coverage_xml, line_rate)

    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "minimum_total_percent": minimum_total_percent,
                "max_total_drop_percent": max_total_drop_percent,
                "tracked_packages": {},
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"total_percent": baseline_total_percent, "package_percent": {}}),
        encoding="utf-8",
    )
    output_json = tmp_path / "trend.json"

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "check_coverage_policy.py",
            "--coverage-file",
            str(coverage_xml),
            "--policy-file",
            str(policy),
            "--baseline-file",
            str(baseline),
            "--output-json",
            str(output_json),
        ],
    )
    return int(module.main())


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


def test_effective_floor_is_baseline_while_below_target() -> None:
    """While the baseline is below the target, the effective floor is the baseline."""
    module = _load_coverage_policy_module()
    # target 60, baseline 15 -> ratchet floor is the baseline, not the target.
    assert module._effective_total_floor(60.0, 15.0) == 15.0


def test_effective_floor_caps_at_target_once_baseline_reaches_it() -> None:
    module = _load_coverage_policy_module()
    assert module._effective_total_floor(60.0, 75.0) == 60.0


def test_coverage_below_effective_floor_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A controlled coverage XML below the ratchet floor must fail the gate."""
    module = _load_coverage_policy_module()
    rc = _run_main(
        module,
        tmp_path,
        line_rate=0.10,  # 10% < 15% baseline floor
        minimum_total_percent=60.0,
        baseline_total_percent=15.0,
        monkeypatch=monkeypatch,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "below effective minimum" in err


def test_coverage_above_effective_floor_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage at/above the ratchet floor but below the target still passes."""
    module = _load_coverage_policy_module()
    rc = _run_main(
        module,
        tmp_path,
        line_rate=0.20,  # 20% >= 15% floor, < 60% target -> ratchet allows it
        minimum_total_percent=60.0,
        baseline_total_percent=15.0,
        monkeypatch=monkeypatch,
    )
    assert rc == 0


def test_output_distinguishes_effective_and_target_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failure output names the effective floor and the target distinctly."""
    module = _load_coverage_policy_module()
    rc = _run_main(
        module,
        tmp_path,
        line_rate=0.05,
        minimum_total_percent=60.0,
        baseline_total_percent=15.0,
        monkeypatch=monkeypatch,
    )
    assert rc == 1
    err = capsys.readouterr().err
    # Effective floor (15.0) and target (60.0) are both reported, distinctly.
    assert "effective minimum 15.0%" in err
    assert "target 60.0%" in err


def test_config_baseline_below_target_is_internally_consistent() -> None:
    """The committed config must describe a coherent ratchet (baseline <= target)."""
    root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (root / "config" / "coverage_policy.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (root / "config" / "coverage_baseline.json").read_text(encoding="utf-8")
    )
    target = float(policy["minimum_total_percent"])
    floor = float(baseline["total_percent"])
    assert 0.0 <= floor <= target, (
        "ratchet baseline must be between 0 and the target; "
        f"got baseline={floor}, target={target}"
    )
