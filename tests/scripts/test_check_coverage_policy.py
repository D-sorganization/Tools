"""Tests for scripts/check_coverage_policy.py.

Verifies coverage policy gate logic:
- ``parse_coverage`` correctly aggregates per-package and total coverage from
  a Cobertura XML report.
- ``_pct`` rounds line rates to two decimal places.
- ``main`` exits 0 when policy is satisfied and 1 when a threshold is breached,
  writing the trend JSON artifact either way.

This module is executed as a standalone script by CI, so we load it via
``importlib.util`` (mirroring ``tests/scripts/test_check_coverage_gates.py``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "check_coverage_policy.py"
    spec = importlib.util.spec_from_file_location(
        "tools_check_coverage_policy", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Synthetic coverage.xml fixtures ──────────────────────────────────────

_COVERAGE_XML_TEMPLATE = """<?xml version="1.0" ?>
<coverage line-rate="{line_rate}" version="7.0">
  <packages>
    <package name="pkg">
      <classes>
{classes}
      </classes>
    </package>
  </packages>
</coverage>
"""

_CLASS_TEMPLATE = """        <class filename="{filename}">
          <lines>
{lines}
          </lines>
        </class>"""


def _render_class(filename: str, hits_by_line: dict[int, int]) -> str:
    line_elems = "\n".join(
        f'            <line number="{n}" hits="{h}"/>'
        for n, h in sorted(hits_by_line.items())
    )
    return _CLASS_TEMPLATE.format(filename=filename, lines=line_elems)


def _write_coverage_xml(
    path: Path,
    line_rate: float,
    classes: list[tuple[str, dict[int, int]]],
) -> Path:
    rendered = "\n".join(_render_class(fn, lines) for fn, lines in classes)
    path.write_text(
        _COVERAGE_XML_TEMPLATE.format(line_rate=line_rate, classes=rendered),
        encoding="utf-8",
    )
    return path


# ── _pct ─────────────────────────────────────────────────────────────────


class TestPct:
    def test_converts_ratio_to_percent(self) -> None:
        module = _load_module()
        assert module._pct(0.25) == 25.0

    def test_rounds_to_two_decimals(self) -> None:
        module = _load_module()
        assert module._pct(0.123456) == 12.35

    def test_zero(self) -> None:
        module = _load_module()
        assert module._pct(0.0) == 0.0

    def test_one(self) -> None:
        module = _load_module()
        assert module._pct(1.0) == 100.0


# ── parse_coverage ───────────────────────────────────────────────────────


class TestParseCoverage:
    def test_returns_total_from_line_rate_attr(self, tmp_path: Path) -> None:
        module = _load_module()
        xml = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.4242,
            classes=[("pkg_a/mod.py", {1: 1, 2: 0})],
        )
        result = module.parse_coverage(xml, tracked_prefixes=[])
        assert result["total_percent"] == 42.42

    def test_aggregates_covered_lines_for_tracked_prefix(self, tmp_path: Path) -> None:
        module = _load_module()
        xml = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.5,
            classes=[
                ("pkg_a/mod1.py", {1: 1, 2: 1, 3: 0, 4: 0}),
                ("pkg_a/mod2.py", {1: 1, 2: 0}),
                ("pkg_b/mod.py", {1: 1, 2: 1, 3: 1}),
            ],
        )
        result = module.parse_coverage(xml, tracked_prefixes=["pkg_a", "pkg_b"])
        # pkg_a: 3/6 = 50%; pkg_b: 3/3 = 100%
        assert result["package_percent"] == {"pkg_a": 50.0, "pkg_b": 100.0}

    def test_tracked_prefix_with_no_matching_files_is_zero(
        self, tmp_path: Path
    ) -> None:
        module = _load_module()
        xml = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.25,
            classes=[("other/mod.py", {1: 1, 2: 0})],
        )
        result = module.parse_coverage(xml, tracked_prefixes=["missing"])
        assert result["package_percent"] == {"missing": 0.0}

    def test_empty_tracked_prefix_list(self, tmp_path: Path) -> None:
        module = _load_module()
        xml = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.1,
            classes=[("pkg/mod.py", {1: 1})],
        )
        result = module.parse_coverage(xml, tracked_prefixes=[])
        assert result == {"total_percent": 10.0, "package_percent": {}}

    def test_prefix_matches_using_startswith(self, tmp_path: Path) -> None:
        """Prefixes are matched via str.startswith, not regex or path components."""
        module = _load_module()
        xml = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.5,
            classes=[
                ("src/shared/notes/a.py", {1: 1, 2: 1}),
                ("src/shared/notes_alt/b.py", {1: 0, 2: 0}),
            ],
        )
        # Both filenames start with "src/shared/notes"
        result = module.parse_coverage(xml, tracked_prefixes=["src/shared/notes"])
        # covered=2 (from a.py), valid=4 total -> 50%
        assert result["package_percent"]["src/shared/notes"] == 50.0


# ── main ─────────────────────────────────────────────────────────────────


def _write_policy_and_baseline(
    tmp_path: Path,
    *,
    minimum_total_percent: float = 25.0,
    max_total_drop_percent: float = 2.0,
    tracked_packages: dict[str, float] | None = None,
    baseline_total: float = 25.08,
    baseline_packages: dict[str, float] | None = None,
) -> tuple[Path, Path]:
    policy = tmp_path / "policy.json"
    baseline = tmp_path / "baseline.json"
    policy.write_text(
        json.dumps(
            {
                "minimum_total_percent": minimum_total_percent,
                "max_total_drop_percent": max_total_drop_percent,
                "tracked_packages": tracked_packages or {},
            }
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            {
                "total_percent": baseline_total,
                "package_percent": baseline_packages or {},
            }
        ),
        encoding="utf-8",
    )
    return policy, baseline


class TestMain:
    def test_passes_when_total_meets_minimum(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _load_module()
        cov = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.30,
            classes=[("pkg/mod.py", {1: 1, 2: 1, 3: 0})],
        )
        policy, baseline = _write_policy_and_baseline(
            tmp_path, minimum_total_percent=25.0, baseline_total=29.0
        )
        out = tmp_path / "trend.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_coverage_policy.py",
                "--coverage-file",
                str(cov),
                "--policy-file",
                str(policy),
                "--baseline-file",
                str(baseline),
                "--output-json",
                str(out),
            ],
        )
        assert module.main() == 0
        captured = capsys.readouterr()
        assert "Coverage policy passed." in captured.out
        assert out.exists()
        trend = json.loads(out.read_text(encoding="utf-8"))
        assert trend["total_percent"] == 30.0

    def test_fails_when_total_below_minimum(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _load_module()
        cov = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.10,
            classes=[("pkg/mod.py", {1: 1, 2: 0, 3: 0, 4: 0})],
        )
        policy, baseline = _write_policy_and_baseline(
            tmp_path, minimum_total_percent=25.0
        )
        out = tmp_path / "trend.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_coverage_policy.py",
                "--coverage-file",
                str(cov),
                "--policy-file",
                str(policy),
                "--baseline-file",
                str(baseline),
                "--output-json",
                str(out),
            ],
        )
        assert module.main() == 1
        captured = capsys.readouterr()
        assert "Coverage policy failed" in captured.err
        assert "below minimum" in captured.err
        assert out.exists()  # trend is still written

    def test_fails_when_regression_exceeds_max_drop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _load_module()
        # Total = 26%, baseline = 30%, allowed drop = 2% -> 26 < 30-2 = 28 -> regression
        cov = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.26,
            classes=[("pkg/mod.py", {1: 1, 2: 0})],
        )
        policy, baseline = _write_policy_and_baseline(
            tmp_path,
            minimum_total_percent=10.0,
            max_total_drop_percent=2.0,
            baseline_total=30.0,
        )
        out = tmp_path / "trend.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_coverage_policy.py",
                "--coverage-file",
                str(cov),
                "--policy-file",
                str(policy),
                "--baseline-file",
                str(baseline),
                "--output-json",
                str(out),
            ],
        )
        assert module.main() == 1
        captured = capsys.readouterr()
        assert "regressed beyond allowed drop" in captured.err

    def test_fails_when_tracked_package_below_threshold(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _load_module()
        cov = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.5,
            classes=[("pkg_a/mod.py", {1: 1, 2: 0, 3: 0, 4: 0})],
        )
        # pkg_a actual = 25%, threshold = 50% -> fail
        policy, baseline = _write_policy_and_baseline(
            tmp_path,
            minimum_total_percent=10.0,
            tracked_packages={"pkg_a": 50.0},
        )
        out = tmp_path / "trend.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_coverage_policy.py",
                "--coverage-file",
                str(cov),
                "--policy-file",
                str(policy),
                "--baseline-file",
                str(baseline),
                "--output-json",
                str(out),
            ],
        )
        assert module.main() == 1
        captured = capsys.readouterr()
        assert "pkg_a" in captured.err
        assert "below threshold" in captured.err

    def test_tracked_package_reported_in_stdout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = _load_module()
        cov = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.6,
            classes=[("pkg_a/mod.py", {1: 1, 2: 1, 3: 1, 4: 0})],
        )
        policy, baseline = _write_policy_and_baseline(
            tmp_path,
            minimum_total_percent=10.0,
            tracked_packages={"pkg_a": 50.0},
            baseline_total=60.0,
        )
        out = tmp_path / "trend.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_coverage_policy.py",
                "--coverage-file",
                str(cov),
                "--policy-file",
                str(policy),
                "--baseline-file",
                str(baseline),
                "--output-json",
                str(out),
            ],
        )
        assert module.main() == 0
        captured = capsys.readouterr()
        assert "total:" in captured.out
        assert "pkg_a:" in captured.out

    def test_trend_json_written_on_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = _load_module()
        cov = _write_coverage_xml(
            tmp_path / "cov.xml",
            line_rate=0.05,
            classes=[("pkg/mod.py", {1: 0, 2: 0})],
        )
        policy, baseline = _write_policy_and_baseline(
            tmp_path, minimum_total_percent=25.0
        )
        out = tmp_path / "trend.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_coverage_policy.py",
                "--coverage-file",
                str(cov),
                "--policy-file",
                str(policy),
                "--baseline-file",
                str(baseline),
                "--output-json",
                str(out),
            ],
        )
        assert module.main() == 1
        trend = json.loads(out.read_text(encoding="utf-8"))
        assert trend["total_percent"] == 5.0
        assert "package_percent" in trend
