from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "check_coverage_gates.py"
    spec = importlib.util.spec_from_file_location(
        "tools_check_coverage_gates", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_reports_missing_coverage_file(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module()
    missing_path = tmp_path / "missing-coverage.json"

    monkeypatch.setattr(
        "sys.argv",
        ["check_coverage_gates.py", "--coverage-json", str(missing_path)],
    )

    result = module.main()

    output = capsys.readouterr().out
    assert result == 1
    assert "not found" in output


def test_main_reports_passing_gates(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module()
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(
        """
        {
          "files": {
            "src/shared/python/upstream_drift_tools/calculators/conversion/example.py": {
              "summary": {"covered_lines": 8, "num_statements": 10}
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["check_coverage_gates.py", "--coverage-json", str(coverage_path)],
    )

    result = module.main()

    output = capsys.readouterr().out
    assert result == 0
    assert "All coverage gates passed." in output
    assert "OK:" in output


def test_main_reports_gate_failures(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module()
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(
        """
        {
          "files": {
            "src/shared/python/upstream_drift_tools/calculators/conversion/example.py": {
              "summary": {"covered_lines": 3, "num_statements": 10}
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["check_coverage_gates.py", "--coverage-json", str(coverage_path)],
    )

    result = module.main()

    output = capsys.readouterr().out
    assert result == 1
    assert "COVERAGE GATE FAILURES:" in output
    assert "FAIL:" in output
