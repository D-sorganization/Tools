# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Red-first regression tests for Tools #6130 (FC-005).

Verifies that:
- ``OptimizationResult.success`` strictly requires ``status == CONVERGED``.
- ``is_feasible`` remains True for feasible iteration-limited candidates.
- CLI summary/exit and ``export_excel`` report categorical status.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from movement_optimizer import cli
from movement_optimizer.export_excel import export_to_excel
from movement_optimizer.trajectory import (
    OptimizationResult,
    OptimizationStatus,
    populate_result_metadata,
)

from .conftest import make_test_result


def _make_valid_result(
    *, success: bool = True, status: OptimizationStatus = OptimizationStatus.CONVERGED
) -> OptimizationResult:
    base = make_test_result(seed=42)
    return OptimizationResult(
        t=base.t,
        q=base.q,
        qd=base.qd,
        qdd=base.qdd,
        torques=base.torques,
        power=base.power,
        com=base.com,
        bar=base.bar,
        success=success,
        cost=base.cost,
        com_horizontal_range_cm=base.com_horizontal_range_cm,
        status=status,
    )


def test_iteration_limit_feasible_candidate_has_success_false_and_is_feasible_true() -> None:
    result = _make_valid_result(success=True)
    res = SimpleNamespace(
        fun=5.0, success=False, status=9, message="Iteration limit reached", nit=10
    )
    populate_result_metadata(result, res)

    assert result.status is OptimizationStatus.ITERATION_LIMIT
    assert result.is_feasible is True
    assert result.success is False


def test_converged_feasible_candidate_has_success_true_and_is_feasible_true() -> None:
    result = _make_valid_result(success=True)
    res = SimpleNamespace(fun=5.0, success=True, status=0, message="Optimization terminated", nit=5)
    populate_result_metadata(result, res)

    assert result.status is OptimizationStatus.CONVERGED
    assert result.is_feasible is True
    assert result.success is True


def test_deprecation_warning_enforces_success_false_when_not_converged() -> None:
    with pytest.deprecated_call(match="CONVERGED"):
        result = _make_valid_result(success=True, status=OptimizationStatus.ITERATION_LIMIT)
    assert result.success is False


def test_cli_summary_and_exit_report_categorical_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _make_valid_result(success=True)
    res = SimpleNamespace(
        fun=5.0, success=False, status=9, message="Iteration limit reached", nit=10
    )
    populate_result_metadata(result, res)

    summary = cli._result_to_summary(result, "squat")
    assert summary["status"] == "ITERATION_LIMIT"
    assert summary["is_feasible"] is True
    assert summary["success"] is False

    fake_opt = SimpleNamespace(optimize=lambda: result)
    monkeypatch.setattr(cli, "_build_optimizer", lambda *args, **kwargs: (fake_opt, None))
    out_json = tmp_path / "summary.json"
    exit_code = cli.main(["--exercise", "squat", "--output", str(out_json)])
    assert exit_code == 1


def test_export_excel_summary_reports_categorical_status(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    result = _make_valid_result(success=True)
    res = SimpleNamespace(
        fun=5.0, success=False, status=9, message="Iteration limit reached", nit=10
    )
    populate_result_metadata(result, res)

    xlsx_path = tmp_path / "report.xlsx"
    export_to_excel(result, xlsx_path, exercise_name="squat")

    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["Summary"]
    rows = {row[0]: row[1] for row in ws.iter_rows(values_only=True) if row[0] is not None}
    assert rows["Status"] == "ITERATION_LIMIT"
    assert rows["Converged"] == "No"
    assert rows["Feasible"] == "Yes"
