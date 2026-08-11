"""Strict regional execution evidence import and plan-binding tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.application.regional_execution_readback import (
    read_regional_execution_evidence,
    regional_execution_readback,
)
from shared.python.swing_sim.ground import (
    MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES,
    RegionalGroundExecutionResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "ground_regional_execution_golden_v1.json"
)


def _result(name: str = "representable") -> RegionalGroundExecutionResult:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return RegionalGroundExecutionResult.from_dict(payload[name]["result"])


def test_readback_reports_frozen_executor_evidence_without_running_physics() -> None:
    result = _result()

    readback = regional_execution_readback(result, result.regional_plan)

    assert readback.status == "partial"
    assert readback.plan_id == "regional-execution-plan-001"
    assert readback.surface_id == "firm-fairway"
    assert readback.termination_reason == "time_limit"
    assert readback.transition_count == 1
    assert readback.skid_distance_m == pytest.approx(0.0)
    assert readback.roll_distance_m == pytest.approx(0.25374857896)
    assert readback.total_distance_m == pytest.approx(0.2937485791)
    assert readback.executor_source_revision == "ground-regional-execution-v1"


def test_readback_rejects_evidence_for_a_different_visible_plan() -> None:
    result = _result()
    different = replace(result.regional_plan, request_id="different-plan")

    with pytest.raises(ValueError, match="does not match the current regional plan"):
        regional_execution_readback(result, different)


def test_file_read_is_bounded_strict_and_plan_bound(tmp_path: Path) -> None:
    result = _result()
    target = tmp_path / "execution.json"
    target.write_text(result.to_json(), encoding="utf-8")

    loaded = read_regional_execution_evidence(target, result.regional_plan)

    assert loaded.result == result
    assert loaded.readback.status == "partial"

    target.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        read_regional_execution_evidence(target, result.regional_plan)

    target.write_bytes(b" " * (MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES + 1))
    with pytest.raises(ValueError, match="maximum wire size"):
        read_regional_execution_evidence(target, result.regional_plan)


def test_pyqt_import_is_transactional_and_invalidated_by_plan_edit(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QWidget

    from rate_of_closure.ui.pyqt6 import regional_execution_evidence
    from rate_of_closure.ui.pyqt6.regional_execution_evidence import (
        RegionalExecutionEvidenceBox,
    )

    result = _result()
    target = tmp_path / "execution.json"
    target.write_text(result.to_json(), encoding="utf-8")
    monkeypatch.setattr(
        regional_execution_evidence.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(target), "JSON files (*.json)"),
    )

    class Host:
        def current_request(self):  # type: ignore[no-untyped-def]
            return result.regional_plan

    parent = QWidget()
    qtbot.addWidget(parent)
    box = RegionalExecutionEvidenceBox(Host(), parent)

    box.open_button.click()

    assert "partial" in box.readback_label.text()
    assert "No physics executed" in box.status_label.text()
    accepted = box.readback_label.text()
    target.write_text('{"request_id":"one","request_id":"two"}', encoding="utf-8")
    box.open_button.click()
    assert box.readback_label.text() == accepted
    assert "Prior accepted evidence was preserved" in box.status_label.text()
    box.clear()
    assert box.readback_label.text() == "No accepted evidence"
