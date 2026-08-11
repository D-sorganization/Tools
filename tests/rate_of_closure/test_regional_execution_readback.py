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
    assert readback.surface_provider_id == "tools.planar-surface"
    assert readback.surface_provider_version == "1.0.0"
    assert readback.termination_reason == "time_limit"
    assert readback.ground_time_s == pytest.approx(1.155)
    assert readback.completed is False
    assert readback.transition_count == 1
    assert readback.carry_distance_m == pytest.approx(0.0)
    assert readback.bounce_air_distance_m == pytest.approx(0.04)
    assert readback.skid_distance_m == pytest.approx(0.0)
    assert readback.roll_distance_m == pytest.approx(0.25374857896)
    assert readback.surface_path_distance_m == pytest.approx(0.25374857896)
    assert readback.total_distance_m == pytest.approx(0.2937485791)
    assert readback.final_downrange_m == pytest.approx(0.2937485791)
    assert readback.final_offline_m == pytest.approx(0.0)
    assert readback.bounce_count == 1
    assert readback.calibration_kind == "literature"
    assert readback.calibration_id == "literature-default-2026-08"
    assert readback.calibration_source == "documented literature basis"
    assert readback.calibration_confidence == pytest.approx(0.6)
    assert readback.observed_phases == ("impact", "skid", "roll")
    assert readback.unit_system == "SI"
    assert len(readback.events) == 4
    assert readback.events[0].sequence == 0
    assert readback.events[0].event_type == "first_contact"
    assert readback.events[0].time_s == pytest.approx(1.005)
    assert readback.events[0].position_m == pytest.approx((0.0, 0.02135, 0.0))
    assert readback.events[0].velocity_before_m_s == pytest.approx((2.0, -0.1, 0.0))
    assert readback.events[0].angular_velocity_after_rad_s == pytest.approx(
        (0.0, 0.0, -93.67681498829)
    )
    assert len(readback.transitions) == 1
    assert readback.transitions[0].event_sequence == 3
    assert readback.transitions[0].from_region_id is None
    assert readback.transitions[0].to_region_id == "rough-band"
    assert readback.transitions[0].from_surface_id == "firm-fairway"
    assert readback.transitions[0].to_surface_id == "regional-rough"
    assert len(readback.warnings) == 4
    assert readback.warnings[-1].code == "CENSORED_ENDPOINT"
    assert readback.warnings[-1].severity == "warning"
    assert readback.executor_source_revision == "ground-regional-execution-v1"


def test_null_result_readback_does_not_fabricate_ground_metrics() -> None:
    result = _result("cancelled")

    readback = regional_execution_readback(result, result.regional_plan)

    assert readback.status == "cancelled"
    assert readback.failure_reason == "cancelled"
    assert readback.ground_time_s is None
    assert readback.completed is None
    assert readback.carry_distance_m is None
    assert readback.bounce_count is None
    assert readback.calibration_kind is None
    assert readback.calibration_id is None
    assert readback.calibration_source is None
    assert readback.calibration_confidence is None
    assert readback.observed_phases == ()
    assert readback.events == ()
    assert readback.transitions == ()
    assert readback.warnings == ()


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

    assert "partial" in box.readback_label.toPlainText()
    assert "bounce 0.040 m" in box.readback_label.toPlainText()
    assert "final offline 0.000 m" in box.readback_label.toPlainText()
    assert "CENSORED_ENDPOINT" in box.readback_label.toPlainText()
    assert "units: SI" in box.readback_label.toPlainText()
    assert box.event_table.rowCount() == 4
    assert box.event_table.item(0, 1).text() == "first_contact"
    assert box.event_table.item(0, 2).text() == "1.005000"
    assert box.event_table.item(0, 3).text() == "(0.000000, 0.021350, 0.000000)"
    assert box.transition_table.rowCount() == 1
    assert box.transition_table.item(0, 3).text() == "base / firm-fairway"
    assert box.transition_table.item(0, 4).text() == "rough-band / regional-rough"
    readback = regional_execution_readback(result, result.regional_plan)
    many_events = tuple(
        replace(readback.events[0], sequence=index) for index in range(257)
    )
    regional_execution_evidence._populate_event_table(box.event_table, many_events)
    assert box.event_table.rowCount() == 256
    assert regional_execution_evidence._ledger_summary("Events", 257) == (
        "Events: showing first 256 of 257 validated rows."
    )
    assert "No physics executed" in box.status_label.text()
    accepted = box.readback_label.toPlainText()
    target.write_text('{"request_id":"one","request_id":"two"}', encoding="utf-8")
    box.open_button.click()
    assert box.readback_label.toPlainText() == accepted
    assert "Prior accepted evidence was preserved" in box.status_label.text()
    box.clear()
    assert box.readback_label.toPlainText() == "No accepted evidence"
    assert box.event_table.rowCount() == 0
    assert box.transition_table.rowCount() == 0
