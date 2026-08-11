"""Matched regional-surface editor and strict-contract adapter tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.application.regional_surface_plan import (
    MAX_EDITOR_REGIONS,
    illustrative_regional_surface_plan_draft,
    validate_regional_surface_plan_draft,
)
from shared.python.swing_sim.canonical_numeric_json import (
    MAX_CANONICAL_SAFE_INTEGER,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_illustrative_draft_delegates_to_regional_wire_contract() -> None:
    draft = illustrative_regional_surface_plan_draft()

    request = validate_regional_surface_plan_draft(draft)

    assert draft.calibration_kind == "unvalidated"
    assert request.schema_version == "ground-regional-material-plan-request/v1"
    assert request.unit_system == "SI"
    assert request.base_surface.surface_id == "illustrative-fairway"
    assert request.regions[0].region_id == "illustrative-rough-band"
    assert request.regions[0].lower_coordinate_m == pytest.approx(120.0)
    assert request.regions[0].upper_coordinate_m == pytest.approx(150.0)
    assert request.provenance.input_sha256 == (
        "2b3bf1b705bf86f5bf3cbe17970ddff63887410ad9f255200e5cfa31e5717db3"
    )


def test_editor_does_not_soften_strict_material_validation() -> None:
    draft = illustrative_regional_surface_plan_draft()
    invalid_base = replace(
        draft.base_surface,
        static_friction=0.2,
        kinetic_friction=0.3,
    )

    with pytest.raises(ValueError, match="kinetic_friction"):
        validate_regional_surface_plan_draft(replace(draft, base_surface=invalid_base))


def test_editor_region_limit_fails_before_wire_construction() -> None:
    draft = illustrative_regional_surface_plan_draft()
    repeated = tuple(
        replace(
            draft.regions[0],
            region_id=f"region-{index}",
            precedence=index,
            surface=replace(
                draft.regions[0].surface,
                surface_id=f"surface-{index}",
            ),
        )
        for index in range(MAX_EDITOR_REGIONS + 1)
    )

    with pytest.raises(ValueError, match=f"at most {MAX_EDITOR_REGIONS}"):
        validate_regional_surface_plan_draft(replace(draft, regions=repeated))


def test_pyqt_editor_exposes_warnings_units_and_validated_readback(qtbot) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    pytest.importorskip("pytestqt")
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )

    tab = RegionalSurfacePlanTab()
    qtbot.addWidget(tab)

    assert "illustrative" in tab.warning_label.text().lower()
    assert "unvalidated" in tab.calibration_combo.currentText().lower()
    assert tab.domain_upper.suffix() == " m"
    assert tab.region_count() == 1

    tab.validate_button.click()

    assert "validated" in tab.status_label.text().lower()
    assert "ground-regional-material-plan-request/v1" in tab.readback.toPlainText()
    assert '"unit_system":"SI"' in tab.readback.toPlainText()


def test_pyqt_editor_reports_invalid_interval_without_losing_draft(qtbot) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    pytest.importorskip("pytestqt")
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )

    tab = RegionalSurfacePlanTab()
    qtbot.addWidget(tab)
    row = tab.region_rows()[0]
    row.lower_coordinate.setValue(160.0)
    row.upper_coordinate.setValue(150.0)

    tab.validate_button.click()

    assert "lower_coordinate_m" in tab.status_label.text()
    assert tab.status_label.accessibleName() == "Regional surface plan validation error"
    assert tab.region_count() == 1


def test_pyqt_editor_invalidates_validated_readback_after_draft_change(qtbot) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    pytest.importorskip("pytestqt")
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )

    tab = RegionalSurfacePlanTab()
    qtbot.addWidget(tab)
    tab.validate_button.click()
    assert tab.readback.toPlainText()

    tab.add_button.click()

    assert tab.readback.toPlainText() == ""
    assert tab.status_label.text() == "Changes not validated"


def test_pyqt_open_applies_only_a_fully_valid_editor_request(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    from rate_of_closure.ui.pyqt6 import regional_surface_plan_io
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )

    request = validate_regional_surface_plan_draft(
        replace(illustrative_regional_surface_plan_draft(), request_id="opened-plan")
    )
    target = tmp_path / "opened.json"
    target.write_text(request.to_json(), encoding="utf-8")
    monkeypatch.setattr(
        regional_surface_plan_io.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(target), "JSON files (*.json)"),
    )
    tab = RegionalSurfacePlanTab()
    qtbot.addWidget(tab)

    tab.open_button.click()

    assert tab.request_id.text() == "opened-plan"
    assert tab.file_actions.recent_path == target
    assert "opened" in tab.status_label.text().lower()
    assert tab.readback.toPlainText() == request.to_json()


def test_pyqt_failed_open_rolls_back_editor_and_recent_path(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    from rate_of_closure.ui.pyqt6 import regional_surface_plan_io
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )

    target = tmp_path / "corrupt.json"
    target.write_text('{"request_id":"one","request_id":"two"}', encoding="utf-8")
    monkeypatch.setattr(
        regional_surface_plan_io.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(target), "JSON files (*.json)"),
    )
    tab = RegionalSurfacePlanTab()
    qtbot.addWidget(tab)
    before = tab.draft()

    tab.open_button.click()

    assert tab.draft() == before
    assert tab.file_actions.recent_path is None
    assert "open failed" in tab.status_label.text().lower()
    assert tab.status_label.accessibleName() == "Regional surface plan file error"


def test_pyqt_save_as_preserves_imported_request_bytes(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    from rate_of_closure.ui.pyqt6 import regional_surface_plan_io
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )

    draft = illustrative_regional_surface_plan_draft()
    request = validate_regional_surface_plan_draft(
        replace(
            draft,
            request_id="saved-plan",
            regions=(replace(draft.regions[0], precedence=MAX_CANONICAL_SAFE_INTEGER),),
        )
    )
    source = tmp_path / "source.json"
    destination = tmp_path / "copy.json"
    source.write_text(request.to_json(), encoding="utf-8")
    choices = iter(((str(source), "JSON files (*.json)"),))
    monkeypatch.setattr(
        regional_surface_plan_io.QFileDialog,
        "getOpenFileName",
        lambda *_args: next(choices),
    )
    monkeypatch.setattr(
        regional_surface_plan_io.QFileDialog,
        "getSaveFileName",
        lambda *_args: (str(destination), "JSON files (*.json)"),
    )
    tab = RegionalSurfacePlanTab()
    qtbot.addWidget(tab)

    tab.open_button.click()
    tab.save_button.click()

    assert destination.read_bytes() == source.read_bytes()
    assert tab.file_actions.recent_path == destination
    assert tab.current_request() == request


def test_pyqt_import_preserves_canonical_precision_and_large_si_values(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )

    draft = illustrative_regional_surface_plan_draft()
    request = validate_regional_surface_plan_draft(
        replace(
            draft,
            upper_coordinate_m=1_000_000_000_000.0,
            base_surface=replace(
                draft.base_surface,
                firmness_pa=0.00000000001,
                grass_height_m=123.12345678901,
                turf_density_kg_m3=20_000.12345678901,
            ),
        )
    )
    tab = RegionalSurfacePlanTab()
    qtbot.addWidget(tab)

    tab.apply_imported_request(request)

    assert tab.current_request() is request


def test_pyqt_cancelled_open_and_save_leave_status_and_recent_unchanged(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    from rate_of_closure.ui.pyqt6 import regional_surface_plan_io
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )

    monkeypatch.setattr(
        regional_surface_plan_io.QFileDialog,
        "getOpenFileName",
        lambda *_args: ("", ""),
    )
    monkeypatch.setattr(
        regional_surface_plan_io.QFileDialog,
        "getSaveFileName",
        lambda *_args: ("", ""),
    )
    tab = RegionalSurfacePlanTab()
    qtbot.addWidget(tab)
    status = tab.status_label.text()

    tab.open_button.click()
    tab.save_button.click()

    assert tab.status_label.text() == status
    assert tab.file_actions.recent_path is None
