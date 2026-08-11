"""Matched regional-surface editor and strict-contract adapter tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.application.regional_surface_plan import (
    MAX_EDITOR_REGIONS,
    illustrative_regional_surface_plan_draft,
    validate_regional_surface_plan_draft,
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
