"""Unit and GUI tests for PyQt6 Club Tester tab (C6, H4)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.ui.pyqt6.accessibility_audit import (
    audit_visible_focusable_controls,
)
from rate_of_closure.ui.pyqt6.club_tester_models import (
    ClubTesterState,
    execute_club_tester_study,
    execute_heavy_hit_sweep,
)
from rate_of_closure.ui.pyqt6.club_tester_tab import ClubTesterTab

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_club_tester_model_execution() -> None:
    state = ClubTesterState(
        preset_club="Driver (10.5°)",
        head_mass_scale=1.1,
        loft_delta_deg=2.0,
        enable_heavy_hit=True,
    )
    result = execute_club_tester_study(state)
    assert result.document.document_id.startswith("driver")
    assert len(result.report.counterfactuals) == 1
    variant = result.report.counterfactuals[0]
    assert variant.delivered_loft_deg > result.report.baseline.delivered_loft_deg
    assert result.coupled_result is not None
    assert result.coupled_result.decoupling_fraction > 0.95
    assert result.rigid_shaft_ball_speed_mps is not None
    assert result.rigid_shaft_ball_speed_mps >= result.coupled_result.ball_speed_mps


def test_club_tester_heavy_hit_sweep_execution() -> None:
    state = ClubTesterState(preset_club="Driver (10.5°)")
    json_text = execute_heavy_hit_sweep(state)
    assert "golf_club.impact_coupling_report/1" in json_text
    assert "counterfactuals" in json_text


def test_club_tester_tab_gui_and_accessibility(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = ClubTesterTab()
    qtbot.addWidget(tab)
    tab.resize(1024, 700)
    tab.show()

    result = tab.last_result()
    assert result is not None
    assert result.report.baseline.ball_speed_mps > 40.0

    # Modify state and rerun
    tab._controls._loft_delta_spin.setValue(2.0)
    tab.run_now()
    updated = tab.last_result()
    assert updated is not None
    cf_loft = updated.report.counterfactuals[0].delivered_loft_deg
    base_loft = updated.report.baseline.delivered_loft_deg
    assert cf_loft > base_loft

    # Run heavy hit sweep
    tab.run_sweep()
    assert "Heavy hit sweep completed" in tab._status_lbl.text()

    # Audit visible focusable controls
    audit = audit_visible_focusable_controls(tab)
    assert audit.findings == ()
    assert audit.control_count > 0
