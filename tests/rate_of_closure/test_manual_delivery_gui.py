"""PyQt controls for the manual delivery and shaft geometry contract."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.simulation import ShaftAxisDatum  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_manual_delivery_controls_are_discoverable_and_build_config(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = SimulationTab()
    qtbot.addWidget(tab)
    try:
        assert tab._manual_delivery_group.isEnabled()
        for widget in (*tab._manual_delivery_spins.values(), tab._shaft_datum_combo):
            assert "Suggested range" in widget.toolTip()
            assert "Source:" in widget.toolTip()
        assert tab._manual_delivery_spins["attack_angle_deg"].minimum() == -89.0
        assert tab._manual_delivery_spins["club_path_deg"].maximum() == 89.0
        assert tab._manual_delivery_spins["forward_shaft_lean_deg"].minimum() == -60.0
        assert tab._manual_delivery_spins["forward_shaft_lean_deg"].maximum() == 60.0

        tab._manual_delivery_spins["attack_angle_deg"].setValue(-10.0)
        tab._manual_delivery_spins["club_path_deg"].setValue(4.0)
        tab._manual_delivery_spins["forward_shaft_lean_deg"].setValue(15.0)
        datum_index = tab._shaft_datum_combo.findData(ShaftAxisDatum.GENERATED_HOSEL)
        tab._shaft_datum_combo.setCurrentIndex(datum_index)

        config = tab.config()
        assert config.manual_attack_angle_deg == -10.0
        assert config.manual_club_path_deg == 4.0
        assert config.manual_forward_shaft_lean_deg == 15.0
        assert config.manual_shaft_axis_datum is ShaftAxisDatum.GENERATED_HOSEL
    finally:
        tab.stop()


def test_manual_delivery_controls_enable_only_for_manual_source(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = SimulationTab()
    qtbot.addWidget(tab)
    try:
        assert tab._manual_delivery_group.isEnabled()
        tab._source_combo.setCurrentIndex(1)
        assert not tab._manual_delivery_group.isEnabled()
        tab._source_combo.setCurrentIndex(0)
        assert tab._manual_delivery_group.isEnabled()
    finally:
        tab.stop()
