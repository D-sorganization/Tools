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
        assert all(not spin.isEnabled() for spin in tab._tilt_spins.values())
        tab._source_combo.setCurrentIndex(1)
        assert not tab._manual_delivery_group.isEnabled()
        assert all(spin.isEnabled() for spin in tab._tilt_spins.values())
        tab._source_combo.setCurrentIndex(0)
        assert tab._manual_delivery_group.isEnabled()
        assert all(not spin.isEnabled() for spin in tab._tilt_spins.values())
    finally:
        tab.stop()


def test_v5_import_config_export_preserves_precise_manual_delivery(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The native editor must not quantize a persisted delivery declaration."""
    tab = SimulationTab()
    qtbot.addWidget(tab)
    try:
        assert tab.run_now() is not None
        document = tab.inspector().run_document()
        document["parameters"]["manual_delivery"] = {
            "attack_angle_deg": -9.1535118584,
            "club_path_deg": 4.123456,
            "forward_shaft_lean_deg": 15.0,
            "shaft_axis_datum": "generated_hosel",
        }

        tab.inspector().load_settings_document(document)
        config = tab.config()
        assert config.manual_attack_angle_deg == pytest.approx(-9.153512, abs=5e-7)
        assert config.manual_club_path_deg == pytest.approx(4.123456, abs=5e-7)
        assert config.manual_forward_shaft_lean_deg == 15.0
        assert config.manual_shaft_axis_datum is ShaftAxisDatum.GENERATED_HOSEL

        assert tab.run_now() is not None
        exported = tab.inspector().run_document()
        assert exported["format"] == "rate_of_closure.simulation_run/5"
        assert exported["parameters"]["manual_delivery"] == {
            "attack_angle_deg": pytest.approx(-9.153512, abs=5e-7),
            "club_path_deg": pytest.approx(4.123456, abs=5e-7),
            "forward_shaft_lean_deg": 15.0,
            "shaft_axis_datum": "generated_hosel",
        }
    finally:
        tab.stop()
