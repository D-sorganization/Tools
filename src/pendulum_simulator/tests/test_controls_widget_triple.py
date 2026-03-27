from typing import Any

"""Tests for ControlsWidgetTriple."""

from typing import Any

from PyQt6.QtWidgets import QWidget
from double_pendulum_golf.gui.controls_widget_triple import ControlsWidgetTriple
import double_pendulum_golf.gui.controls_widget_triple as cwt


def test_controls_triple_init_and_getters(qapp) -> Any:
    w = ControlsWidgetTriple()

    assert len(w._get_joint_names()) == 3
    inputs = w._get_torque_inputs()
    assert len(inputs) == 3

    w._apply_preset("Unknown Preset")

    params = w.get_params()
    assert params["enable_clamp"] is False
    assert params["enable_limits"] is False


def test_controls_triple_uai_ui(qapp, monkeypatch) -> Any:
    class MockUAI(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._val = kwargs.get("default_value", 0.0)

        @property
        def value(self) -> Any:
            return str(self._val)

        def value_si(self) -> Any:
            return self._val

        def set_value(self, val, is_si=False) -> Any:
            pass

    monkeypatch.setattr(cwt, "_HAS_UAI", True)
    monkeypatch.setattr(cwt, "UnitAwareInput", MockUAI, raising=False)

    w = ControlsWidgetTriple()

    w._apply_preset("Free Triple Pendulum")
    params = w.get_params()
    assert params["m1"] > 0


def test_controls_triple_limits_and_clamps(qapp) -> Any:
    w = ControlsWidgetTriple()

    w.chk_clamp.setChecked(True)
    for inp in w.clamp_inputs:
        inp.set_value("20")

    w.chk_limits.setChecked(True)
    for inp in w.limit_min_inputs:
        inp.set_value("-90")
    for inp in w.limit_max_inputs:
        inp.set_value("90")

    w.inp_limit_k.set_value("300")

    params = w.get_params()
    assert params["enable_clamp"] is True
    assert len(params["torque_limits"]) == 3

    assert params["enable_limits"] is True
    assert params["limit_stiffness"] == 300.0


def test_controls_triple_preview_error(qapp) -> Any:
    w = ControlsWidgetTriple()
    w.inp_tend.set_value("invalid")
    w._update_torque_preview()
    # It must silently catch ValueError and set duration to 2.0
