from typing import Any

"""Tests for ControlsWidgetGolfer."""

import pytest
from double_pendulum_golf.gui.controls_widget_golfer import ControlsWidgetGolfer
import double_pendulum_golf.gui.controls_widget_golfer as cwg


def test_controls_golfer_init_and_getters(qapp) -> Any:
    w = ControlsWidgetGolfer()

    assert len(w._get_joint_names()) == 7
    inputs = w._get_torque_inputs()
    assert len(inputs) == 7

    w._apply_preset("Unknown Preset")

    params = w.get_params()
    assert params["enable_clamp"] is False
    assert params["enable_limits"] is False


def test_controls_golfer_uai_ui(qapp, monkeypatch) -> Any:
    from PyQt6.QtWidgets import QWidget

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

    monkeypatch.setattr(cwg, "_HAS_UAI", True)
    monkeypatch.setattr(cwg, "UnitAwareInput", MockUAI, raising=False)

    w = ControlsWidgetGolfer()

    w._apply_preset("Address Position")
    params = w.get_params()
    assert params["m_club"] > 0


def test_controls_golfer_limits_and_clamps(qapp) -> Any:
    w = ControlsWidgetGolfer()
    # Check limit boxes
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
    assert len(params["torque_limits"]) == 7

    assert params["enable_limits"] is True
    assert params["limit_stiffness"] == 300.0


def test_invalid_grip(qapp) -> Any:
    w = ControlsWidgetGolfer()
    w.inp_L_club.set_value("1.0")
    w.inp_grip_right.set_value("1.5")
    with pytest.raises(ValueError):
        w.get_params()

    w.inp_grip_right.set_value("0.5")
    w.inp_grip_left.set_value("1.5")
    with pytest.raises(ValueError):
        w.get_params()
