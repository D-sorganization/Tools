from typing import Any

"""Tests for ControlsWidget."""

from typing import Any

from PyQt6.QtWidgets import QWidget
from double_pendulum_golf.gui.controls_widget import ControlsWidget, LabeledInput, _row
import double_pendulum_golf.gui.controls_widget as cw


def test_labeled_input(qapp) -> Any:
    li = LabeledInput("Label", "Val", "Tooltip")
    assert li.value == "Val"
    li.set_value("NewVal")
    assert li.value == "NewVal"


def test_row(qapp) -> Any:
    w1 = QWidget()
    w2 = QWidget()
    lyt = _row(w1, w2)
    assert lyt.count() == 2


def test_controls_init_and_getters(qapp) -> Any:
    w = ControlsWidget()
    assert len(w._get_joint_names()) == 2
    assert len(w._get_torque_inputs()) == 2

    # Sliders
    w.set_slider_range(100)
    assert w.lbl_frame.text() == "Frame: 0 / 100"
    w.set_slider_value(50)
    assert "Frame: 50 / 100" in w.lbl_frame.text()

    # Play toggle
    w._on_play_toggled(True)
    assert "Pause" in w.btn_play.text()
    w._on_play_toggled(False)
    assert "Play" in w.btn_play.text()


def test_controls_uai_ui(qapp, monkeypatch) -> Any:
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

    monkeypatch.setattr(cw, "_HAS_UAI", True)
    monkeypatch.setattr(cw, "UnitAwareInput", MockUAI, raising=False)

    w = ControlsWidget()

    w._apply_preset("Golf Swing (passive wrist)")

    # edit signals
    tilt_emitted = []
    w.tilt_changed.connect(tilt_emitted.append)
    w._on_tilt_edited("30")
    w._on_tilt_edited("invalid")
    assert len(tilt_emitted) == 1

    azi_emitted = []
    w.azimuth_changed.connect(azi_emitted.append)
    w._on_azimuth_edited("45")
    w._on_azimuth_edited("invalid")
    assert len(azi_emitted) == 1

    params = w.get_params()
    assert params["m1"] > 0


def test_controls_preview_error(qapp) -> Any:
    w = ControlsWidget()
    w.inp_tend.set_value("invalid")
    w._update_torque_preview()

    w.chk_clamp.setChecked(True)
    w.inp_max_tau1.set_value("invalid")
    w.inp_max_tau2.set_value("invalid")
    w._update_torque_preview()


def test_controls_uai_or_parse(qapp, monkeypatch) -> Any:
    class MockUAI(QWidget):
        def value_si(self) -> Any:
            return 99.0

    monkeypatch.setattr(cw, "_HAS_UAI", True)
    monkeypatch.setattr(cw, "UnitAwareInput", MockUAI, raising=False)
    assert ControlsWidget._uai_or_parse(MockUAI(), "test") == 99.0
