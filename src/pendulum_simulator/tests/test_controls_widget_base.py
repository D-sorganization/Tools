from typing import Any

"""Tests for ControlsWidgetBase."""


from unittest.mock import MagicMock

from double_pendulum_golf.gui.controls_widget_base import ControlsWidgetBase
from double_pendulum_golf.gui.controls_widget import LabeledInput


class DummyControls(ControlsWidgetBase):
    PRESETS = {"Default": {}}

    def _build_model_sections(self, layout) -> Any:
        pass

    def _apply_preset(self, name) -> Any:
        pass

    def get_params(self) -> Any:
        return {}

    def _get_joint_names(self) -> Any:
        return ["JointA", "JointB"]

    def _get_torque_inputs(self) -> Any:
        if not hasattr(self, "ti_a"):
            self.ti_a = MagicMock()
            self.ti_b = MagicMock()
        return {"JointA": self.ti_a, "JointB": self.ti_b}

    def _update_torque_preview(self) -> Any:
        pass


def test_controls_base_construction(qapp) -> Any:
    w = DummyControls()

    # We must attach to prevent garbage collection of the boxes when returning
    from PyQt6.QtWidgets import QVBoxLayout

    lay = QVBoxLayout(w)

    # Test hidden widgets
    w._build_hidden_compat_widgets()
    assert hasattr(w, "btn_play")
    assert hasattr(w, "slider")

    # Torque clamps
    g1 = w._build_torque_clamp_section_ndof(["A", "B"], [10, 20])
    lay.addWidget(g1)
    assert w.chk_clamp is not None
    assert len(w.clamp_inputs) == 2

    # Test _parse_torque_limits
    assert w._parse_torque_limits() is None
    w.chk_clamp.setChecked(True)
    # mock values
    w.clamp_inputs[0].text_val = MagicMock(return_value="10")
    w.clamp_inputs[1].text_val = MagicMock(return_value="20")
    limits = w._parse_torque_limits()
    assert limits == [10.0, 20.0]

    # Joint limits
    g2 = w._build_joint_limits_section_ndof(["A", "B"], [-10, -20], [10, 20])
    lay.addWidget(g2)
    assert w.chk_limits is not None

    # Test _parse_joint_limits
    assert w._parse_joint_limits() is None
    w.chk_limits.setChecked(True)
    mins, maxs, k = w._parse_joint_limits()
    assert len(mins) == 2
    assert len(maxs) == 2
    assert k == 500.0

    # Other builders
    g3 = w._build_preset_section()
    lay.addWidget(g3)
    assert w.preset_combo is not None

    lyt = w._build_run_reset_buttons()
    lay.addLayout(lyt)
    assert w.btn_run is not None
    assert w.btn_reset is not None

    g4 = w._build_export_section()
    lay.addWidget(g4)
    assert w.btn_export_data is not None
    assert w.btn_export_video is not None
    assert w.btn_export_image is not None

    btn_funcgen = w._build_funcgen_button()
    # Label uses a BMP-range sine-wave symbol so it renders without an
    # emoji font on bare Linux/WSL.
    assert btn_funcgen.text() == "∿ Signal Toolkit…"

    # Expose inputs
    w._get_torque_inputs()


def test_open_function_generator(monkeypatch, qapp) -> Any:
    w = DummyControls()

    mock_dlg = MagicMock()

    def mock_init(self, *args, **kwargs) -> Any:
        pass

    import double_pendulum_golf.gui.function_generator_dialog as fgd

    monkeypatch.setattr(fgd.FunctionGeneratorDialog, "__init__", mock_init)
    monkeypatch.setattr(fgd.FunctionGeneratorDialog, "exec", mock_dlg)
    # create dummy signal property
    fgd.FunctionGeneratorDialog.torque_imported = MagicMock()

    w._open_function_generator()
    mock_dlg.assert_called_once()


def test_on_torque_imported(qapp) -> Any:
    w = DummyControls()
    inputs = w._get_torque_inputs()

    w._on_torque_imported("JointA", [1.0, 2.5])
    inputs["JointA"].set_value.assert_called_with("1, 2.5")


def test_playback_methods(qapp) -> Any:
    w = DummyControls()
    w._build_hidden_compat_widgets()
    w._build_gravity_section()

    play_emitted = False

    def on_play(checked) -> Any:
        nonlocal play_emitted
        play_emitted = checked

    w.play_toggled.connect(on_play)

    w._on_play_toggled(True)
    assert play_emitted is True

    w.set_slider_range(100)
    w.set_slider_value(50)
    assert w.slider.value() == 50
    assert w.slider.maximum() == 100

    w.stop_playback()
    assert not w.btn_play.isChecked()

    assert w.gravity_on() is True
    assert w.show_forces() is False


def test_uai_or_parse(qapp, monkeypatch) -> Any:
    DummyControls()
    li = LabeledInput("Test", "42.0", "")
    assert ControlsWidgetBase._uai_or_parse(li, "Test") == 42.0

    # Mock UnitAwareInput
    import sys

    class MockUAI:
        def value_si(self) -> Any:
            return 99.0

    mock_module = MagicMock()
    mock_module.UnitAwareInput = MockUAI
    monkeypatch.setitem(
        sys.modules, "upstream_drift_tools.ui.widgets.unit_aware_input", mock_module
    )

    uai = MockUAI()
    assert ControlsWidgetBase._uai_or_parse(uai, "Test UAI") == 99.0
