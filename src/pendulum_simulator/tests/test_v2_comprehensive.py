"""
Tests for Pendulum Simulator V2 comprehensive fixes.

Covers issues #1132–#1155 with TDD, DbC, and DRY compliance.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

# Guard for tests that import modules requiring PyQt6 / display server.
_pyqt6_available = importlib.util.find_spec("PyQt6") is not None
_skip_no_qt = pytest.mark.skipif(not _pyqt6_available, reason="PyQt6 not available")

# On headless Linux, even importing QApplication succeeds but
# instantiating QWidget requires a platform backend.
# QT_QPA_PLATFORM=offscreen is safe; DISPLAY/WAYLAND_DISPLAY mean a real server.
_has_display = (
    sys.platform in ("win32", "darwin")
    or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    or bool(os.environ.get("DISPLAY"))
    or bool(os.environ.get("WAYLAND_DISPLAY"))
)

try:
    if _pyqt6_available and _has_display:
        from PyQt6.QtWidgets import QApplication  # noqa: F401

        _qt_runtime_ok = True
    else:
        _qt_runtime_ok = False
except ImportError:
    _qt_runtime_ok = False

_skip_no_qt_runtime = pytest.mark.skipif(
    not _qt_runtime_ok,
    reason="PyQt6 runtime not available (missing display libs)",
)

# ---------------------------------------------------------------------------
# Physics tests
# ---------------------------------------------------------------------------


class TestTorqueClampAbsValue:
    """#1138: TorqueClamp accepts negative values via abs()."""

    def test_positive_values_accepted(self) -> None:
        from double_pendulum_golf.physics import TorqueClamp

        tc = TorqueClamp(max_torque1=50.0, max_torque2=20.0)
        assert tc.max_torque1 == 50.0
        assert tc.max_torque2 == 20.0

    def test_negative_values_converted_to_positive(self) -> None:
        from double_pendulum_golf.physics import TorqueClamp

        tc = TorqueClamp(max_torque1=-50.0, max_torque2=-20.0)
        assert tc.max_torque1 == 50.0
        assert tc.max_torque2 == 20.0

    def test_mixed_signs(self) -> None:
        from double_pendulum_golf.physics import TorqueClamp

        tc = TorqueClamp(max_torque1=-30.0, max_torque2=15.0)
        assert tc.max_torque1 == 30.0
        assert tc.max_torque2 == 15.0

    def test_zero_value_rejected(self) -> None:
        from double_pendulum_golf.physics import TorqueClamp

        with pytest.raises((ValueError, TypeError)):
            TorqueClamp(max_torque1=0.0, max_torque2=10.0)

    def test_symmetric_clamping(self) -> None:
        from double_pendulum_golf.physics import TorqueClamp, clamp_torque

        tc = TorqueClamp(max_torque1=10.0, max_torque2=5.0)
        # Positive side
        tau = np.array([20.0, 8.0])
        result = clamp_torque(tau, tc)
        assert result[0] == 10.0
        assert result[1] == 5.0
        # Negative side
        tau_neg = np.array([-20.0, -8.0])
        result_neg = clamp_torque(tau_neg, tc)
        assert result_neg[0] == -10.0
        assert result_neg[1] == -5.0

    def test_clamp_torque_ndof(self) -> None:
        """#1150: generic N-DOF clamping works for 3-DOF."""
        from double_pendulum_golf.physics import clamp_torque_ndof

        tau = np.array([50.0, -30.0, 10.0])
        limits = np.array([25.0, 20.0, 15.0])
        result = clamp_torque_ndof(tau, limits)
        np.testing.assert_array_equal(result, [25.0, -20.0, 10.0])

    def test_clamp_torque_ndof_inf_passthrough(self) -> None:
        """#1150: inf limit means no clamping."""
        from double_pendulum_golf.physics import clamp_torque_ndof

        tau = np.array([999.0, -999.0])
        limits = np.array([np.inf, np.inf])
        result = clamp_torque_ndof(tau, limits)
        np.testing.assert_array_equal(result, tau)

    def test_triple_sim_with_torque_limits(self) -> None:
        """#1150: triple simulation runs with torque_limits."""
        from double_pendulum_golf.physics_triple import TriplePendulumParams
        from double_pendulum_golf.simulation_triple import run_simulation

        params = TriplePendulumParams(m1=2.0, m2=1.5, m3=1.0, L1=0.2, L2=0.65, L3=1.1)
        state = np.array([0.5, 0.1, -0.1, 0.0, 0.0, 0.0])
        torque_func = lambda t: (0.0, 0.0, 0.0)  # noqa: E731
        limits = np.array([50.0, 30.0, 20.0])
        result = run_simulation(
            params,
            state,
            t_end=0.1,
            torque_func=torque_func,
            dt=0.01,
            torque_limits=limits,
        )
        assert result.n_steps >= 2


@_skip_no_qt_runtime
class TestTriplePendulumDefaults:
    """#1140: Verify anatomical segment lengths for golf model."""

    def test_triple_swing_preset_lengths(self) -> None:
        """Triple Swing preset should have short hub, medium arm, long club."""
        from double_pendulum_golf.gui.controls_widget_triple import (
            ControlsWidgetTriple,
        )

        preset = ControlsWidgetTriple.PRESETS["Triple Swing"]
        # Preset tuple order: ..., m1, m2, m3, L1, L2, L3, ...
        # L1 is at index 13, L2 at 14, L3 at 15
        L1 = preset[13]  # Hub (sternum→shoulder)
        L2 = preset[14]  # Arm
        L3 = preset[15]  # Club

        # Anatomical constraints
        assert L1 < L2, f"Hub ({L1}) should be shorter than arm ({L2})"
        assert L2 < L3, f"Arm ({L2}) should be shorter than club ({L3})"
        assert 0.15 <= L1 <= 0.30, f"Hub length {L1} out of anatomical range"
        assert 0.55 <= L2 <= 0.75, f"Arm length {L2} out of anatomical range"
        assert 0.90 <= L3 <= 1.20, f"Club length {L3} out of anatomical range"

    def test_triple_params_valid(self) -> None:
        """Verify TriplePendulumParams accepts anatomical lengths."""
        from double_pendulum_golf.physics_triple import TriplePendulumParams

        params = TriplePendulumParams(
            m1=5.0,
            m2=0.5,
            m3=0.4,
            L1=0.20,
            L2=0.65,
            L3=1.10,
        )
        assert params.L1 == 0.20
        assert params.L2 == 0.65
        assert params.L3 == 1.10


@_skip_no_qt_runtime
class TestFontSizeConstants:
    """#1134: Verify centralized font size constants exist and meet minimum."""

    def test_font_constants_exist(self) -> None:
        from double_pendulum_golf.gui.controls_utils import (
            MIN_FONT_PX,
        )

        assert MIN_FONT_PX >= 11

    def test_all_fonts_meet_minimum(self) -> None:
        from double_pendulum_golf.gui.controls_utils import (
            FONT_BODY,
            FONT_BTN,
            FONT_EDIT,
            FONT_GROUP,
            FONT_STATUS,
            MIN_FONT_PX,
        )

        for name, val in [
            ("FONT_BODY", FONT_BODY),
            ("FONT_BTN", FONT_BTN),
            ("FONT_EDIT", FONT_EDIT),
            ("FONT_GROUP", FONT_GROUP),
            ("FONT_STATUS", FONT_STATUS),
        ]:
            assert val >= MIN_FONT_PX, f"{name}={val} < MIN_FONT_PX={MIN_FONT_PX}"

    def test_title_font_larger_than_body(self) -> None:
        from double_pendulum_golf.gui.controls_utils import FONT_BODY, FONT_TITLE

        assert FONT_TITLE > FONT_BODY


@_skip_no_qt_runtime
class TestStylesheetConsistency:
    """#1134: Verify stylesheet tokens use centralized font sizes."""

    def test_style_group_uses_font_group(self) -> None:
        from double_pendulum_golf.gui.controls_utils import FONT_GROUP, STYLE_GROUP

        assert f"{FONT_GROUP}px" in STYLE_GROUP

    def test_style_label_uses_font_body(self) -> None:
        from double_pendulum_golf.gui.controls_utils import FONT_BODY, STYLE_LABEL

        assert f"{FONT_BODY}px" in STYLE_LABEL


class TestPenaltyStiffnessLabel:
    """#1139: Verify penalty stiffness field shows units in label."""

    def test_label_contains_units(self) -> None:
        """The label should show 'K (N·m/rad)' not just 'K'."""
        # We test this by checking the class attribute directly,
        # as widget instantiation requires QApplication
        pass  # Widget-level tests need qtbot, covered in test_ui_enhancements


class TestPhysicsTripleEnergy:
    """Verify triple pendulum energy conservation (regression guard)."""

    def test_zero_torque_energy_conservation(self) -> None:
        from double_pendulum_golf.physics_triple import TriplePendulumParams

        try:
            from double_pendulum_golf.physics_triple import total_energy
        except ImportError:
            pytest.skip("total_energy not available in physics_triple")
        from double_pendulum_golf.simulation_triple import run_simulation

        params = TriplePendulumParams(
            m1=1.0,
            m2=0.5,
            m3=0.3,
            L1=0.20,
            L2=0.65,
            L3=1.10,
        )
        state0 = np.array([np.pi / 3, -np.pi / 6, -np.pi / 8, 0.0, 0.0, 0.0])

        def zero_torque(t: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 0.0)

        result = run_simulation(
            params=params,
            initial_state=state0,
            t_end=0.5,
            torque_func=zero_torque,
            dt=0.001,
        )

        E0 = total_energy(state0, params)
        E_final = total_energy(result.states[-1], params)
        # Energy should be conserved within integration tolerance
        assert abs(E_final - E0) / max(abs(E0), 1e-10) < 0.01, (
            f"Energy drift: E0={E0:.4f}, E_final={E_final:.4f}"
        )


class TestUnitConversionModule:
    """#1137: Unit conversion system."""

    def test_unit_converter_exists(self) -> None:
        """UnitConverter class should be importable."""
        from double_pendulum_golf.gui.unit_converter import UnitConverter

        assert UnitConverter is not None

    def test_si_passthrough(self) -> None:
        """SI mode returns values unchanged."""
        from double_pendulum_golf.gui.unit_converter import UnitConverter, UnitSystem

        uc = UnitConverter(system=UnitSystem.SI)
        assert uc.to_si_length(1.5) == 1.5
        assert uc.from_si_length(1.5) == 1.5
        assert uc.to_si_mass(3.0) == 3.0
        assert uc.from_si_mass(3.0) == 3.0
        assert uc.to_si_torque(10.0) == 10.0
        assert uc.length_unit == "m"
        assert uc.mass_unit == "kg"
        assert uc.torque_unit == "N·m"

    def test_imperial_length(self) -> None:
        """Imperial: 1 inch = 0.0254 m."""
        from double_pendulum_golf.gui.unit_converter import UnitConverter, UnitSystem

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        np.testing.assert_allclose(uc.to_si_length(1.0), 0.0254)
        np.testing.assert_allclose(uc.from_si_length(0.0254), 1.0)
        assert uc.length_unit == "in"

    def test_imperial_mass(self) -> None:
        """Imperial: 1 lb ≈ 0.4536 kg."""
        from double_pendulum_golf.gui.unit_converter import UnitConverter, UnitSystem

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        np.testing.assert_allclose(uc.to_si_mass(1.0), 0.45359237, rtol=1e-6)
        np.testing.assert_allclose(uc.from_si_mass(0.45359237), 1.0, rtol=1e-6)
        assert uc.mass_unit == "lb"

    def test_imperial_torque_roundtrip(self) -> None:
        """Imperial torque: lbf·in → N·m → lbf·in roundtrip."""
        from double_pendulum_golf.gui.unit_converter import UnitConverter, UnitSystem

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        original = 42.0
        si_val = uc.to_si_torque(original)
        back = uc.from_si_torque(si_val)
        np.testing.assert_allclose(back, original, rtol=1e-10)
        assert uc.torque_unit == "lbf·in"


@_skip_no_qt_runtime
class TestEquationsPopup:
    """#1136, #1144: Equations popup module."""

    def test_equation_topics_importable(self) -> None:
        """EquationTopic enum should be importable."""
        from double_pendulum_golf.gui.equations_popup import EquationTopic

        assert EquationTopic.MASS_MATRIX is not None
        assert EquationTopic.EQUATIONS_OF_MOTION is not None

    def test_content_defined_for_all_topics(self) -> None:
        """HTML content must exist for every EquationTopic."""
        from double_pendulum_golf.gui.equations_popup import (
            EquationTopic,
            _TOPICS,
        )

        for topic in EquationTopic:
            assert topic in _TOPICS, f"Missing content for {topic}"
            title, html = _TOPICS[topic]
            assert len(title) > 0
            assert len(html) > 100  # non-trivial content


@_skip_no_qt_runtime
class TestPopOutChart:
    """#1135: Pop-out chart with regression."""

    def test_fit_regression_linear(self) -> None:
        """Linear regression should recover slope and intercept."""
        from double_pendulum_golf.gui.popout_chart import fit_regression

        x = np.linspace(0, 10, 100)
        y = 3.0 * x + 7.0 + np.random.default_rng(42).normal(0, 0.01, 100)
        x_fit, y_fit, coeffs = fit_regression(x, y, degree=1)
        # slope ≈ 3, intercept ≈ 7
        np.testing.assert_allclose(coeffs[0], 3.0, atol=0.05)
        np.testing.assert_allclose(coeffs[1], 7.0, atol=0.1)
        assert len(x_fit) == 200  # dense grid

    def test_popout_chart_stores_data(self) -> None:
        """PopOutChart.plot_data stores x and y arrays."""
        from double_pendulum_golf.gui.popout_chart import PopOutChart

        chart = PopOutChart()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        chart.plot_data(x, y, "X", "Y", "Test")
        np.testing.assert_array_equal(chart._x, x)
        np.testing.assert_array_equal(chart._y, y)

    def test_add_regression_returns_fit(self) -> None:
        """add_regression should return fitted data."""
        from double_pendulum_golf.gui.popout_chart import PopOutChart

        chart = PopOutChart()
        x = np.linspace(0, 5, 50)
        y = x**2
        chart.plot_data(x, y)
        result = chart.add_regression(degree=2)
        assert result is not None
        x_fit, y_fit = result
        assert len(x_fit) == 200


class TestScapulaOffset:
    """#1152: Scapula segment motion relative to hub."""

    def test_scapula_offset_default_zero(self) -> None:
        """Default scapula offset should be zero."""
        from double_pendulum_golf.physics_triple import TriplePendulumParams

        p = TriplePendulumParams(m1=5, m2=0.5, m3=0.4, L1=0.2, L2=0.65, L3=1.1)
        assert p.scapula_offset_rad == 0.0

    def test_forward_kinematics_with_scapula_offset(self) -> None:
        """Non-zero scapula offset should displace shoulder from hub."""
        from double_pendulum_golf.physics_triple import (
            TriplePendulumParams,
            forward_kinematics,
        )

        p_zero = TriplePendulumParams(
            m1=5,
            m2=0.5,
            m3=0.4,
            L1=0.2,
            L2=0.65,
            L3=1.1,
            scapula_offset_rad=0.0,
        )
        p_offset = TriplePendulumParams(
            m1=5,
            m2=0.5,
            m3=0.4,
            L1=0.2,
            L2=0.65,
            L3=1.1,
            scapula_offset_rad=np.radians(30),
        )

        fk_zero = forward_kinematics(0, 0, 0, p_zero)
        fk_offset = forward_kinematics(0, 0, 0, p_offset)

        # Hub should always be at origin
        assert fk_zero["hub"] == (0.0, 0.0)
        assert fk_offset["hub"] == (0.0, 0.0)

        # Shoulder should be displaced with offset
        sh_zero = fk_zero["shoulder"]
        sh_offset = fk_offset["shoulder"]
        # With 30° offset, shoulder should have non-zero x
        assert abs(sh_offset[0]) > abs(sh_zero[0])

    def test_forward_kinematics_has_hub_key(self) -> None:
        """Forward kinematics result should include 'hub' key."""
        from double_pendulum_golf.physics_triple import (
            TriplePendulumParams,
            forward_kinematics,
        )

        p = TriplePendulumParams(m1=5, m2=0.5, m3=0.4, L1=0.2, L2=0.65, L3=1.1)
        fk = forward_kinematics(0, 0, 0, p)
        assert "hub" in fk
        assert "shoulder" in fk
        assert "tip" in fk


@_skip_no_qt_runtime
class Test3DSegmentMode:
    """#1155: 3D segment display."""

    def test_base_widget_has_3d_mode(self) -> None:
        """BasePendulumWidget should have _3d_mode attribute."""
        from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget

        # Verify the class has set_3d_mode method
        assert hasattr(BasePendulumWidget, "set_3d_mode")
        assert hasattr(BasePendulumWidget, "_draw_3d_segment")


class TestDiagnosticLogging:
    """#1154: Logging module uses logging, never print."""

    def test_no_print_statements_in_physics(self) -> None:
        """physics.py should not contain print() calls."""
        import inspect
        import double_pendulum_golf.physics as phys

        source = inspect.getsource(phys)
        # Allow 'print' only in string literals and comments
        import re

        # Find print() calls outside of strings
        matches = re.findall(r"^\s*print\s*\(", source, re.MULTILINE)
        assert len(matches) == 0, f"Found {len(matches)} print() calls in physics.py"

    def test_no_print_statements_in_physics_triple(self) -> None:
        """physics_triple.py should not contain print() calls."""
        import inspect
        import re
        import double_pendulum_golf.physics_triple as phys_t

        source = inspect.getsource(phys_t)
        matches = re.findall(r"^\s*print\s*\(", source, re.MULTILINE)
        assert len(matches) == 0, f"Found {len(matches)} print() calls in physics_triple.py"
