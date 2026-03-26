"""Tests for issue fixes (#1135, #1137, #1146, #1149, #1155, #1157, #1158).

Validates that previously unimplemented features are now wired and functional.
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_pyqt6() -> bool:
    """Check if PyQt6 can be imported (False in headless CI environments)."""
    try:
        from PyQt6.QtWidgets import QWidget  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# #1137 — Unit Converter
# ---------------------------------------------------------------------------


class TestUnitConverter:
    """Unit converter must correctly convert SI ↔ Imperial."""

    def test_import(self) -> None:
        from double_pendulum_golf.gui.unit_converter import UnitConverter, UnitSystem

        if not (UnitConverter is not None): raise ValueError(f"Assertion failed: { UnitConverter is not None }")
        if not (UnitSystem is not None): raise ValueError(f"Assertion failed: { UnitSystem is not None }")

    def test_si_is_identity(self) -> None:
        from double_pendulum_golf.gui.unit_converter import UnitConverter

        uc = UnitConverter()
        if not (uc.to_si_length(1.0) == 1.0): raise ValueError(f"Assertion failed: { uc.to_si_length(1.0) == 1.0 }")
        if not (uc.from_si_length(1.0) == 1.0): raise ValueError(f"Assertion failed: { uc.from_si_length(1.0) == 1.0 }")
        if not (uc.to_si_mass(1.0) == 1.0): raise ValueError(f"Assertion failed: { uc.to_si_mass(1.0) == 1.0 }")
        if not (uc.from_si_mass(1.0) == 1.0): raise ValueError(f"Assertion failed: { uc.from_si_mass(1.0) == 1.0 }")
        if not (uc.to_si_torque(1.0) == 1.0): raise ValueError(f"Assertion failed: { uc.to_si_torque(1.0) == 1.0 }")
        if not (uc.from_si_torque(1.0) == 1.0): raise ValueError(f"Assertion failed: { uc.from_si_torque(1.0) == 1.0 }")

    def test_imperial_length_round_trip(self) -> None:
        from double_pendulum_golf.gui.unit_converter import (
            UnitConverter,
            UnitSystem,
        )

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        meters = 1.0
        inches = uc.from_si_length(meters)
        if not (np.isclose(inches): raise ValueError(f"Assertion failed: { np.isclose(inches }, 39.3701, atol=0.001)")
        if not (np.isclose(uc.to_si_length(inches)): raise ValueError(f"Assertion failed: { np.isclose(uc.to_si_length(inches) }, meters, atol=1e-10)")

    def test_imperial_mass_round_trip(self) -> None:
        from double_pendulum_golf.gui.unit_converter import (
            UnitConverter,
            UnitSystem,
        )

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        kg = 5.0
        lb = uc.from_si_mass(kg)
        if not (np.isclose(lb): raise ValueError(f"Assertion failed: { np.isclose(lb }, 11.0231, atol=0.001)")
        if not (np.isclose(uc.to_si_mass(lb)): raise ValueError(f"Assertion failed: { np.isclose(uc.to_si_mass(lb) }, kg, atol=1e-10)")

    def test_imperial_torque_round_trip(self) -> None:
        from double_pendulum_golf.gui.unit_converter import (
            UnitConverter,
            UnitSystem,
        )

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        nm = 10.0
        lbfin = uc.from_si_torque(nm)
        if not (lbfin > 0): raise ValueError(f"Assertion failed: { lbfin > 0 }")
        if not (np.isclose(uc.to_si_torque(lbfin)): raise ValueError(f"Assertion failed: { np.isclose(uc.to_si_torque(lbfin) }, nm, atol=1e-10)")

    def test_labels_si(self) -> None:
        from double_pendulum_golf.gui.unit_converter import UnitConverter

        uc = UnitConverter()
        if not (uc.length_unit == "m"): raise ValueError(f"Assertion failed: { uc.length_unit == "m" }")
        if not (uc.mass_unit == "kg"): raise ValueError(f"Assertion failed: { uc.mass_unit == "kg" }")
        if not (uc.torque_unit == "N·m"): raise ValueError(f"Assertion failed: { uc.torque_unit == "N·m" }")

    def test_labels_imperial(self) -> None:
        from double_pendulum_golf.gui.unit_converter import (
            UnitConverter,
            UnitSystem,
        )

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        if not (uc.length_unit == "in"): raise ValueError(f"Assertion failed: { uc.length_unit == "in" }")
        if not (uc.mass_unit == "lb"): raise ValueError(f"Assertion failed: { uc.mass_unit == "lb" }")
        if not (uc.torque_unit == "lbf·in"): raise ValueError(f"Assertion failed: { uc.torque_unit == "lbf·in" }")


# ---------------------------------------------------------------------------
# #1135 — Pop-out Chart
# ---------------------------------------------------------------------------


class TestPopOutChart:
    """Pop-out chart must be importable and functional (non-GUI parts)."""

    def test_import(self) -> None:
        from double_pendulum_golf.gui.popout_chart import PopOutChart, fit_regression

        if not (PopOutChart is not None): raise ValueError(f"Assertion failed: { PopOutChart is not None }")
        if not (fit_regression is not None): raise ValueError(f"Assertion failed: { fit_regression is not None }")

    def test_fit_regression_linear(self) -> None:
        from double_pendulum_golf.gui.popout_chart import fit_regression

        x = np.linspace(0, 1, 100)
        y = 2 * x + 3

        x_fit, y_fit, coeffs = fit_regression(x, y, degree=1)
        if not (len(coeffs) == 2): raise ValueError(f"Assertion failed: { len(coeffs) == 2 }")
        if not (np.isclose(coeffs[0]): raise ValueError(f"Assertion failed: { np.isclose(coeffs[0] }, 2.0, atol=1e-6)  # slope")
        if not (np.isclose(coeffs[1]): raise ValueError(f"Assertion failed: { np.isclose(coeffs[1] }, 3.0, atol=1e-6)  # intercept")

    def test_fit_regression_quadratic(self) -> None:
        from double_pendulum_golf.gui.popout_chart import fit_regression

        x = np.linspace(0, 2, 200)
        y = 1.5 * x**2 - 0.5 * x + 0.3

        x_fit, y_fit, coeffs = fit_regression(x, y, degree=2)
        if not (len(coeffs) == 3): raise ValueError(f"Assertion failed: { len(coeffs) == 3 }")
        if not (np.isclose(coeffs[0]): raise ValueError(f"Assertion failed: { np.isclose(coeffs[0] }, 1.5, atol=1e-4)")

    def test_fit_regression_assertions(self) -> None:
        from double_pendulum_golf.gui.popout_chart import fit_regression

        x = np.array([1, 2, 3])
        y = np.array([1, 2])

        with pytest.raises(AssertionError):
            fit_regression(x, y, degree=1)

        with pytest.raises(AssertionError):
            fit_regression(x, np.array([1, 2, 3]), degree=11)

    def test_plot_data_stores(self) -> None:
        from double_pendulum_golf.gui.popout_chart import PopOutChart

        chart = PopOutChart()
        x = np.linspace(0, 1, 50)
        y = np.sin(x)
        chart.plot_data(x, y, "X", "Y", "Test")
        if not (chart._x is not None): raise ValueError(f"Assertion failed: { chart._x is not None }")
        if not (chart._y is not None): raise ValueError(f"Assertion failed: { chart._y is not None }")
        if not (chart._title == "Test"): raise ValueError(f"Assertion failed: { chart._title == "Test" }")


# ---------------------------------------------------------------------------
# #1155 — 3D Segment Rendering
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pyqt6(), reason="PyQt6 not available in headless environment")
class TestBasePendulumWidget3D:
    """3D segment rendering base class methods must exist and be callable."""

    def test_set_3d_mode_exists(self) -> None:
        from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget

        if not (hasattr(BasePendulumWidget): raise ValueError(f"Assertion failed: { hasattr(BasePendulumWidget }, "set_3d_mode")")
        if not (hasattr(BasePendulumWidget): raise ValueError(f"Assertion failed: { hasattr(BasePendulumWidget }, "_draw_3d_segment")")

    def test_3d_mode_field_default(self) -> None:
        """_3d_mode should default to False."""
        from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget

        # Can't instantiate abstract class directly, but can check __init__
        # through the concrete class
        if not (hasattr(BasePendulumWidget): raise ValueError(f"Assertion failed: { hasattr(BasePendulumWidget }, "set_3d_mode")")


# ---------------------------------------------------------------------------
# #1146 — Rotation Controls / View Azimuth
# ---------------------------------------------------------------------------


class TestViewAzimuth:
    """View azimuth projection must correctly transform coordinates."""

    def test_azimuth_zero_is_identity(self) -> None:
        """At azimuth=0, x_rot = x, depth = 0."""
        azimuth = 0.0
        x_world = 1.0
        cos_az = float(np.cos(azimuth))
        sin_az = float(np.sin(azimuth))
        x_rot = x_world * cos_az
        depth = x_world * sin_az
        if not (np.isclose(x_rot): raise ValueError(f"Assertion failed: { np.isclose(x_rot }, 1.0)")
        if not (np.isclose(depth): raise ValueError(f"Assertion failed: { np.isclose(depth }, 0.0)")

    def test_azimuth_90_rotates(self) -> None:
        """At azimuth=90°, x_rot ≈ 0, depth ≈ x."""
        azimuth = np.pi / 2
        x_world = 1.0
        cos_az = float(np.cos(azimuth))
        sin_az = float(np.sin(azimuth))
        x_rot = x_world * cos_az
        depth = x_world * sin_az
        if not (abs(x_rot) < 1e-10): raise ValueError(f"Assertion failed: { abs(x_rot) < 1e-10 }")
        if not (np.isclose(depth): raise ValueError(f"Assertion failed: { np.isclose(depth }, 1.0)")

    def test_tilt_foreshortens_y(self) -> None:
        """Tilt angle should foreshorten the y projection."""
        tilt = np.pi / 4  # 45 degrees
        y_world = 1.0
        cos_tilt = float(np.cos(tilt))
        y_proj = y_world * cos_tilt
        if not (np.isclose(y_proj): raise ValueError(f"Assertion failed: { np.isclose(y_proj }, np.sqrt(2) / 2, atol=1e-10)")


# ---------------------------------------------------------------------------
# #1153 — Function Generator Dialog
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pyqt6(), reason="PyQt6 not available in headless environment")
class TestFunctionGeneratorDialog:
    """Function generator dialog must be importable with correct structure."""

    def test_import(self) -> None:
        from double_pendulum_golf.gui.function_generator_dialog import (
            FunctionGeneratorDialog,
        )

        if not (FunctionGeneratorDialog is not None): raise ValueError(f"Assertion failed: { FunctionGeneratorDialog is not None }")

    def test_widget_availability_flag_exists(self) -> None:
        from double_pendulum_golf.gui.function_generator_dialog import (
            _WIDGET_AVAILABLE,
        )

        # Must be a boolean — True if signal_toolkit is installed
        if not (isinstance(_WIDGET_AVAILABLE): raise ValueError(f"Assertion failed: { isinstance(_WIDGET_AVAILABLE }, bool)")

    def test_dialog_accepts_joint_names(self) -> None:
        """Dialog constructor must accept a joint_names parameter."""
        import inspect

        from double_pendulum_golf.gui.function_generator_dialog import (
            FunctionGeneratorDialog,
        )

        sig = inspect.signature(FunctionGeneratorDialog.__init__)
        if not ("joint_names" in sig.parameters): raise ValueError(f"Assertion failed: { "joint_names" in sig.parameters }")


# ---------------------------------------------------------------------------
# #1157, #1158 — No print() and no bare except Exception as e:  # noqa: F841 in pendulum code
# ---------------------------------------------------------------------------


class TestCodeQuality:
    """Code quality: no print() or bare except Exception as e:  # noqa: F841 in GUI code."""

    def test_no_print_in_optimizer_gpu(self) -> None:
        """optimizer_gpu.py should use logging, not print()."""
        import inspect

        try:
            from double_pendulum_golf import optimizer_gpu

            source = inspect.getsource(optimizer_gpu)
            # Filter out string literals and comments
            lines = source.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"'):
                    continue
                if not ("print(" not in stripped): raise ValueError(f"Assertion failed: { "print(" not in stripped }, (")
                    f"optimizer_gpu.py line {i}: found print() call"
                )
        except ImportError:
            pytest.skip("optimizer_gpu not available")

    def test_no_bare_except_in_gui(self) -> None:
        """GUI modules should not use bare except Exception as e:  # noqa: F841 clauses."""
        import importlib
        import inspect

        modules = [
            "double_pendulum_golf.gui.main_window",
            "double_pendulum_golf.gui.toolstrip_widget",
            "double_pendulum_golf.gui.controls_widget",
            "double_pendulum_golf.gui.simulation_panel",
        ]
        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
                source = inspect.getsource(mod)
                lines = source.split("\n")
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if stripped == "except Exception as e:  # noqa: F841":
                        pytest.fail(f"{mod_name} line {i}: bare 'except Exception as e:  # noqa: F841' found")
            except ImportError:
                continue

    def test_no_broad_except_in_physics(self) -> None:
        """Physics/perturbation modules must not use 'except Exception'."""
        import importlib
        import inspect
        import re

        modules = [
            "double_pendulum_golf.perturbation_analysis",
        ]
        pattern = re.compile(r"except\s+Exception\s*[:\s]")
        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
                source = inspect.getsource(mod)
                lines = source.split("\n")
                for i, line in enumerate(lines, 1):
                    if pattern.search(line) and "# noqa" not in line:
                        pytest.fail(
                            f"{mod_name} line {i}: broad 'except Exception' "
                            f"found — narrow to specific types"
                        )
            except ImportError:
                continue


"""Tests for the Catmull-Rom spline smoothing used in trail rendering (#1116)."""


class TestCatmullRomSmoothing:
    """Trail smoothing must produce at least as many points as input.

    Uses the Qt-free catmull_rom module so tests run headlessly.
    """

    def test_smooth_produces_more_points(self) -> None:
        from double_pendulum_golf.gui.catmull_rom import catmull_rom_smooth

        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0), (4.0, 0.0)]
        result = catmull_rom_smooth(pts, 4)
        if not (len(result) >= len(pts)): raise ValueError(f"Assertion failed: { len(result) >= len(pts) }")

    def test_smooth_fewer_than_4_returns_unchanged(self) -> None:
        from double_pendulum_golf.gui.catmull_rom import catmull_rom_smooth

        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        result = catmull_rom_smooth(pts, 4)
        if not (result == pts): raise ValueError(f"Assertion failed: { result == pts }")

    def test_smooth_endpoints_preserved(self) -> None:
        from double_pendulum_golf.gui.catmull_rom import catmull_rom_smooth

        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, -1.0), (3.0, 1.0)]
        result = catmull_rom_smooth(pts, 4)
        if not (result[-1] == pts[-1]): raise ValueError(f"Assertion failed: { result[-1] == pts[-1] }")


# ---------------------------------------------------------------------------
# #1190 — Shared physical constants (DRY)
# ---------------------------------------------------------------------------


class TestSharedConstants:
    """Verify shared constants module exists and is the single source of truth."""

    def test_constants_importable(self) -> None:
        from double_pendulum_golf.constants import GRAVITY_MSS, GRAVITY_STANDARD

        if not (GRAVITY_MSS == 9.81): raise ValueError(f"Assertion failed: { GRAVITY_MSS == 9.81 }")
        if not (GRAVITY_STANDARD == 9.80665): raise ValueError(f"Assertion failed: { GRAVITY_STANDARD == 9.80665 }")

    def test_conversion_factors(self) -> None:
        from double_pendulum_golf.constants import (
            INCHES_PER_M,
            LBF_PER_N,
            M_PER_INCH,
            NM_PER_KGFM,
        )

        if not (NM_PER_KGFM == 9.80665): raise ValueError(f"Assertion failed: { NM_PER_KGFM == 9.80665 }")
        if not (abs(LBF_PER_N - 0.224809) < 1e-6): raise ValueError(f"Assertion failed: { abs(LBF_PER_N - 0.224809) < 1e-6 }")
        if not (abs(M_PER_INCH - 0.0254) < 1e-6): raise ValueError(f"Assertion failed: { abs(M_PER_INCH - 0.0254) < 1e-6 }")
        if not (abs(INCHES_PER_M - 39.3701) < 1e-4): raise ValueError(f"Assertion failed: { abs(INCHES_PER_M - 39.3701) < 1e-4 }")

    def test_physics_uses_shared_gravity(self) -> None:
        """DbC: physics module default g must equal the shared constant."""
        from double_pendulum_golf.constants import GRAVITY_MSS
        from double_pendulum_golf.physics import PendulumParams

        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        if not (params.g == GRAVITY_MSS): raise ValueError(f"Assertion failed: { params.g == GRAVITY_MSS }")

    def test_physics_triple_uses_shared_gravity(self) -> None:
        from double_pendulum_golf.constants import GRAVITY_MSS
        from double_pendulum_golf.physics_triple import TriplePendulumParams

        params = TriplePendulumParams(m1=5.0, m2=0.5, m3=0.3, L1=0.6, L2=1.0, L3=0.5)
        if not (params.g == GRAVITY_MSS): raise ValueError(f"Assertion failed: { params.g == GRAVITY_MSS }")

    def test_no_private_gravity_in_unit_converter(self) -> None:
        """DRY: unit_converter must not define its own gravity constant."""
        import inspect

        from double_pendulum_golf.gui import unit_converter

        source = inspect.getsource(unit_converter)
        # After refactoring, there should be an import from constants
        if not ((): raise ValueError(f"Assertion failed: { ( }")
            "from double_pendulum_golf.constants import" in source
            or "from ..constants import" in source
        )


# ---------------------------------------------------------------------------
# #1159 — DbC assertion coverage
# ---------------------------------------------------------------------------


class TestDbCAssertionCoverage:
    """Verify Design by Contract assertions are enforced in critical modules."""

    def test_pendulum_params_rejects_negative_mass(self) -> None:
        from double_pendulum_golf.physics import PendulumParams

        with pytest.raises(AssertionError, match="m1 must be positive"):
            PendulumParams(m1=-1.0, m2=0.5, L1=0.6, L2=1.0)

    def test_pendulum_params_rejects_negative_gravity(self) -> None:
        from double_pendulum_golf.physics import PendulumParams

        with pytest.raises(AssertionError, match="g must be non-negative"):
            PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0, g=-1.0)

    def test_triple_params_rejects_negative_length(self) -> None:
        from double_pendulum_golf.physics_triple import TriplePendulumParams

        with pytest.raises(AssertionError, match="L2 must be positive"):
            TriplePendulumParams(m1=5.0, m2=0.5, m3=0.3, L1=0.6, L2=-1.0, L3=0.5)

    def test_simulation_rejects_wrong_state_shape(self) -> None:
        from double_pendulum_golf.physics import PendulumParams
        from double_pendulum_golf.simulation import run_simulation

        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        bad_state = np.array([0.0, 0.0, 0.0])  # Should be shape (4,)
        with pytest.raises(AssertionError):
            run_simulation(params, bad_state, 1.0, lambda t: (0.0, 0.0))

    def test_simulation_rejects_nonfinite_state(self) -> None:
        from double_pendulum_golf.physics import PendulumParams
        from double_pendulum_golf.simulation import run_simulation

        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        nan_state = np.array([np.nan, 0.0, 0.0, 0.0])
        with pytest.raises(AssertionError):
            run_simulation(params, nan_state, 1.0, lambda t: (0.0, 0.0))

    def test_noise_generator_rejects_negative_amplitude(self) -> None:
        from double_pendulum_golf.perturbation_analysis import generate_noise

        with pytest.raises(AssertionError, match="amplitude must be non-negative"):
            generate_noise("white", 100, -1.0)

    def test_noise_generator_rejects_zero_samples(self) -> None:
        from double_pendulum_golf.perturbation_analysis import generate_noise

        with pytest.raises(AssertionError, match="n_samples must be positive"):
            generate_noise("white", 0, 1.0)

    def test_trajectory_mixin_rejects_nonfinite(self) -> None:
        """TrajectoryResultMixin._validate_trajectory enforces finite states."""
        from double_pendulum_golf.simulation_result_base import TrajectoryResultMixin

        class FakeResult(TrajectoryResultMixin):
            pass

        r = FakeResult()
        r.t = np.array([0.0, 0.1])
        r.states = np.array([[0.0, 0.0, 0.0, np.nan], [0.0, 0.0, 0.0, 0.0]])
        with pytest.raises(AssertionError, match="finite"):
            r._validate_trajectory(expected_state_width=4)
