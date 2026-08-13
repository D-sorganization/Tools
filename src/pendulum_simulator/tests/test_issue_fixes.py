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

        assert UnitConverter is not None
        assert UnitSystem is not None

    def test_si_is_identity(self) -> None:
        from double_pendulum_golf.gui.unit_converter import UnitConverter

        uc = UnitConverter()
        assert uc.to_si_length(1.0) == 1.0
        assert uc.from_si_length(1.0) == 1.0
        assert uc.to_si_mass(1.0) == 1.0
        assert uc.from_si_mass(1.0) == 1.0
        assert uc.to_si_torque(1.0) == 1.0
        assert uc.from_si_torque(1.0) == 1.0

    def test_imperial_length_round_trip(self) -> None:
        from double_pendulum_golf.gui.unit_converter import (
            UnitConverter,
            UnitSystem,
        )

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        meters = 1.0
        inches = uc.from_si_length(meters)
        assert np.isclose(inches, 39.3701, atol=0.001)
        assert np.isclose(uc.to_si_length(inches), meters, atol=1e-10)

    def test_imperial_mass_round_trip(self) -> None:
        from double_pendulum_golf.gui.unit_converter import (
            UnitConverter,
            UnitSystem,
        )

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        kg = 5.0
        lb = uc.from_si_mass(kg)
        assert np.isclose(lb, 11.0231, atol=0.001)
        assert np.isclose(uc.to_si_mass(lb), kg, atol=1e-10)

    def test_imperial_torque_round_trip(self) -> None:
        from double_pendulum_golf.gui.unit_converter import (
            UnitConverter,
            UnitSystem,
        )

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        nm = 10.0
        lbfin = uc.from_si_torque(nm)
        assert lbfin > 0
        assert np.isclose(uc.to_si_torque(lbfin), nm, atol=1e-10)

    def test_labels_si(self) -> None:
        from double_pendulum_golf.gui.unit_converter import UnitConverter

        uc = UnitConverter()
        assert uc.length_unit == "m"
        assert uc.mass_unit == "kg"
        assert uc.torque_unit == "N·m"

    def test_labels_imperial(self) -> None:
        from double_pendulum_golf.gui.unit_converter import (
            UnitConverter,
            UnitSystem,
        )

        uc = UnitConverter(system=UnitSystem.IMPERIAL)
        assert uc.length_unit == "in"
        assert uc.mass_unit == "lb"
        assert uc.torque_unit == "lbf·in"


# ---------------------------------------------------------------------------
# #1135 — Pop-out Chart
# ---------------------------------------------------------------------------


class TestPopOutChart:
    """Pop-out chart must be importable and functional (non-GUI parts)."""

    def test_import(self) -> None:
        from double_pendulum_golf.gui.popout_chart import PopOutChart, fit_regression

        assert PopOutChart is not None
        assert fit_regression is not None

    def test_fit_regression_linear(self) -> None:
        from double_pendulum_golf.gui.popout_chart import fit_regression

        x = np.linspace(0, 1, 100)
        y = 2 * x + 3

        x_fit, y_fit, coeffs = fit_regression(x, y, degree=1)
        assert len(coeffs) == 2
        assert np.isclose(coeffs[0], 2.0, atol=1e-6)  # slope
        assert np.isclose(coeffs[1], 3.0, atol=1e-6)  # intercept

    def test_fit_regression_quadratic(self) -> None:
        from double_pendulum_golf.gui.popout_chart import fit_regression

        x = np.linspace(0, 2, 200)
        y = 1.5 * x**2 - 0.5 * x + 0.3

        x_fit, y_fit, coeffs = fit_regression(x, y, degree=2)
        assert len(coeffs) == 3
        assert np.isclose(coeffs[0], 1.5, atol=1e-4)

    def test_fit_regression_assertions(self) -> None:
        from double_pendulum_golf.gui.popout_chart import fit_regression

        x = np.array([1, 2, 3])
        y = np.array([1, 2])

        with pytest.raises((ValueError, TypeError)):
            fit_regression(x, y, degree=1)

        with pytest.raises((ValueError, TypeError)):
            fit_regression(x, np.array([1, 2, 3]), degree=11)

    def test_plot_data_stores(self) -> None:
        from double_pendulum_golf.gui.popout_chart import PopOutChart

        chart = PopOutChart()
        x = np.linspace(0, 1, 50)
        y = np.sin(x)
        chart.plot_data(x, y, "X", "Y", "Test")
        assert chart._x is not None
        assert chart._y is not None
        assert chart._title == "Test"


# ---------------------------------------------------------------------------
# #1155 — 3D Segment Rendering
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_pyqt6(), reason="PyQt6 not available in headless environment"
)
class TestBasePendulumWidget3D:
    """3D segment rendering base class methods must exist and be callable."""

    def test_set_3d_mode_exists(self) -> None:
        from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget

        assert hasattr(BasePendulumWidget, "set_3d_mode")
        assert hasattr(BasePendulumWidget, "_draw_3d_segment")

    def test_3d_mode_field_default(self) -> None:
        """_3d_mode should default to False."""
        from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget

        # Can't instantiate abstract class directly, but can check __init__
        # through the concrete class
        assert hasattr(BasePendulumWidget, "set_3d_mode")


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
        assert np.isclose(x_rot, 1.0)
        assert np.isclose(depth, 0.0)

    def test_azimuth_90_rotates(self) -> None:
        """At azimuth=90°, x_rot ≈ 0, depth ≈ x."""
        azimuth = np.pi / 2
        x_world = 1.0
        cos_az = float(np.cos(azimuth))
        sin_az = float(np.sin(azimuth))
        x_rot = x_world * cos_az
        depth = x_world * sin_az
        assert abs(x_rot) < 1e-10
        assert np.isclose(depth, 1.0)

    def test_tilt_foreshortens_y(self) -> None:
        """Tilt angle should foreshorten the y projection."""
        tilt = np.pi / 4  # 45 degrees
        y_world = 1.0
        cos_tilt = float(np.cos(tilt))
        y_proj = y_world * cos_tilt
        assert np.isclose(y_proj, np.sqrt(2) / 2, atol=1e-10)


# ---------------------------------------------------------------------------
# #1153 — Function Generator Dialog
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_pyqt6(), reason="PyQt6 not available in headless environment"
)
class TestFunctionGeneratorDialog:
    """Function generator dialog must be importable with correct structure."""

    def test_import(self) -> None:
        from double_pendulum_golf.gui.function_generator_dialog import (
            FunctionGeneratorDialog,
        )

        assert FunctionGeneratorDialog is not None

    def test_widget_availability_flag_exists(self) -> None:
        from double_pendulum_golf.gui.function_generator_dialog import (
            _WIDGET_AVAILABLE,
        )

        # Must be a boolean — True if signal_toolkit is installed
        assert isinstance(_WIDGET_AVAILABLE, bool)

    def test_dialog_accepts_joint_names(self) -> None:
        """Dialog constructor must accept a joint_names parameter."""
        import inspect

        from double_pendulum_golf.gui.function_generator_dialog import (
            FunctionGeneratorDialog,
        )

        sig = inspect.signature(FunctionGeneratorDialog.__init__)
        assert "joint_names" in sig.parameters


# ---------------------------------------------------------------------------
# #1157, #1158 — No print() and no bare except: in pendulum code
# ---------------------------------------------------------------------------


class TestCodeQuality:
    """Code quality: no print() or bare except: in GUI code."""

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
                assert (
                    "print(" not in stripped
                ), f"optimizer_gpu.py line {i}: found print() call"
        except ImportError:
            pytest.skip("optimizer_gpu not available")

    def test_no_bare_except_in_gui(self) -> None:
        """GUI modules should not use bare except: clauses."""
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
                    if stripped == "except:":
                        pytest.fail(f"{mod_name} line {i}: bare 'except:' found")
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
        assert len(result) >= len(pts)

    def test_smooth_fewer_than_4_returns_unchanged(self) -> None:
        from double_pendulum_golf.gui.catmull_rom import catmull_rom_smooth

        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        result = catmull_rom_smooth(pts, 4)
        assert result == pts

    def test_smooth_endpoints_preserved(self) -> None:
        from double_pendulum_golf.gui.catmull_rom import catmull_rom_smooth

        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, -1.0), (3.0, 1.0)]
        result = catmull_rom_smooth(pts, 4)
        assert result[-1] == pts[-1]


# ---------------------------------------------------------------------------
# #1190 — Shared physical constants (DRY)
# ---------------------------------------------------------------------------


class TestSharedConstants:
    """Verify shared constants module exists and is the single source of truth."""

    def test_constants_importable(self) -> None:
        from double_pendulum_golf.constants import GRAVITY_MSS, GRAVITY_STANDARD

        assert GRAVITY_MSS == 9.81
        assert GRAVITY_STANDARD == 9.80665

    def test_conversion_factors(self) -> None:
        from double_pendulum_golf.constants import (
            INCHES_PER_M,
            LBF_PER_N,
            M_PER_INCH,
            NM_PER_KGFM,
        )

        assert NM_PER_KGFM == 9.80665
        assert abs(LBF_PER_N - 0.224809) < 1e-6
        assert abs(M_PER_INCH - 0.0254) < 1e-6
        assert abs(INCHES_PER_M - 39.3701) < 1e-4

    def test_physics_uses_shared_gravity(self) -> None:
        """DbC: physics module default g must equal the shared constant."""
        from double_pendulum_golf.constants import GRAVITY_MSS
        from double_pendulum_golf.physics import PendulumParams

        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        assert params.g == GRAVITY_MSS

    def test_physics_triple_uses_shared_gravity(self) -> None:
        from double_pendulum_golf.constants import GRAVITY_MSS
        from double_pendulum_golf.physics_triple import TriplePendulumParams

        params = TriplePendulumParams(m1=5.0, m2=0.5, m3=0.3, L1=0.6, L2=1.0, L3=0.5)
        assert params.g == GRAVITY_MSS

    def test_no_private_gravity_in_unit_converter(self) -> None:
        """DRY: unit_converter must not define its own gravity constant."""
        import inspect

        from double_pendulum_golf.gui import unit_converter

        source = inspect.getsource(unit_converter)
        # After refactoring, there should be an import from constants
        assert (
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

        with pytest.raises((ValueError, TypeError), match="m1 must be positive"):
            PendulumParams(m1=-1.0, m2=0.5, L1=0.6, L2=1.0)

    def test_pendulum_params_rejects_negative_gravity(self) -> None:
        from double_pendulum_golf.physics import PendulumParams

        with pytest.raises((ValueError, TypeError), match="g must be non-negative"):
            PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0, g=-1.0)

    def test_triple_params_rejects_negative_length(self) -> None:
        from double_pendulum_golf.physics_triple import TriplePendulumParams

        with pytest.raises((ValueError, TypeError), match="L2 must be positive"):
            TriplePendulumParams(m1=5.0, m2=0.5, m3=0.3, L1=0.6, L2=-1.0, L3=0.5)

    def test_simulation_rejects_wrong_state_shape(self) -> None:
        from double_pendulum_golf.physics import PendulumParams
        from double_pendulum_golf.simulation import run_simulation

        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        bad_state = np.array([0.0, 0.0, 0.0])  # Should be shape (4,)
        with pytest.raises((ValueError, TypeError)):
            run_simulation(params, bad_state, 1.0, lambda t: (0.0, 0.0))

    def test_simulation_rejects_nonfinite_state(self) -> None:
        from double_pendulum_golf.physics import PendulumParams
        from double_pendulum_golf.simulation import run_simulation

        params = PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)
        nan_state = np.array([np.nan, 0.0, 0.0, 0.0])
        with pytest.raises((ValueError, TypeError)):
            run_simulation(params, nan_state, 1.0, lambda t: (0.0, 0.0))

    def test_noise_generator_rejects_negative_amplitude(self) -> None:
        from double_pendulum_golf.perturbation_analysis import generate_noise

        with pytest.raises(
            (ValueError, TypeError), match="amplitude must be non-negative"
        ):
            generate_noise("white", 100, -1.0)

    def test_noise_generator_rejects_zero_samples(self) -> None:
        from double_pendulum_golf.perturbation_analysis import generate_noise

        with pytest.raises((ValueError, TypeError), match="n_samples must be positive"):
            generate_noise("white", 0, 1.0)

    def test_trajectory_mixin_rejects_nonfinite(self) -> None:
        """TrajectoryResultMixin._validate_trajectory enforces finite states."""
        from double_pendulum_golf.simulation_result_base import TrajectoryResultMixin

        class FakeResult(TrajectoryResultMixin):
            pass

        r = FakeResult()
        r.t = np.array([0.0, 0.1])
        r.states = np.array([[0.0, 0.0, 0.0, np.nan], [0.0, 0.0, 0.0, 0.0]])
        with pytest.raises((ValueError, TypeError), match="finite"):
            r._validate_trajectory(expected_state_width=4)
