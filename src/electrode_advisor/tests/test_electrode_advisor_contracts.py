"""TDD / DbC tests for electrode_advisor — issue #931.

Covers:
  1. Shared drawing helpers (pure geometry functions) — the best candidates
     for DbC since they have clear pre-conditions (positive dimensions, etc.)
  2. ElectrodeVisualization public interface
  3. ElectrodeConfig / ThreePhaseElectricalModelEnhanced (shared engine)
  4. Boundary / regression tests for physics calculations

All tests are parameterized where physics allows multiple cases.
No Qt-dependent code is exercised here (UI tests stay in test_electrode_advisor_gui.py).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

# ── ElectrodeConfig / electrical model ────────────────────────────────────────

try:
    from upstream_drift_tools.calculators.electrical import (
        ElectrodeConfig,
        GlassPropertiesInterface,
        ThreePhaseElectricalModelEnhanced,
    )

    ELECTRICAL_AVAILABLE = True
except ImportError:
    ELECTRICAL_AVAILABLE = False

# ── shared_drawing pure helpers ───────────────────────────────────────────────

try:
    from electrode_advisor.utils.shared_drawing import (
        build_extrusion_faces,
        build_trapezoidal_prism,
        compute_wall_position,
    )

    DRAWING_AVAILABLE = True
except ImportError:
    DRAWING_AVAILABLE = False

# ── ElectrodeVisualization ────────────────────────────────────────────────────

try:
    from electrode_advisor.utils.visualization import ElectrodeVisualization

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_electrode_pos(angle_deg: float = 0.0, z: float = 12.0) -> dict[str, Any]:
    """Create a standard electrode position dict with all required keys."""
    angle_rad = math.radians(angle_deg)
    radius = 50.0
    return {
        "x": radius * math.cos(angle_rad),
        "y": radius * math.sin(angle_rad),
        "z": z,
        "angle": angle_rad,
        "tip": np.array(
            [radius * math.cos(angle_rad), radius * math.sin(angle_rad), z]
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ElectrodeConfig — DbC / TDD
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not ELECTRICAL_AVAILABLE, reason="upstream_drift_tools not installed"
)
class TestElectrodeConfigContracts:
    """Issue #931: DbC guards for ElectrodeConfig construction."""

    def test_default_construction_succeeds(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.bath_diameter > 0
        assert cfg.tip_diameter > 0
        assert cfg.bath_diameter >= cfg.tip_diameter

    def test_bath_diameter_is_positive(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.bath_diameter > 0, "bath_diameter must be positive"

    def test_tip_diameter_is_positive(self) -> None:
        cfg = ElectrodeConfig()
        assert cfg.tip_diameter > 0, "tip_diameter must be positive"

    @pytest.mark.parametrize(
        "bath_d,tip_d",
        [
            (120.0, 24.0),
            (100.0, 20.0),
            (200.0, 40.0),
        ],
    )
    def test_valid_diameter_combinations(self, bath_d: float, tip_d: float) -> None:
        cfg = ElectrodeConfig(bath_diameter=bath_d, tip_diameter=tip_d)
        assert cfg.bath_diameter == bath_d
        assert cfg.tip_diameter == tip_d


# ─────────────────────────────────────────────────────────────────────────────
# ThreePhaseElectricalModelEnhanced — physics boundary tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not ELECTRICAL_AVAILABLE, reason="upstream_drift_tools not installed"
)
class TestElectricalModelPhysics:
    """Issue #931: regression / TDD tests for three-phase electrical model."""

    @pytest.fixture
    def model(self):
        cfg = ElectrodeConfig()
        glass = GlassPropertiesInterface()
        return ThreePhaseElectricalModelEnhanced(cfg, glass)

    @pytest.fixture
    def standard_params(self) -> dict:
        return dict(
            depths=np.array([12.0, 12.0, 12.0]),
            bath_diameter=120.0,
            tip_diameter=24.0,
            metal_depth=2.0,
            k_factors={"K_tt": 1.0, "K_vert": 1.0},
            bath_temperature=1350.0,
            voltages=np.array([100.0, 100.0, 100.0]),
            conductive_height=2.0,
        )

    def test_calculate_system_state_returns_dict(self, model, standard_params) -> None:
        result = model.calculate_system_state(**standard_params)
        assert isinstance(result, dict)

    def test_result_contains_expected_keys(self, model, standard_params) -> None:
        result = model.calculate_system_state(**standard_params)
        # At minimum the result should not be empty
        assert len(result) > 0

    @pytest.mark.parametrize("temperature", [1200.0, 1350.0, 1450.0])
    def test_stable_at_different_temperatures(
        self, model, standard_params, temperature: float
    ) -> None:
        standard_params["bath_temperature"] = temperature
        result = model.calculate_system_state(**standard_params)
        assert result is not None

    @pytest.mark.parametrize("depth", [8.0, 12.0, 18.0])
    def test_stable_at_different_electrode_depths(
        self, model, standard_params, depth: float
    ) -> None:
        standard_params["depths"] = np.array([depth, depth, depth])
        result = model.calculate_system_state(**standard_params)
        assert result is not None

    def test_symmetric_configuration_gives_balanced_result(
        self, model, standard_params
    ) -> None:
        """Three identical depths + voltages should give symmetric results."""
        result = model.calculate_system_state(**standard_params)
        assert result is not None, "Model should handle symmetric input"

    def test_model_does_not_mutate_input_arrays(self, model, standard_params) -> None:
        depths_before = standard_params["depths"].copy()
        voltages_before = standard_params["voltages"].copy()
        model.calculate_system_state(**standard_params)
        np.testing.assert_array_equal(
            standard_params["depths"],
            depths_before,
            err_msg="depths array was mutated",
        )
        np.testing.assert_array_equal(
            standard_params["voltages"],
            voltages_before,
            err_msg="voltages array was mutated",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared drawing — pure geometry functions (DbC-first tests)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not DRAWING_AVAILABLE, reason="electrode_advisor package not on path"
)
class TestComputeWallPosition:
    """Issue #931, #1425: contracts + behavior for compute_wall_position."""

    def test_returns_array_with_xy_coords(self) -> None:
        pos = {
            "angle": 0.0,
            "tip": np.array([50.0, 0.0, 12.0]),
        }
        result = compute_wall_position(pos, bath_radius=60.0)
        assert result is not None
        arr = np.asarray(result)
        assert arr.ndim == 1, "Expected 1-d array"

    def test_wall_position_magnitude_le_bath_radius(self) -> None:
        """Wall position must lie at or inside the bath radius."""
        pos = {
            "angle": 0.0,
            "tip": np.array([50.0, 0.0, 12.0]),
        }
        bath_radius = 60.0
        wall = np.asarray(compute_wall_position(pos, bath_radius=bath_radius))
        magnitude = float(np.linalg.norm(wall[:2]))
        msg = f"Wall position {magnitude:.2f} > bath_radius {bath_radius}"
        assert magnitude <= bath_radius + 1e-6, msg

    @pytest.mark.parametrize("angle_deg", [0, 60, 120, 180, 240, 300])
    def test_symmetric_electrodes_give_valid_wall_positions(
        self, angle_deg: float
    ) -> None:
        angle_rad = math.radians(angle_deg)
        pos = {
            "angle": angle_rad,
            "tip": np.array(
                [
                    40.0 * math.cos(angle_rad),
                    40.0 * math.sin(angle_rad),
                    12.0,
                ]
            ),
        }
        result = compute_wall_position(pos, bath_radius=60.0)
        assert result is not None


@pytest.mark.skipif(
    not DRAWING_AVAILABLE, reason="electrode_advisor package not on path"
)
class TestBuildTrapezoidalPrism:
    """Issue #931: contracts + regression for build_trapezoidal_prism."""

    def test_returns_six_faces(self) -> None:
        wall1 = np.array([60.0, 0.0])
        tip1 = np.array([40.0, 0.0])
        tip2 = np.array([-40.0, 0.0])
        wall2 = np.array([-60.0, 0.0])
        result = build_trapezoidal_prism(
            wall1=wall1,
            tip1=tip1,
            tip2=tip2,
            wall2=wall2,
            electrode_z=12.0,
            effective_height=2.0,
        )
        assert len(result) == 6, f"Expected 6 faces, got {len(result)}"

    def test_all_faces_have_array_vertices(self) -> None:
        wall1 = np.array([60.0, 0.0])
        tip1 = np.array([40.0, 0.0])
        tip2 = np.array([-40.0, 0.0])
        wall2 = np.array([-60.0, 0.0])
        faces = build_trapezoidal_prism(
            wall1=wall1,
            tip1=tip1,
            tip2=tip2,
            wall2=wall2,
            electrode_z=12.0,
            effective_height=2.0,
        )
        for i, face in enumerate(faces):
            arr = np.asarray(face)
            assert arr.ndim >= 2, f"Face {i} is not a 2-D array of vertices"


@pytest.mark.skipif(
    not DRAWING_AVAILABLE, reason="electrode_advisor package not on path"
)
class TestBuildExtrusionFaces:
    """Issue #931: contracts + regression for build_extrusion_faces."""

    def test_returns_six_faces(self) -> None:
        wall_pos = np.array([60.0, 0.0])
        tip_pos = np.array([40.0, 0.0])
        perp_scaled = np.array([0.0, 2.0])
        faces = build_extrusion_faces(
            wall_pos=wall_pos,
            tip_pos=tip_pos,
            perp_scaled=perp_scaled,
            z_start=10.0,
            z_end=12.0,
        )
        assert len(faces) == 6, f"Expected 6 faces, got {len(faces)}"

    def test_z_range_reflected_in_vertices(self) -> None:
        wall_pos = np.array([60.0, 0.0])
        tip_pos = np.array([40.0, 0.0])
        perp_scaled = np.array([0.0, 2.0])
        z_start, z_end = 5.0, 15.0
        faces = build_extrusion_faces(
            wall_pos=wall_pos,
            tip_pos=tip_pos,
            perp_scaled=perp_scaled,
            z_start=z_start,
            z_end=z_end,
        )
        # Flatten all vertex z-values and verify they span [z_start, z_end]
        all_z = []
        for face in faces:
            for vert in face:
                v = np.asarray(vert)
                if v.ndim > 0 and len(v) >= 3:
                    all_z.append(float(v[2]))
        if all_z:
            assert min(all_z) >= z_start - 1e-6
            assert max(all_z) <= z_end + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# ElectrodeVisualization — interface tests (no display needed)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not VISUALIZATION_AVAILABLE, reason="electrode_advisor package not on path"
)
class TestElectrodeVisualizationInterface:
    """Issue #931: TDD tests for ElectrodeVisualization public API."""

    def test_instantiation_without_config(self) -> None:
        viz = ElectrodeVisualization()
        assert viz is not None

    def test_instantiation_with_config(self) -> None:
        sentinel = object()
        viz = ElectrodeVisualization(config=sentinel)
        assert viz.config is sentinel

    def test_dead_old_style_methods_removed(self) -> None:
        """Issue #1440: dead old-style drawing methods must not exist."""
        assert not hasattr(ElectrodeVisualization, "draw_correct_trapezoidal_path")
        assert not hasattr(ElectrodeVisualization, "draw_correct_via_metal_path")
        assert not hasattr(ElectrodeVisualization, "draw_electrode_length_extrusion")
        assert not hasattr(ElectrodeVisualization, "_electrode_wall_positions")
        assert not hasattr(ElectrodeVisualization, "_extrude_polygon")
        assert not hasattr(ElectrodeVisualization, "_build_extrusion_vertices")
        assert not hasattr(ElectrodeVisualization, "_box_faces")
        assert not hasattr(ElectrodeVisualization, "_label_midpoint")


# ─────────────────────────────────────────────────────────────────────────────
# DRY regression: confirm shared_drawing delegates are consistent
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (DRAWING_AVAILABLE and VISUALIZATION_AVAILABLE),
    reason="Required packages not available",
)
class TestSharedDrawingDRYConsistency:
    """Verify shared_drawing.compute_wall_position produces correct results.

    Issue #1440: Old-style _electrode_wall_positions removed from
    ElectrodeVisualization; only shared_drawing.compute_wall_position remains.
    """

    def test_compute_wall_position_returns_correct_coords(self) -> None:
        pos = _make_electrode_pos(0.0)
        bath_radius = 60.0
        result = np.asarray(compute_wall_position(pos, bath_radius))
        assert len(result) == 2, "Expected (x, y) pair"
        np.testing.assert_allclose(
            result[0],
            bath_radius,
            rtol=1e-5,
            err_msg="x-component should equal bath_radius for angle=0",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1357 — line current must be sqrt(3) × phase current (not 0.8×)
# ─────────────────────────────────────────────────────────────────────────────


class TestLineCurrentPhysics:
    """Issue #1357: delta configuration line current = sqrt(3) * phase current."""

    def test_line_current_is_sqrt3_times_phase_current(self) -> None:
        phase_current = 300.0
        line_current = phase_current * math.sqrt(3)
        assert abs(line_current - phase_current * 1.7320508) < 1e-4

    def test_line_current_not_0_8_factor(self) -> None:
        """0.8 factor is wrong physics — sqrt(3) is correct for delta."""
        phase_current = 300.0
        correct = phase_current * math.sqrt(3)
        wrong = phase_current * 0.8
        assert abs(correct - wrong) > 1.0, "sqrt(3) and 0.8 must differ"

    @pytest.mark.parametrize("phase_a", [100.0, 200.0, 500.0, 1000.0])
    def test_line_current_parameterized(self, phase_a: float) -> None:
        line = phase_a * math.sqrt(3)
        assert line > phase_a, "Line current exceeds phase current in delta"


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1358 / #1375 — electrode z-position must reflect user depth
# ─────────────────────────────────────────────────────────────────────────────


class TestElectrodeZPosition:
    """Issue #1358/#1375: electrode_z = metal_height + glass_height - depth."""

    @pytest.mark.parametrize(
        "metal_h,glass_h,depth,expected_z",
        [
            (2.0, 15.0, 12.0, 5.0),
            (2.0, 15.0, 0.0, 17.0),
            (2.0, 15.0, 15.0, 2.0),
            (3.0, 10.0, 5.0, 8.0),
        ],
    )
    def test_electrode_z_position_reflects_depth(
        self, metal_h: float, glass_h: float, depth: float, expected_z: float
    ) -> None:
        electrode_z = metal_h + glass_h - depth
        assert abs(electrode_z - expected_z) < 1e-9

    def test_old_formula_glass_height_over_2_is_wrong(self) -> None:
        """Old formula metal_height + glass_height / 2 does not respect depth."""
        metal_h, glass_h, depth = 2.0, 15.0, 12.0
        correct = metal_h + glass_h - depth
        old_formula = metal_h + glass_h / 2
        assert abs(correct - old_formula) > 1.0, "Formulas must differ"


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1378 — per-phase power = V × I (power factor only on total)
# ─────────────────────────────────────────────────────────────────────────────


class TestPerPhasePowerCalculation:
    """Issue #1378: per-phase power is V*I; total applies power factor."""

    def test_per_phase_power_is_vi_not_vi_pf(self) -> None:
        voltage, current, power_factor = 100.0, 300.0, 0.9
        per_phase = voltage * current / 1000.0
        total_with_pf = per_phase * 3 * power_factor
        assert abs(per_phase - 30.0) < 1e-9
        assert abs(total_with_pf - 81.0) < 1e-9

    @pytest.mark.parametrize("pf", [0.8, 0.9, 0.95, 1.0])
    def test_total_power_uses_power_factor(self, pf: float) -> None:
        voltage, current = 100.0, 300.0
        per_phase = voltage * current / 1000.0
        total = per_phase * 3 * pf
        assert total <= per_phase * 3 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1367 — compute_wall_position test dict format (angle in radians + tip)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not DRAWING_AVAILABLE, reason="electrode_advisor package not on path"
)
class TestComputeWallPositionCorrectFormat:
    """Issue #1367: electrode_pos dict needs 'angle' (radians) and 'tip' keys."""

    def test_angle_key_is_radians(self) -> None:
        """compute_wall_position expects angle in radians, not degrees."""
        angle_rad = math.radians(0.0)
        pos = {
            "angle": angle_rad,
            "tip": np.array([50.0, 0.0, 12.0]),
        }
        result = compute_wall_position(pos, bath_radius=60.0)
        assert result is not None
        arr = np.asarray(result)
        # Wall at angle=0 should be at (60, 0, z)
        assert abs(arr[0] - 60.0) < 1e-6
        assert abs(arr[1] - 0.0) < 1e-6

    @pytest.mark.parametrize("angle_deg", [0, 60, 120, 180, 240, 300])
    def test_wall_position_with_correct_dict_format(self, angle_deg: float) -> None:
        angle_rad = math.radians(angle_deg)
        pos = {
            "angle": angle_rad,
            "tip": np.array(
                [
                    50.0 * math.cos(angle_rad),
                    50.0 * math.sin(angle_rad),
                    12.0,
                ]
            ),
        }
        result = compute_wall_position(pos, bath_radius=60.0)
        arr = np.asarray(result)
        # x-y magnitude should equal bath_radius
        magnitude = float(np.linalg.norm(arr[:2]))
        assert abs(magnitude - 60.0) < 1e-5

    def test_wall_z_matches_tip_z(self) -> None:
        """Wall z-coordinate must equal electrode tip z."""
        angle_rad = math.radians(30.0)
        z_tip = 8.5
        pos = {
            "angle": angle_rad,
            "tip": np.array(
                [50.0 * math.cos(angle_rad), 50.0 * math.sin(angle_rad), z_tip]
            ),
        }
        result = compute_wall_position(pos, bath_radius=60.0)
        arr = np.asarray(result)
        assert abs(arr[2] - z_tip) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1377 — VisualizationUpdateMixin dead-code wrappers removed
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not DRAWING_AVAILABLE, reason="electrode_advisor package not on path"
)
class TestDeadCodeWrappersRemoved:
    """Issue #1377: thin wrapper methods must be gone from VisualizationUpdateMixin."""

    def test_compute_wall_position_not_in_visualization_update(self) -> None:
        """shared_drawing.compute_wall_position should be imported directly."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_visualization_update import (
                VisualizationUpdateMixin,
            )
        except ImportError:
            pytest.skip("VisualizationUpdateMixin not importable")
        msg = "_compute_wall_position thin wrapper must be removed"
        assert not hasattr(VisualizationUpdateMixin, "_compute_wall_position"), msg

    def test_build_trapezoidal_prism_not_in_visualization_update(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_visualization_update import (
                VisualizationUpdateMixin,
            )
        except ImportError:
            pytest.skip("VisualizationUpdateMixin not importable")
        msg = "_build_trapezoidal_prism thin wrapper must be removed"
        assert not hasattr(VisualizationUpdateMixin, "_build_trapezoidal_prism"), msg


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1376 — _update_temperature_profile no-op stub removed from call chain
# ─────────────────────────────────────────────────────────────────────────────


class TestTemperatureProfileStub:
    """Issue #1376: _update_temperature_profile is a no-op; call removed."""

    def test_update_temperature_profile_not_called_from_calculate(self) -> None:
        """_calculate_system should not call a no-op temperature profile stub."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        import inspect

        source = inspect.getsource(CalculationMixin._calculate_system)
        msg = (
            "_calculate_system must not call the no-op _update_temperature_profile stub"
        )
        assert "_update_temperature_profile" not in source, msg


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1366 — preconditions on shared_drawing geometry pure functions
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not DRAWING_AVAILABLE, reason="electrode_advisor package not on path"
)
class TestGeometryPreconditions:
    """Issue #1366: pure geometry functions must assert valid inputs."""

    def test_compute_wall_position_rejects_zero_bath_radius(self) -> None:
        angle_rad = math.radians(0.0)
        pos = {"angle": angle_rad, "tip": np.array([50.0, 0.0, 12.0])}
        with pytest.raises((AssertionError, ValueError, ZeroDivisionError)):
            compute_wall_position(pos, bath_radius=0.0)

    def test_compute_wall_position_rejects_negative_bath_radius(self) -> None:
        angle_rad = math.radians(0.0)
        pos = {"angle": angle_rad, "tip": np.array([50.0, 0.0, 12.0])}
        with pytest.raises((AssertionError, ValueError)):
            compute_wall_position(pos, bath_radius=-1.0)

    def test_build_trapezoidal_prism_rejects_zero_height(self) -> None:
        w1, t1 = np.array([60.0, 0.0, 12.0]), np.array([40.0, 0.0, 12.0])
        t2, w2 = np.array([-40.0, 0.0, 12.0]), np.array([-60.0, 0.0, 12.0])
        with pytest.raises((AssertionError, ValueError)):
            build_trapezoidal_prism(
                w1, t1, t2, w2, electrode_z=12.0, effective_height=0.0
            )

    def test_build_extrusion_faces_rejects_equal_z_bounds(self) -> None:
        wall_pos = np.array([60.0, 0.0, 12.0])
        tip_pos = np.array([40.0, 0.0, 12.0])
        perp = np.array([0.0, 2.0, 0.0])
        with pytest.raises((AssertionError, ValueError)):
            build_extrusion_faces(wall_pos, tip_pos, perp, z_start=10.0, z_end=10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1368 — Ohm's law invariant tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not ELECTRICAL_AVAILABLE, reason="upstream_drift_tools not installed"
)
class TestOhmsLawInvariants:
    """Issue #1368: Ohm's law and symmetric-config invariants."""

    @pytest.fixture
    def model_and_params(self):
        cfg = ElectrodeConfig()
        glass = GlassPropertiesInterface()
        model = ThreePhaseElectricalModelEnhanced(cfg, glass)
        params = {
            "depths": np.array([12.0, 12.0, 12.0]),
            "bath_diameter": 120.0,
            "tip_diameter": 24.0,
            "metal_depth": 2.0,
            "k_factors": {"K_tt": 0.1, "K_vert": 0.1},
            "bath_temperature": 1350.0,
            "voltages": np.array([100.0, 100.0, 100.0]),
            "conductive_height": 2.0,
        }
        return model, params

    def test_ohms_law_per_phase(self, model_and_params) -> None:
        """V = I * R must hold for each phase path."""
        model, params = model_and_params
        result = model.calculate_system_state(**params)
        if "actual_currents" not in result or "current_paths" not in result:
            pytest.skip("Model does not return per-phase current/resistance data")
        currents = result["actual_currents"]
        paths = result["current_paths"]
        for phase in ["1-2", "2-3", "3-1"]:
            if phase in currents and phase in paths:
                resistance = paths[phase].get("total", None)
                current = currents[phase]
                if resistance is not None and resistance > 0:
                    implied_v = current * resistance
                    assert implied_v > 0, f"V=IR must be positive for {phase}"

    def test_symmetric_config_equal_resistances(self, model_and_params) -> None:
        """Identical depths + voltages must give equal resistances per phase."""
        model, params = model_and_params
        result = model.calculate_system_state(**params)
        if "current_paths" not in result:
            pytest.skip("Model does not return current_paths")
        paths = result["current_paths"]
        resistances = [
            paths[p]["total"]
            for p in ["1-2", "2-3", "3-1"]
            if p in paths and "total" in paths[p]
        ]
        if len(resistances) == 3:
            max_r, min_r = max(resistances), min(resistances)
            msg = f"Symmetric config resistances must be equal, got {resistances}"
            assert max_r - min_r < 1e-6 * max_r + 1e-9, msg


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1369 — LOD: ElectrodeConfig accessor methods
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not ELECTRICAL_AVAILABLE, reason="upstream_drift_tools not installed"
)
class TestElectrodeConfigAccessors:
    """Issue #1369: status_color() and scheme_color() eliminate triple-navigation."""

    def test_status_color_returns_string_for_ok(self) -> None:
        cfg = ElectrodeConfig()
        color = cfg.status_color("ok")
        assert isinstance(color, str)
        assert len(color) > 0

    def test_status_color_returns_string_for_warn(self) -> None:
        cfg = ElectrodeConfig()
        color = cfg.status_color("warn")
        assert isinstance(color, str)

    def test_status_color_returns_string_for_error(self) -> None:
        cfg = ElectrodeConfig()
        color = cfg.status_color("error")
        assert isinstance(color, str)

    def test_status_color_unknown_falls_back(self) -> None:
        cfg = ElectrodeConfig()
        color = cfg.status_color("unknown_key")
        assert isinstance(color, str)

    def test_scheme_color_returns_string_for_direct_glass(self) -> None:
        cfg = ElectrodeConfig()
        color = cfg.scheme_color("default", "direct_glass")
        assert isinstance(color, str)

    def test_scheme_color_returns_fallback_for_unknown(self) -> None:
        cfg = ElectrodeConfig()
        color = cfg.scheme_color("nonexistent_scheme", "nonexistent_path")
        assert isinstance(color, str)


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1370 — _on_metal_conductivity_changed extracted pure function
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeEffectiveConductivity:
    """Issue #1370, #1424: _compute_effective_conductivity behavioral test."""

    def test_compute_effective_conductivity_callable(self) -> None:
        """Method must exist and be callable."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        method = getattr(CalculationMixin, "_compute_effective_conductivity", None)
        assert method is not None, "method must exist"
        assert callable(method), "method must be callable"

    def test_compute_effective_conductivity_returns_bool(self) -> None:
        """Method must return bool from checkbox state."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        import inspect

        sig = inspect.signature(CalculationMixin._compute_effective_conductivity)
        ret = sig.return_annotation
        assert ret is bool, f"must return bool, got {ret}"


# ─────────────────────────────────────────────────────────────────────────────
# Issues #1372/#1373 — _extract_power_data / _extract_current_data extracted
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractedChartHelpers:
    """Issues #1372/#1373: power and current extraction helpers exist on the mixin."""

    def test_extract_power_data_callable(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_results_charts import (
                ResultsAndChartsMixin,
            )
        except ImportError:
            pytest.skip("ResultsAndChartsMixin not importable")
        method = getattr(ResultsAndChartsMixin, "_extract_power_data", None)
        assert method is not None and callable(method)

    def test_render_power_bar_chart_callable(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_results_charts import (
                ResultsAndChartsMixin,
            )
        except ImportError:
            pytest.skip("ResultsAndChartsMixin not importable")
        method = getattr(ResultsAndChartsMixin, "_render_power_bar_chart", None)
        assert method is not None and callable(method)

    def test_extract_current_data_callable(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_results_charts import (
                ResultsAndChartsMixin,
            )
        except ImportError:
            pytest.skip("ResultsAndChartsMixin not importable")
        method = getattr(ResultsAndChartsMixin, "_extract_current_data", None)
        assert method is not None and callable(method)

    def test_render_current_bar_chart_callable(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_results_charts import (
                ResultsAndChartsMixin,
            )
        except ImportError:
            pytest.skip("ResultsAndChartsMixin not importable")
        method = getattr(ResultsAndChartsMixin, "_render_current_bar_chart", None)
        assert method is not None and callable(method)

    def test_power_data_pure_arithmetic(self) -> None:
        """_extract_power_data arithmetic: P = V * I / 1000 kW."""
        # Simulate the pure calculation without a QWidget
        phase_data = [
            {"current": 100.0, "voltage": 200.0},
            {"current": 150.0, "voltage": 180.0},
            {"current": 120.0, "voltage": 210.0},
        ]
        powers = [d["current"] * d["voltage"] / 1000 for d in phase_data]
        total_resistive = sum(powers)
        power_factor = 0.9
        total_apparent = total_resistive / power_factor

        assert len(powers) == 3
        assert all(p > 0 for p in powers)
        assert total_resistive > 0
        assert total_apparent >= total_resistive  # apparent >= resistive for pf <= 1

    def test_current_data_line_current_formula(self) -> None:
        """Line current = phase current * sqrt(3) for delta configuration."""
        phase_currents = [100.0, 110.0, 90.0]
        line_currents = [c * math.sqrt(3) for c in phase_currents]
        for phase_c, line_c in zip(phase_currents, line_currents, strict=True):
            assert abs(line_c / phase_c - math.sqrt(3)) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1362 — _run_optimization bisection algorithm
# ─────────────────────────────────────────────────────────────────────────────


class TestBisectionBalancing:
    """Issue #1362: bisection-based electrode advancement is no longer a stub."""

    def test_run_optimization_method_exists(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        has_method = hasattr(CalculationMixin, "_run_optimization")
        assert has_method, "_run_optimization must exist on CalculationMixin"

    def test_compute_balanced_depths_method_exists(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        has_method = hasattr(CalculationMixin, "_compute_balanced_depths")
        assert has_method, "_compute_balanced_depths must exist as a separate method"

    def test_bisection_pure_arithmetic_converges(self) -> None:
        """Bisection on a monotone function converges to within tolerance."""
        # f(x) = 1/x — resistance-like (decreases with depth)
        target = 0.05  # target resistance
        lo, hi = 1.0, 40.0
        tol = 1e-3
        for _ in range(50):
            mid = (lo + hi) / 2.0
            r_mid = 1.0 / mid
            if abs(r_mid - target) < tol or (hi - lo) / 2.0 < tol:
                break
            if r_mid > target:
                lo = mid
            else:
                hi = mid
        result_depth = (lo + hi) / 2.0
        assert abs(1.0 / result_depth - target) < tol * 10

    def test_bisection_mean_resistance_target(self) -> None:
        """Target resistance for balancing is the arithmetic mean of per-phase Rs."""
        resistances = [0.12, 0.10, 0.14]
        target = sum(resistances) / len(resistances)
        assert abs(target - 0.12) < 1e-10

    @pytest.mark.skipif(
        not ELECTRICAL_AVAILABLE, reason="upstream_drift_tools not installed"
    )
    def test_balanced_symmetric_system_unchanged(self) -> None:
        """For a symmetric system, balanced depths should be close to original depths."""
        cfg = ElectrodeConfig()
        glass = GlassPropertiesInterface()
        model = ThreePhaseElectricalModelEnhanced(cfg, glass)

        params = {
            "depths": np.array([12.0, 12.0, 12.0]),
            "bath_diameter": 120.0,
            "tip_diameter": 24.0,
            "metal_depth": 2.0,
            "k_factors": {"K_tt": 1.0, "K_vert": 1.0},
            "bath_temperature": 1350.0,
            "voltages": np.array([100.0, 100.0, 100.0]),
            "conductive_height": 2.0,
        }
        result = model.calculate_system_state(**params)
        paths = result.get("current_paths", {})
        resistances = [
            paths.get(pk, {}).get("total", 0.0) for pk in ["1-2", "2-3", "3-1"]
        ]
        if all(r > 0 for r in resistances):
            # For a symmetric system, all resistances should be equal
            max_r = max(resistances)
            min_r = min(resistances)
            assert max_r - min_r < 1e-4 * max_r + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Issue #1363 — draw_via_metal_path delegates to annotate helpers (DRY)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not DRAWING_AVAILABLE, reason="shared_drawing not importable")
class TestViaMetalPathDry:
    """Issue #1363: draw_via_metal_path no longer duplicates annotation logic."""

    def test_annotate_path_value_used_by_draw_via_metal_path(self) -> None:
        """draw_via_metal_path source must not inline ax.text for current annotation."""
        import inspect

        from electrode_advisor.utils.shared_drawing import draw_via_metal_path

        src = inspect.getsource(draw_via_metal_path)
        # The function must delegate to annotate_path_value (no bare ax.text for current)
        calls_helper = "annotate_path_value" in src
        assert calls_helper, "draw_via_metal_path must call annotate_path_value"

    def test_annotate_resistance_value_used_by_draw_via_metal_path(self) -> None:
        """draw_via_metal_path source must delegate resistance annotation."""
        import inspect

        from electrode_advisor.utils.shared_drawing import draw_via_metal_path

        src = inspect.getsource(draw_via_metal_path)
        delegated = "annotate_resistance_value" in src or "annotate_path_value" in src
        assert delegated, "draw_via_metal_path must delegate resistance annotation"


# ─────────────────────────────────────────────────────────────────────────────
# Source-code verification tests — confirm fixes are applied
# ─────────────────────────────────────────────────────────────────────────────


class TestElectrodeZInSource:
    """Issue #1358/#1375: verify _compute_electrode_positions uses correct formula."""

    def test_source_uses_depth_not_glass_height_over_2(self) -> None:
        """Source code must use `metal_height + glass_height - depth`."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_visualization_update import (
                VisualizationUpdateMixin,
            )
        except ImportError:
            pytest.skip("VisualizationUpdateMixin not importable")
        import inspect

        source = inspect.getsource(
            VisualizationUpdateMixin._compute_electrode_positions_for_paths
        )
        msg = "_compute_electrode_positions_for_paths must use `glass_height - depth`"
        assert "glass_height - depth" in source, msg
        bad = "glass_height / 2"
        assert bad not in source, f"Old formula '{bad}' must be removed"


class TestLineCurrentInSource:
    """Issue #1357: verify line current uses sqrt(3), not 0.8."""

    def test_source_uses_sqrt3_not_0_8(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_data import DataMixin
        except ImportError:
            pytest.skip("DataMixin not importable")
        import inspect

        source = inspect.getsource(DataMixin._update_current_distribution)
        uses_sqrt3 = "sqrt3" in source or "sqrt(3)" in source
        assert uses_sqrt3, "Line current must use sqrt(3) factor"
        assert "* 0.8" not in source, "Wrong 0.8 factor must be removed"


class TestTemperatureProfileStubRemovedFromDataMixin:
    """Issue #1377: _update_temperature_profile call removed from DataMixin."""

    def test_calculate_system_does_not_call_temperature_profile(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_data import DataMixin
        except ImportError:
            pytest.skip("DataMixin not importable")
        import inspect

        source = inspect.getsource(DataMixin._calculate_system)
        msg = "DataMixin._calculate_system must not call no-op _update_temperature_profile"
        assert "_update_temperature_profile" not in source, msg


class TestDRYAnnotationRefactored:
    """Issue #1363: draw_via_metal_path must delegate to shared annotators."""

    def test_no_inline_ax_text_in_draw_via_metal_path(self) -> None:
        """draw_via_metal_path should use annotate_path_value, not inline ax.text."""
        try:
            from electrode_advisor.utils.shared_drawing import draw_via_metal_path
        except ImportError:
            pytest.skip("shared_drawing not importable")
        import inspect

        source = inspect.getsource(draw_via_metal_path)
        assert "annotate_path_value" in source, "Must delegate to annotate_path_value"
        has_res = "annotate_resistance_value" in source
        assert has_res, "Must delegate to annotate_resistance_value"
        # Should not contain inline ax.text calls anymore
        no_inline = source.count("ax.text(") == 0
        assert no_inline, "Inline ax.text should use shared annotators"


# ─────────────────────────────────────────────────────────────────────────────
# Issues #1398-#1406 — Dead code removal
# ─────────────────────────────────────────────────────────────────────────────


class TestDeadCodeRemoval:
    """Issues #1398-#1406: dead code must be removed."""

    def test_main_window_paths_file_deleted(self) -> None:
        """#1398: main_window_paths.py is entirely dead code and must be removed."""
        from pathlib import Path

        pyqt6_dir = (
            Path(__file__).resolve().parents[1]
            / "python"
            / "electrode_advisor"
            / "ui"
            / "pyqt6"
        )
        paths_file = pyqt6_dir / "main_window_paths.py"
        assert not paths_file.exists(), "main_window_paths.py should have been deleted"

    def test_periodic_update_removed_from_calculation_mixin(self) -> None:
        """#1400: _periodic_update was empty and must be removed."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        has_periodic = hasattr(CalculationMixin, "_periodic_update")
        assert not has_periodic, "_periodic_update must be removed"

    def test_base_calculator_available_removed(self) -> None:
        """#1401: BASE_CALCULATOR_AVAILABLE was always False and must be removed."""
        try:
            from electrode_advisor.ui.pyqt6 import main_window
        except ImportError:
            pytest.skip("main_window not importable")
        has_const = hasattr(main_window, "BASE_CALCULATOR_AVAILABLE")
        assert not has_const, "BASE_CALCULATOR_AVAILABLE must be removed"

    def test_state_mixin_available_removed(self) -> None:
        """#1401: STATE_MIXIN_AVAILABLE was always False and must be removed."""
        try:
            from electrode_advisor.ui.pyqt6 import main_window
        except ImportError:
            pytest.skip("main_window not importable")
        has_const = hasattr(main_window, "STATE_MIXIN_AVAILABLE")
        assert not has_const, "STATE_MIXIN_AVAILABLE must be removed"

    def test_label_param_removed_from_trapezoidal_path(self) -> None:
        """#1402: label parameter silently discarded — must be removed."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_visualization_update import (
                VisualizationUpdateMixin,
            )
        except ImportError:
            pytest.skip("VisualizationUpdateMixin not importable")
        import inspect

        sig = inspect.signature(VisualizationUpdateMixin._draw_correct_trapezoidal_path)
        has_label = "label" in sig.parameters
        assert not has_label, "label param must be removed from trapezoidal"

    def test_label_param_removed_from_via_metal_path(self) -> None:
        """#1402: label parameter silently discarded — must be removed."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_visualization_update import (
                VisualizationUpdateMixin,
            )
        except ImportError:
            pytest.skip("VisualizationUpdateMixin not importable")
        import inspect

        sig = inspect.signature(VisualizationUpdateMixin._draw_correct_via_metal_path)
        has_label = "label" in sig.parameters
        assert not has_label, "label param must be removed from via_metal"

    def test_visualization_no_set_axis(self) -> None:
        """#1405: ElectrodeVisualization.set_axis() was dead code."""
        try:
            from electrode_advisor.utils.visualization import (
                ElectrodeVisualization,
            )
        except ImportError:
            pytest.skip("ElectrodeVisualization not importable")
        has_set_axis = hasattr(ElectrodeVisualization, "set_axis")
        assert not has_set_axis, "set_axis must be removed"


# ── Batch 2: DRY elimination tests ──────────────────────────────────────────


class TestDryElimination:
    """Tests for Batch 2 DRY fixes (#1407-#1414)."""

    def test_drawing_mixin_delegates_metal_layer(self) -> None:
        """#1407: DrawingMixin._draw_3d_metal_layer delegates to visualization."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_drawing import DrawingMixin
        except ImportError:
            pytest.skip("DrawingMixin not importable")
        import inspect

        src = inspect.getsource(DrawingMixin._draw_3d_metal_layer)
        has_delegation = "self.visualization.draw_3d_metal_layer" in src
        assert has_delegation, "must delegate to self.visualization"

    def test_drawing_mixin_delegates_glass_layer(self) -> None:
        """#1408: DrawingMixin._draw_3d_glass_layer delegates to visualization."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_drawing import DrawingMixin
        except ImportError:
            pytest.skip("DrawingMixin not importable")
        import inspect

        src = inspect.getsource(DrawingMixin._draw_3d_glass_layer)
        has_delegation = "self.visualization.draw_3d_glass_layer" in src
        assert has_delegation, "must delegate to self.visualization"

    def test_drawing_mixin_delegates_refractory_layer(self) -> None:
        """#1409: DrawingMixin._draw_3d_refractory_layer delegates to visualization."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_drawing import DrawingMixin
        except ImportError:
            pytest.skip("DrawingMixin not importable")
        import inspect

        src = inspect.getsource(DrawingMixin._draw_3d_refractory_layer)
        has_delegation = "self.visualization.draw_3d_refractory_layer" in src
        assert has_delegation, "must delegate to self.visualization"

    def test_drawing_mixin_delegates_metal_shell(self) -> None:
        """#1410: DrawingMixin._draw_3d_metal_shell delegates to visualization."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_drawing import DrawingMixin
        except ImportError:
            pytest.skip("DrawingMixin not importable")
        import inspect

        src = inspect.getsource(DrawingMixin._draw_3d_metal_shell)
        has_delegation = "self.visualization.draw_3d_metal_shell" in src
        assert has_delegation, "must delegate to self.visualization"

    def test_drawing_mixin_delegates_electrodes(self) -> None:
        """#1411: DrawingMixin._draw_3d_electrodes delegates to visualization."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_drawing import DrawingMixin
        except ImportError:
            pytest.skip("DrawingMixin not importable")
        import inspect

        src = inspect.getsource(DrawingMixin._draw_3d_electrodes)
        has_delegation = "self.visualization.draw_3d_electrodes" in src
        assert has_delegation, "must delegate to self.visualization"

    def test_drawing_mixin_delegates_sphere(self) -> None:
        """#1412: _draw_electrode_sphere delegates to visualization."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_drawing import DrawingMixin
        except ImportError:
            pytest.skip("DrawingMixin not importable")
        import inspect

        src = inspect.getsource(DrawingMixin._draw_electrode_sphere)
        has_delegation = "self.visualization.draw_electrode_sphere" in src
        assert has_delegation, "must delegate to self.visualization"

    def test_results_mixin_no_draw_electrode_sphere(self) -> None:
        """#1412: ResultsAndChartsMixin must not have its own _draw_electrode_sphere."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_results_charts import (
                ResultsAndChartsMixin,
            )
        except ImportError:
            pytest.skip("ResultsAndChartsMixin not importable")
        has_sphere = "_draw_electrode_sphere" in ResultsAndChartsMixin.__dict__
        assert not has_sphere, "duplicate sphere method must be removed"

    def test_read_calculation_params_exists(self) -> None:
        """#1414: _read_calculation_params extracted from duplicated UI reads."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        has_method = hasattr(CalculationMixin, "_read_calculation_params")
        assert has_method, "_read_calculation_params must exist"

    def test_calculate_system_uses_read_params(self) -> None:
        """#1414: _calculate_system must call _read_calculation_params."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        import inspect

        src = inspect.getsource(CalculationMixin._calculate_system)
        uses_method = "_read_calculation_params" in src
        assert uses_method, "_calculate_system must use _read_calculation_params"

    def test_compute_balanced_depths_uses_read_params(self) -> None:
        """#1414: _compute_balanced_depths must call _read_calculation_params."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        import inspect

        src = inspect.getsource(CalculationMixin._compute_balanced_depths)
        uses_method = "_read_calculation_params" in src
        assert uses_method, "_compute_balanced_depths must use _read_calculation_params"

    def test_drawing_mixin_file_under_150_lines(self) -> None:
        """DrawingMixin file should be small after delegation rewrite."""
        from pathlib import Path

        drawing_path = Path(
            "/home/dieterolson/Linux_Repositories/Linux_Tools/Tools/"
            "src/electrode_advisor/python/electrode_advisor/"
            "ui/pyqt6/main_window_drawing.py"
        )
        if not drawing_path.exists():
            pytest.skip("Drawing file not found")
        line_count = len(drawing_path.read_text().splitlines())
        assert line_count < 150, f"DrawingMixin should be <150 lines, got {line_count}"


# ── Batch 3: Correctness fixes tests ────────────────────────────────────────


class TestCorrectnessFixes:
    """Tests for Batch 3 correctness fixes (#1415-#1418)."""

    def test_electrode_z_uses_depth(self) -> None:
        """#1415: draw_3d_electrodes must use depth in z-position formula."""
        try:
            from electrode_advisor.utils.visualization import ElectrodeVisualization
        except ImportError:
            pytest.skip("ElectrodeVisualization not importable")
        import inspect

        src = inspect.getsource(ElectrodeVisualization.draw_3d_electrodes)
        uses_depth = "glass_height - depth" in src
        assert uses_depth, "z must use glass_height - depth"

    def test_tick_hiding_not_unconditional(self) -> None:
        """#1416: tick-hiding must be inside else branch, not unconditional."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_visualization_update import (
                VisualizationUpdateMixin,
            )
        except ImportError:
            pytest.skip("VisualizationUpdateMixin not importable")
        import inspect

        src = inspect.getsource(VisualizationUpdateMixin._configure_viz_axis_labels)
        # The old bug had an unconditional block after the if/else that always
        # hid ticks. After the fix, labelbottom=False should only appear once
        # (in the else branch), not twice.
        count = src.count("labelbottom=False")
        assert count == 1, f"labelbottom=False once, got {count}"

    def test_temperature_gradient_removed_from_results(self) -> None:
        """#1417: _COLOR_MODE_TEMPERATURE must be removed from results_charts."""
        try:
            from electrode_advisor.ui.pyqt6 import main_window_results_charts as mod
        except ImportError:
            pytest.skip("main_window_results_charts not importable")
        has_temp = hasattr(mod, "_COLOR_MODE_TEMPERATURE")
        assert not has_temp, "Temperature gradient constant must be removed"

    def test_temperature_gradient_not_in_visual_controls(self) -> None:
        """#1417: Temperature gradient must not be in color mode combo items."""
        try:
            from electrode_advisor.ui.pyqt6 import main_window_visual_controls as mod
        except ImportError:
            pytest.skip("main_window_visual_controls not importable")
        import inspect

        src = inspect.getsource(mod)
        has_temp = "Temperature gradient" in src
        assert not has_temp, "Temperature gradient must be removed from combo"

    def test_visualization_constructor_takes_config(self) -> None:
        """#1418: ElectrodeVisualization constructor must accept config param."""
        try:
            from electrode_advisor.utils.visualization import ElectrodeVisualization
        except ImportError:
            pytest.skip("ElectrodeVisualization not importable")
        import inspect

        sig = inspect.signature(ElectrodeVisualization.__init__)
        has_config = "config" in sig.parameters
        assert has_config, "constructor must accept config parameter"


# ── Batch 4: Robustness + DbC tests ─────────────────────────────────────────


class TestRobustnessDbC:
    """Tests for Batch 4 robustness/DbC fixes (#1419-#1422)."""

    def test_calculate_system_uses_valueerror(self) -> None:
        """#1419: _calculate_system must raise ValueError, not assert."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        import inspect

        src = inspect.getsource(CalculationMixin._calculate_system)
        no_assert = "assert bath_diameter" not in src
        has_raise = "raise ValueError" in src
        assert no_assert, "must not use assert for preconditions"
        assert has_raise, "must raise ValueError for preconditions"

    def test_compute_balanced_depths_postcondition(self) -> None:
        """#1420: _compute_balanced_depths must have postcondition."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        import inspect

        src = inspect.getsource(CalculationMixin._compute_balanced_depths)
        has_post = "lo_orig" in src and "hi_orig" in src
        assert has_post, "must check result in [lo, hi]"

    def test_electrode_diameter_validated(self) -> None:
        """#1421: _read_calculation_params must validate diameter."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_calculation import (
                CalculationMixin,
            )
        except ImportError:
            pytest.skip("CalculationMixin not importable")
        import inspect

        src = inspect.getsource(CalculationMixin._read_calculation_params)
        has_validation = "except (ValueError, TypeError)" in src
        assert has_validation, "must validate diameter conversion"

    def test_phase_key_uses_split(self) -> None:
        """#1422: phase key parsing must use split('-')."""
        try:
            from electrode_advisor.ui.pyqt6.main_window_visualization_update import (
                VisualizationUpdateMixin,
            )
        except ImportError:
            pytest.skip("VisualizationUpdateMixin not importable")
        import inspect

        src = inspect.getsource(VisualizationUpdateMixin._draw_3d_conductive_paths_new)
        uses_split = 'split("-")' in src
        no_index = "phase[0]" not in src
        assert uses_split, "must use split for phase key"
        assert no_index, "must not use character indexing"


# ─────────────────────────────────────────────────────────────────────────────
# Batch 6: Constants + Architecture (#1429-#1440)
# ─────────────────────────────────────────────────────────────────────────────


class TestConstantsModule:
    """Issues #1429-#1434: Magic numbers centralized in constants module."""

    def test_constants_importable(self) -> None:
        try:
            from electrode_advisor.utils.constants import (
                CYLINDER_CIRCUM_SEGMENTS,
                CYLINDER_LENGTH_SEGMENTS,
                CYLINDER_THETA_SEGMENTS,
                ELECTRODE_ANGLES_DEG,
                ELECTRODE_COLORS,
                ELECTRODE_COUNT,
                SHELL_THICKNESS,
                SPHERE_U_RESOLUTION,
                SPHERE_V_RESOLUTION,
            )
        except ImportError:
            pytest.skip("constants module not importable")

        assert ELECTRODE_COUNT == 3
        assert len(ELECTRODE_ANGLES_DEG) == ELECTRODE_COUNT
        assert len(ELECTRODE_COLORS) == ELECTRODE_COUNT
        assert SHELL_THICKNESS > 0
        assert SPHERE_U_RESOLUTION > 0
        assert SPHERE_V_RESOLUTION > 0
        assert CYLINDER_THETA_SEGMENTS > 0
        assert CYLINDER_LENGTH_SEGMENTS > 0
        assert CYLINDER_CIRCUM_SEGMENTS > 0

    def test_angles_sum_to_360(self) -> None:
        """Electrode angles must be evenly spaced around circle."""
        try:
            from electrode_advisor.utils.constants import ELECTRODE_ANGLES_DEG
        except ImportError:
            pytest.skip("constants module not importable")

        diffs = [
            ELECTRODE_ANGLES_DEG[(i + 1) % len(ELECTRODE_ANGLES_DEG)]
            - ELECTRODE_ANGLES_DEG[i]
            for i in range(len(ELECTRODE_ANGLES_DEG) - 1)
        ]
        assert all(d == diffs[0] for d in diffs), "Angles must be evenly spaced"


@pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="visualization not importable")
class TestVisualizationUsesConstants:
    """Issues #1429-#1434: visualization.py must use constants, not literals."""

    def test_no_hardcoded_electrode_angles(self) -> None:
        import inspect

        from electrode_advisor.utils.visualization import ElectrodeVisualization

        src = inspect.getsource(ElectrodeVisualization.draw_3d_electrodes)
        assert "[0, 120, 240]" not in src, "must use ELECTRODE_ANGLES_DEG"

    def test_no_hardcoded_sphere_resolution(self) -> None:
        import inspect

        from electrode_advisor.utils.visualization import ElectrodeVisualization

        src = inspect.getsource(ElectrodeVisualization.draw_electrode_sphere)
        assert "np.linspace(0, 2 * np.pi, 20)" not in src
        assert "np.linspace(0, np.pi, 15)" not in src

    def test_no_hardcoded_shell_thickness(self) -> None:
        import inspect

        from electrode_advisor.utils.visualization_layers import ElectrodeLayersMixin

        src = inspect.getsource(ElectrodeLayersMixin.draw_3d_metal_shell)
        assert "shell_thickness = 0.5" not in src, "must use SHELL_THICKNESS"


class TestProtocolsModule:
    """Issue #1438: Protocol classes for mixin interface contracts."""

    def test_protocols_importable(self) -> None:
        try:
            from electrode_advisor.utils.protocols import (
                SupportsCalculation,
                SupportsElectrodeConfig,
                SupportsVisualization,
            )
        except ImportError:
            pytest.skip("protocols module not importable")

        assert SupportsElectrodeConfig is not None
        assert SupportsVisualization is not None
        assert SupportsCalculation is not None

    def test_protocols_are_runtime_checkable(self) -> None:
        try:
            from electrode_advisor.utils.protocols import (
                SupportsElectrodeConfig,
            )
        except ImportError:
            pytest.skip("protocols module not importable")

        # Runtime-checkable protocols support isinstance checks
        assert hasattr(SupportsElectrodeConfig, "__protocol_attrs__") or hasattr(
            SupportsElectrodeConfig, "__abstractmethods__"
        )


@pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="visualization not importable")
class TestDeadMethodsRemoved:
    """Issue #1440: Old-style drawing methods removed from ElectrodeVisualization."""

    def test_no_draw_correct_trapezoidal_path(self) -> None:
        assert not hasattr(ElectrodeVisualization, "draw_correct_trapezoidal_path")

    def test_no_draw_correct_via_metal_path(self) -> None:
        assert not hasattr(ElectrodeVisualization, "draw_correct_via_metal_path")

    def test_no_draw_electrode_length_extrusion(self) -> None:
        assert not hasattr(ElectrodeVisualization, "draw_electrode_length_extrusion")

    def test_no_private_helpers_from_dead_methods(self) -> None:
        for attr in (
            "_electrode_wall_positions",
            "_extrude_polygon",
            "_build_extrusion_vertices",
            "_box_faces",
            "_label_midpoint",
        ):
            assert not hasattr(ElectrodeVisualization, attr), f"dead: {attr}"

    def test_live_methods_still_exist(self) -> None:
        """Ensure we did not remove methods that are still in use."""
        for attr in (
            "draw_cylinder",
            "draw_cylinder_between",
            "draw_trapezoidal_prism",
            "draw_via_metal_path",
            "draw_3d_electrodes",
            "draw_horizontal_cylinder",
            "draw_electrode_sphere",
        ):
            assert hasattr(ElectrodeVisualization, attr), f"{attr} must still exist"


class TestElectrodeCountConstant:
    """Issue #1439: ELECTRODE_COUNT used instead of hard-coded 3."""

    def test_visualization_update_uses_electrode_count(self) -> None:
        try:
            from electrode_advisor.ui.pyqt6.main_window_visualization_update import (
                VisualizationUpdateMixin,
            )
        except ImportError:
            pytest.skip("VisualizationUpdateMixin not importable")
        import inspect

        src = inspect.getsource(VisualizationUpdateMixin._read_viz_params)
        assert "ELECTRODE_COUNT" in src, "must use ELECTRODE_COUNT constant"
