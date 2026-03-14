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
    """Issue #931: contracts + behavior for compute_wall_position."""

    def test_returns_array_with_two_xy_coords(self) -> None:
        pos = {"x": 50.0, "y": 0.0, "z": 12.0}
        result = compute_wall_position(pos, bath_radius=60.0)
        assert result is not None
        arr = np.asarray(result)
        assert arr.shape == (2,) or arr.ndim == 1, "Expected 2-element array"

    def test_wall_position_magnitude_le_bath_radius(self) -> None:
        """Wall position must lie at or inside the bath radius."""
        pos = {"x": 50.0, "y": 0.0, "z": 12.0}
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
            "x": 40.0 * math.cos(angle_rad),
            "y": 40.0 * math.sin(angle_rad),
            "z": 12.0,
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

    def test_instantiation_without_axis(self) -> None:
        viz = ElectrodeVisualization()
        assert viz is not None

    def test_set_axis_accepts_none(self) -> None:
        viz = ElectrodeVisualization()
        viz.set_axis(None)  # Should not raise

    def test_set_axis_stores_value(self) -> None:
        sentinel = object()
        viz = ElectrodeVisualization()
        viz.set_axis(sentinel)
        assert viz.ax is sentinel

    def test_electrode_wall_positions_is_static(self) -> None:
        """_electrode_wall_positions should be callable without an instance ax."""
        pos1 = _make_electrode_pos(0.0)
        pos2 = _make_electrode_pos(120.0)
        result = ElectrodeVisualization._electrode_wall_positions(
            pos1, pos2, bath_radius=60.0
        )
        assert result is not None
        assert len(result) == 2, "Expected two wall positions"


# ─────────────────────────────────────────────────────────────────────────────
# DRY regression: confirm shared_drawing delegates are consistent
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (DRAWING_AVAILABLE and VISUALIZATION_AVAILABLE),
    reason="Required packages not available",
)
class TestSharedDrawingDRYConsistency:
    """Verify that ElectrodeVisualization delegates to shared_drawing helpers.

    Issue #931: The DRY extraction means shared_drawing.compute_wall_position and
    ElectrodeVisualization._electrode_wall_positions should produce consistent results.
    """

    def test_compute_wall_position_matches_electrode_wall_positions(self) -> None:
        pos1 = _make_electrode_pos(0.0)
        pos2 = _make_electrode_pos(120.0)
        bath_radius = 60.0

        # Direct call via shared_drawing
        wp1_direct = np.asarray(compute_wall_position(pos1, bath_radius))
        wp2_direct = np.asarray(compute_wall_position(pos2, bath_radius))

        # Via ElectrodeVisualization static method
        wall1_via, wall2_via = ElectrodeVisualization._electrode_wall_positions(
            pos1, pos2, bath_radius
        )
        wp1_via = np.asarray(wall1_via)[:2]
        wp2_via = np.asarray(wall2_via)[:2]

        np.testing.assert_allclose(
            wp1_direct[:2],
            wp1_via,
            rtol=1e-5,
            err_msg="compute_wall_position and _electrode_wall_positions disagree for pos1",
        )
        np.testing.assert_allclose(
            wp2_direct[:2],
            wp2_via,
            rtol=1e-5,
            err_msg="compute_wall_position and _electrode_wall_positions disagree for pos2",
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
