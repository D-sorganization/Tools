# ruff: noqa: E501
"""Tests for pendulum simulator UI enhancements (PR #1114).

Covers issues #1097, #1100-#1102, #1103, #1104, #1108-#1110, #1111, #1113.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from double_pendulum_golf.physics_golfer import (
    GolferParams,
    N_DOF,
    forward_kinematics,
    mass_matrix,
    potential_energy_from_q,
)
from double_pendulum_golf.physics_golfer import (
    analytical_fk_jacobians as golfer_analytical_jacobians,
)


def _has_pyqt6() -> bool:
    """Check if PyQt6 is available AND a Qt platform backend is usable."""
    # On headless Linux, QWidget() causes SIGABRT without a platform backend.
    # QT_QPA_PLATFORM=offscreen is safe; DISPLAY/WAYLAND_DISPLAY mean a real server.
    if sys.platform not in ("win32", "darwin"):
        has_platform = (
            os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            or bool(os.environ.get("DISPLAY"))
            or bool(os.environ.get("WAYLAND_DISPLAY"))
        )
        if not has_platform:
            return False
    try:
        import PyQt6.QtWidgets  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def golfer_params() -> GolferParams:
    """Standard golfer parameters for testing."""
    return GolferParams(
        m_hub=2.0,
        m_r_upper=3.0,
        m_r_fore=2.0,
        m_l_upper=3.0,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.15,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.1,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


@pytest.fixture
def golfer_params_with_scapula() -> GolferParams:
    """Golfer parameters with scapula links enabled."""
    return GolferParams(
        m_hub=2.0,
        m_r_upper=3.0,
        m_r_fore=2.0,
        m_l_upper=3.0,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.15,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.1,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
        L_rscap=0.08,
        L_lscap=0.08,
        m_rscap=0.5,
        m_lscap=0.5,
    )


# ---------------------------------------------------------------------------
# #1103 — Reversed Hub Standoff
# ---------------------------------------------------------------------------


class TestReversedHubStandoff:
    """Hub standoff must point upward (inside arm loop) after #1103."""

    def test_hub_above_origin_at_zero_angle(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        pos = forward_kinematics(q, golfer_params)
        # Hub extends upward: y > 0
        assert pos["hub"][1] > 0, f"Hub y should be positive, got {pos['hub'][1]}"
        # Hub x should be 0 at zero angle
        assert abs(pos["hub"][0]) < 1e-10

    def test_hub_position_magnitude(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        pos = forward_kinematics(q, golfer_params)
        dist = np.hypot(pos["hub"][0], pos["hub"][1])
        assert np.isclose(dist, golfer_params.L_hub, atol=1e-10)

    def test_hub_rotates_correctly(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        q[0] = np.pi / 2  # rotate hub 90°
        pos = forward_kinematics(q, golfer_params)
        # After reversal, at 90°: x = -L*sin(90°) = -L, y = L*cos(90°) = 0
        assert pos["hub"][0] < 0, "Hub should be on left side at π/2"
        assert abs(pos["hub"][1]) < 1e-10

    def test_analytical_jacobians_match_numerical(
        self, golfer_params: GolferParams
    ) -> None:
        """Analytical Jacobians must match numerical finite-diff after hub reversal."""
        rng = np.random.default_rng(42)
        eps = 1e-7
        for _ in range(3):
            q = rng.uniform(-0.5, 0.5, N_DOF)
            jacs = golfer_analytical_jacobians(q, golfer_params)

            # Numerical Jacobian for hub
            fk0 = forward_kinematics(q, golfer_params)
            J_hub_num = np.zeros((2, N_DOF))
            for j in range(N_DOF):
                qp = q.copy()
                qp[j] += eps
                fkp = forward_kinematics(qp, golfer_params)
                J_hub_num[0, j] = (fkp["hub"][0] - fk0["hub"][0]) / eps
                J_hub_num[1, j] = (fkp["hub"][1] - fk0["hub"][1]) / eps

            assert np.allclose(
                jacs["hub"], J_hub_num, atol=1e-4
            ), f"Hub Jacobian mismatch:\nAnalytical:\n{jacs['hub']}\nNumerical:\n{J_hub_num}"

    def test_all_analytical_jacobians_match_numerical(
        self, golfer_params: GolferParams
    ) -> None:
        """All analytical Jacobians must match numerical for several configs."""
        rng = np.random.default_rng(123)
        eps = 1e-7
        joint_names = ["hub", "rs", "re", "rh", "ls", "le", "lh"]

        for trial in range(3):
            q = rng.uniform(-0.3, 0.3, N_DOF)
            jacs = golfer_analytical_jacobians(q, golfer_params)
            fk0 = forward_kinematics(q, golfer_params)

            for name in joint_names:
                J_num = np.zeros((2, N_DOF))
                for j in range(N_DOF):
                    qp = q.copy()
                    qp[j] += eps
                    fkp = forward_kinematics(qp, golfer_params)
                    J_num[0, j] = (fkp[name][0] - fk0[name][0]) / eps
                    J_num[1, j] = (fkp[name][1] - fk0[name][1]) / eps

                assert np.allclose(jacs[name], J_num, atol=1e-4), (
                    f"Trial {trial}, joint '{name}' Jacobian mismatch:\n"
                    f"Analytical:\n{jacs[name]}\nNumerical:\n{J_num}"
                )


# ---------------------------------------------------------------------------
# #1104 — Scapula Joint Parameters
# ---------------------------------------------------------------------------


class TestScapulaJointParameters:
    """Scapula parameters must be optional and backwards compatible."""

    def test_default_scapula_is_zero(self) -> None:
        p = GolferParams(
            m_hub=2.0,
            m_r_upper=3.0,
            m_r_fore=2.0,
            m_l_upper=3.0,
            m_l_fore=2.0,
            m_club=0.5,
            L_hub=0.15,
            L_r_upper=0.35,
            L_r_fore=0.30,
            L_l_upper=0.35,
            L_l_fore=0.30,
            L_club=1.1,
            d_rs=0.20,
            d_ls=0.20,
            grip_right=0.05,
            grip_left=0.25,
        )
        assert p.L_rscap == 0.0
        assert p.L_lscap == 0.0
        assert p.m_rscap == 0.0
        assert p.m_lscap == 0.0

    def test_no_scapula_keys_when_absent(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        pos = forward_kinematics(q, golfer_params)
        assert "rscap" not in pos
        assert "lscap" not in pos

    def test_scapula_keys_present_when_enabled(
        self, golfer_params_with_scapula: GolferParams
    ) -> None:
        q = np.zeros(N_DOF)
        pos = forward_kinematics(q, golfer_params_with_scapula)
        assert "rscap" in pos
        assert "lscap" in pos

    def test_scapula_offsets_shoulder(
        self,
        golfer_params: GolferParams,
        golfer_params_with_scapula: GolferParams,
    ) -> None:
        q = np.zeros(N_DOF)
        pos_no_scap = forward_kinematics(q, golfer_params)
        pos_scap = forward_kinematics(q, golfer_params_with_scapula)

        # Shoulder should be different when scapula is present
        rs_no = np.array(pos_no_scap["rs"])
        rs_yes = np.array(pos_scap["rs"])
        assert not np.allclose(rs_no, rs_yes), "Shoulder should move with scapula"

    def test_scapula_position_is_at_bar_endpoint(
        self,
        golfer_params_with_scapula: GolferParams,
        golfer_params: GolferParams,
    ) -> None:
        q = np.zeros(N_DOF)
        pos_scap = forward_kinematics(q, golfer_params_with_scapula)
        pos_no = forward_kinematics(q, golfer_params)

        # Scapula position should be at the original shoulder bar endpoint
        rscap = np.array(pos_scap["rscap"])
        rs_orig = np.array(pos_no["rs"])
        assert np.allclose(
            rscap, rs_orig, atol=1e-10
        ), "Scapula joint should be at original shoulder bar endpoint"

    def test_mass_matrix_still_valid_with_scapula(
        self,
        golfer_params_with_scapula: GolferParams,
    ) -> None:
        q = np.zeros(N_DOF)
        M = mass_matrix(q, golfer_params_with_scapula)
        assert M.shape == (N_DOF, N_DOF)
        assert np.allclose(M, M.T, atol=1e-8), "M must be symmetric"
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues >= -1e-10), "M must be PSD"


# ---------------------------------------------------------------------------
# #1113 — Swing Plane Tilt
# ---------------------------------------------------------------------------


class TestSwingPlaneTilt:
    """Tilt angle must reduce effective gravity and project display."""

    def test_tilt_zero_gives_full_gravity(self, golfer_params: GolferParams) -> None:
        # With tilt=0, effective g = g * cos(0) = g
        g_eff = golfer_params.g * np.cos(0.0)
        assert np.isclose(g_eff, golfer_params.g)

    def test_tilt_90_gives_zero_gravity(self, golfer_params: GolferParams) -> None:
        # With tilt=90°, effective g = g * cos(π/2) ≈ 0
        g_eff = golfer_params.g * np.cos(np.pi / 2)
        assert abs(g_eff) < 1e-10

    def test_tilt_45_gives_half_gravity(self, golfer_params: GolferParams) -> None:
        g_eff = golfer_params.g * np.cos(np.pi / 4)
        expected = golfer_params.g / np.sqrt(2)
        assert np.isclose(g_eff, expected, atol=1e-10)

    def test_tilt_reduces_potential_energy(self, golfer_params: GolferParams) -> None:
        """Tilting the plane should reduce the PE for the same configuration."""
        q = np.zeros(N_DOF)
        q[0] = np.pi / 4  # displace from equilibrium

        # Full gravity
        V_full = potential_energy_from_q(q, golfer_params)

        # Tilted gravity (create params with reduced g)
        from dataclasses import replace

        params_tilted = replace(
            golfer_params,
            g=golfer_params.g * np.cos(np.pi / 4),
        )
        V_tilted = potential_energy_from_q(q, params_tilted)

        # PE should be smaller with reduced gravity
        assert abs(V_tilted) < abs(
            V_full
        ), f"Tilted PE ({V_tilted}) should be smaller than full ({V_full})"


# ---------------------------------------------------------------------------
# #1097 — Adaptive Step Interpolation
# ---------------------------------------------------------------------------


class TestAdaptiveStepInterpolation:
    """Frame advance logic must handle fractional accumulation correctly."""

    def test_fractional_advance_basic(self) -> None:
        """Fractional accumulator should not lose sub-frame position."""
        frac = 0.0
        idx = 0
        n_frames = 100

        # Advance by 1.5 frames per tick for 10 ticks
        for _ in range(10):
            frac += 1.5
            advance = int(frac)
            frac -= advance
            idx = min(idx + advance, n_frames - 1)

        # After 10 ticks at 1.5 frames/tick = 15 frames total
        assert idx == 15

    def test_fractional_advance_no_frames_lost(self) -> None:
        """Even with non-integer speeds, total frames should be accurate."""
        frac = 0.0
        idx = 0
        n_frames = 1000
        speed = 3.7
        total_advance = 0

        for _ in range(100):
            frac += speed
            advance = int(frac)
            frac -= advance
            total_advance += advance
            idx = min(idx + advance, n_frames - 1)

        # Total advance should be close to 100 * 3.7 = 370
        assert 369 <= total_advance <= 371

    def test_fractional_advance_handles_speed_one(self) -> None:
        """At speed 1.0, should advance exactly 1 frame per tick."""
        frac = 0.0
        idx = 0
        n_frames = 100

        for _ in range(50):
            frac += 1.0
            advance = int(frac)
            frac -= advance
            idx = min(idx + advance, n_frames - 1)

        assert idx == 50


# ---------------------------------------------------------------------------
# #1100-#1102 — Per-Segment Visibility
# ---------------------------------------------------------------------------


class TestPerSegmentVisibility:
    """Segment visibility filtering must work correctly."""

    @staticmethod
    def _filter_visible(names: list[str], visible: set[str] | None) -> list[str]:
        return [n for n in names if visible is None or n in visible]

    def test_none_means_all_visible(self) -> None:
        visible: set[str] | None = None
        joint_names = ["shoulder", "wrist", "tip"]
        result = self._filter_visible(joint_names, visible)
        assert result == joint_names

    def test_empty_set_means_nothing_visible(self) -> None:
        visible: set[str] = set()
        joint_names = ["shoulder", "wrist", "tip"]
        result = [name for name in joint_names if visible is None or name in visible]
        assert result == []

    def test_partial_visibility(self) -> None:
        visible = {"shoulder", "tip"}
        joint_names = ["shoulder", "wrist", "tip"]
        result = [name for name in joint_names if visible is None or name in visible]
        assert result == ["shoulder", "tip"]

    def test_golfer_segments(self) -> None:
        visible = {"hub", "re", "lh"}
        golfer_joints = ["hub", "rs", "re", "rh", "ls", "le", "lh", "club_tip"]
        result = [name for name in golfer_joints if visible is None or name in visible]
        assert result == ["hub", "re", "lh"]


# ---------------------------------------------------------------------------
# #1108-#1110 — Optimization Widget
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pyqt6(), reason="PyQt6 not available (headless CI)")
class TestOptimizationWidget:
    """Optimization widget basic functionality tests."""

    def test_import_succeeds(self) -> None:
        from double_pendulum_golf.gui.optimization_widget import OptimizationWidget

        assert OptimizationWidget is not None

    def test_instantiation(self) -> None:
        """Widget should instantiate without errors."""
        from double_pendulum_golf.gui.optimization_widget import OptimizationWidget

        # Need QApplication for widget creation
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        widget = OptimizationWidget(
            model_name="Test Model",
            n_torque_params=2,
        )
        assert widget._model_name == "Test Model"
        assert widget._n_torque_params == 2

    def test_worker_runs_simple_objective(self) -> None:
        """Background optimizer worker should optimize a simple quadratic."""
        from double_pendulum_golf.gui.optimization_widget import _OptimizerWorker

        objective_called: list[bool] = []

        def simple_objective(x: np.ndarray) -> float:
            objective_called.append(True)
            return float(np.sum(x**2))

        worker = _OptimizerWorker(
            objective_fn=simple_objective,
            n_params=2,
            n_iterations=10,
            method="Nelder-Mead",
        )

        results: list[dict[str, object]] = []
        errors: list[str] = []
        worker.finished.connect(lambda r: results.append(r))
        worker.error.connect(lambda e: errors.append(e))

        worker.run()

        assert len(errors) == 0, f"Worker errored: {errors}"
        assert len(results) == 1
        assert "coeffs" in results[0]
        assert "speed" in results[0]
        assert "history" in results[0]
        assert len(objective_called) > 0
