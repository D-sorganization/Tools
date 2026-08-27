"""Direct-collocation transcription of the downswing optimal control problem.

States and torques at every node are decision variables; the dynamics are imposed
as equality constraints between consecutive nodes. This module owns the
transcription only — packing, defects, bounds, slew margins and the
non-dimensional scaling — so the solver module stays about solving.

The scaling is not cosmetic. SLSQP applies one trust region across the whole
decision vector, so mixing radians (order 1), angular rates (order 30) and
torques (order 200) makes it declare convergence after a handful of iterations
having done nothing but find feasibility. Solving in units of each variable's
characteristic scale drives the dynamics defects down by roughly ten orders of
magnitude.

Closes #4769.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from double_pendulum_golf.swing_objectives.downswing_config import (
    RATE_SCALE,
    DownswingConfig,
)
from double_pendulum_golf.swing_objectives.signals import generalized_accelerations

__all__ = ["CollocationTranscription"]

FloatArray = npt.NDArray[np.float64]

_STATE_WIDTH = 4
_TORQUE_WIDTH = 2
_HERMITE_MIDPOINT_WEIGHT = 8.0
_SIMPSON_WEIGHT = 6.0
_RELEASE_GUESS_ONSET = 0.45


class CollocationTranscription:
    """Packs, bounds and constrains a downswing collocation problem.

    Args:
        config: The immutable downswing configuration.
    """

    def __init__(self, config: DownswingConfig) -> None:
        """Initialize the transcription for one configuration."""
        self._config = config
        self._node_count = config.node_count
        self._state_var_count = _STATE_WIDTH * config.node_count

    # --- Packing --------------------------------------------------------------

    def pack(self, states: FloatArray, torques: FloatArray) -> FloatArray:
        """Flatten state and torque trajectories into one decision vector."""
        return np.concatenate([states.ravel(), torques.ravel()])

    def unpack(self, decision: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Split a decision vector back into state and torque trajectories."""
        states = decision[: self._state_var_count].reshape((self._node_count, _STATE_WIDTH))
        torques = decision[self._state_var_count :].reshape((self._node_count, _TORQUE_WIDTH))
        return states, torques

    # --- Scaling --------------------------------------------------------------

    @property
    def variable_scales(self) -> FloatArray:
        """Characteristic magnitude of every decision variable.

        Returns ones when scaling is disabled, so the same code path serves both
        the conditioned and unconditioned formulations.
        """
        if not self._config.use_variable_scaling:
            return np.ones(self._state_var_count + _TORQUE_WIDTH * self._node_count)
        state_scales = np.tile(self._config.state_scale, self._node_count)
        torque_scales = np.tile(self._config.torque_limit_vector, self._node_count)
        scales: FloatArray = np.concatenate([state_scales, torque_scales])
        return scales

    def scaled_bounds(self) -> list[tuple[float, float]]:
        """Variable bounds expressed in the same units as the scaled problem."""
        raw: list[tuple[float, float]] = []
        for _ in range(self._node_count):
            raw.extend(self._config.state_bounds())
        for _ in range(self._node_count):
            raw.extend(self._config.torque_bounds())
        scales = self.variable_scales
        return [
            (low / scale, high / scale) for (low, high), scale in zip(raw, scales, strict=True)
        ]

    # --- Dynamics defects -----------------------------------------------------

    def defects(self, states: FloatArray, torques: FloatArray) -> FloatArray:
        """Compute vectorized dynamics defects between adjacent nodes.

        Args:
            states: ``(N, 4)`` state trajectory.
            torques: ``(N, 2)`` torque trajectory.

        Returns:
            Flattened defect vector; all zeros means the trajectory obeys the
            equations of motion.
        """
        step = self._config.time_step
        left_states, right_states = states[:-1], states[1:]
        left_torques, right_torques = torques[:-1], torques[1:]

        left_rates = self._state_derivatives(left_states, left_torques)
        right_rates = self._state_derivatives(right_states, right_torques)

        if self._config.collocation_method == "trapezoidal":
            defect = right_states - left_states - 0.5 * step * (left_rates + right_rates)
            return np.asarray(defect.ravel(), dtype=np.float64)

        midpoint_states = 0.5 * (left_states + right_states) + (
            step / _HERMITE_MIDPOINT_WEIGHT
        ) * (left_rates - right_rates)
        midpoint_torques = 0.5 * (left_torques + right_torques)
        midpoint_rates = self._state_derivatives(midpoint_states, midpoint_torques)
        defect = (
            right_states
            - left_states
            - (step / _SIMPSON_WEIGHT) * (left_rates + 4.0 * midpoint_rates + right_rates)
        )
        return np.asarray(defect.ravel(), dtype=np.float64)

    def _state_derivatives(self, states: FloatArray, torques: FloatArray) -> FloatArray:
        """Return ``dx/dt`` for a block of states."""
        accelerations = generalized_accelerations(states, torques, self._config.params)
        return np.hstack([states[:, 2:_STATE_WIDTH], accelerations])

    # --- Constraints ----------------------------------------------------------

    def boundary_residuals(self, states: FloatArray) -> FloatArray:
        """Residuals for the start posture and the impact condition."""
        config = self._config
        start_residual = (states[0] - config.initial_state) / config.state_scale
        terminal = [states[-1, 0] - config.impact_theta1_rad]
        if config.require_release:
            terminal.append(states[-1, 1])
        return np.concatenate([start_residual, np.asarray(terminal, dtype=np.float64)])

    def slew_margins(self, torques: FloatArray) -> FloatArray:
        """Remaining torque slew-rate margin between adjacent nodes.

        Positive entries mean the limit is satisfied. Without this constraint the
        optimizer reverses full hub torque between two adjacent nodes to stop the
        arms dead at impact — optimal on paper, impossible for a golfer.
        """
        allowed_step = self._config.torque_rate_vector * self._config.time_step
        margins = allowed_step - np.abs(np.diff(torques, axis=0))
        return np.asarray(margins.ravel(), dtype=np.float64)

    @property
    def slew_normalizer(self) -> float:
        """Scale that brings slew margins to order one."""
        return float(np.mean(self._config.torque_rate_vector * self._config.time_step))

    @property
    def defect_normalizer(self) -> float:
        """Scale that brings dynamics defects to order one."""
        if not self._config.use_variable_scaling:
            return 1.0
        return float(self._config.time_step * RATE_SCALE)

    # --- Initial guess --------------------------------------------------------

    def initial_guess(self) -> tuple[FloatArray, FloatArray]:
        """Build a physically sensible starting trajectory.

        The guess sweeps the arms from the top to the ball on an accelerating
        profile while releasing the wrists over the back half of the downswing,
        under a constant driving hub torque. Starting from a plausible swing
        matters: the NLP is non-convex, and a lazy guess converges to lazy swings.
        """
        config = self._config
        progress = np.linspace(0.0, 1.0, self._node_count)
        step = config.time_step

        start_arm = float(config.initial_state[0])
        start_wrist = float(config.initial_state[1])
        arm_angle = start_arm + (config.impact_theta1_rad - start_arm) * progress**2
        release = np.clip((progress - _RELEASE_GUESS_ONSET), 0.0, None)
        release /= max(release.max(), 1.0e-12)
        wrist_angle = start_wrist * (1.0 - release**2)

        arm_rate = np.gradient(arm_angle, step)
        wrist_rate = np.gradient(wrist_angle, step)
        arm_rate[0] = float(config.initial_state[2])
        wrist_rate[0] = float(config.initial_state[3])

        states = np.column_stack([arm_angle, wrist_angle, arm_rate, wrist_rate])
        driving_torque = -0.6 * config.torque_clamp.max_torque1
        torques = np.column_stack(
            [np.full(self._node_count, driving_torque), np.zeros(self._node_count)]
        )
        return states, torques
