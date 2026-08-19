"""Impact recorder and engine-agnostic solver API.

Ported from UpstreamDrift ``src/shared/python/physics/impact_model/solver.py``
(epic #4103 / issue #4106), rewritten self-contained.

Changes from the UpstreamDrift source:

- BUG FIX (recon #4104, defect a): ``solve_with_gear_effect`` previously
  computed the base impulse WITHOUT the impact offset — the
  ``PreImpactState`` it built silently dropped ``impact_offset``, so
  off-center hits skipped the MOI effective-mass reduction and launched
  at full center-strike ball speed. The offset is now carried into the
  pre-impact state (and the recorded event), so off-center ball speed is
  correctly lower than a center strike.
- Gear effect is now the physics-derived head-recoil model from
  :mod:`.gear_effect` (the three empirical scaling constants are gone),
  with an optional ``face_normal_at_offset`` bulge/roll callable supplied
  by the app's club package.
- ``solve_impact`` accepts optional ``impact_offset`` /
  ``clubhead_moi`` / ``clubhead_moi_tensor`` passthroughs.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from shared.python.contracts import precondition

from .constants import DRIVER_MASS_KG, DRIVER_MOI_KG_M2, GOLF_BALL_MASS_KG
from .gear_effect import FaceNormalAtOffset, compute_gear_effect, resolve_contact_normal
from .models import _norm, create_impact_model
from .types import (
    ImpactEvent,
    ImpactModelType,
    ImpactParameters,
    PostImpactState,
    PreImpactState,
)
from .utils import validate_energy_balance

logger = logging.getLogger(__name__)


class ImpactRecorder:
    """Records impact events during simulation.

    Surfaces pre-impact and post-impact states in recorder outputs and
    provides energy-balance checks for each impact.
    """

    def __init__(self) -> None:
        """Initialize impact recorder."""
        self.events: list[ImpactEvent] = []
        self._impact_counter = 0

    def record_impact(
        self,
        timestamp: float,
        pre_state: PreImpactState,
        post_state: PostImpactState,
        params: ImpactParameters,
        model_type: ImpactModelType = ImpactModelType.RIGID_BODY,
    ) -> ImpactEvent:
        """Record an impact event.

        Args:
            timestamp: Simulation time [s]
            pre_state: Pre-impact state
            post_state: Post-impact state
            params: Impact parameters used
            model_type: Type of impact model used

        Returns:
            Recorded ImpactEvent
        """
        if timestamp is None:
            raise ValueError("timestamp must be provided")
        energy_balance = validate_energy_balance(pre_state, post_state, params)

        event = ImpactEvent(
            timestamp=timestamp,
            pre_state=pre_state,
            post_state=post_state,
            energy_balance=energy_balance,
            impact_id=self._impact_counter,
            model_type=model_type,
        )
        self.events.append(event)
        self._impact_counter += 1

        logger.info(
            "Impact #%d recorded at t=%.4fs, ball speed: %.1f m/s, energy loss: %.1f%%",
            event.impact_id,
            timestamp,
            energy_balance["ball_launch_speed"],
            100.0 * energy_balance["energy_loss_ratio"],
        )
        return event

    def get_all_events(self) -> list[ImpactEvent]:
        """Get all recorded impact events."""
        return self.events.copy()

    def export_to_dict(self) -> dict:
        """Export all events as dictionary for JSON serialization."""
        events_data = [
            {
                "impact_id": event.impact_id,
                "timestamp": event.timestamp,
                "model_type": event.model_type.name,
                "pre_impact": {
                    "clubhead_velocity": event.pre_state.clubhead_velocity.tolist(),
                    "ball_velocity": event.pre_state.ball_velocity.tolist(),
                    "ball_spin": event.pre_state.ball_angular_velocity.tolist(),
                },
                "post_impact": {
                    "ball_velocity": event.post_state.ball_velocity.tolist(),
                    "ball_spin": event.post_state.ball_angular_velocity.tolist(),
                    "clubhead_velocity": event.post_state.clubhead_velocity.tolist(),
                    "contact_duration": event.post_state.contact_duration,
                    "energy_transfer": event.post_state.energy_transfer,
                },
                "energy_balance": event.energy_balance,
            }
            for event in self.events
        ]
        return {
            "num_impacts": len(self.events),
            "events": events_data,
            "summary": self.get_summary(),
        }

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics for all impacts."""
        if not self.events:
            return {"num_impacts": 0}
        speeds = [e.energy_balance["ball_launch_speed"] for e in self.events]
        losses = [e.energy_balance["energy_loss_ratio"] for e in self.events]
        return {
            "num_impacts": len(self.events),
            "mean_ball_speed": float(np.mean(speeds)),
            "max_ball_speed": float(np.max(speeds)),
            "mean_energy_loss_ratio": float(np.mean(losses)),
        }

    def reset(self) -> None:
        """Clear all recorded events."""
        self.events.clear()
        self._impact_counter = 0


class ImpactSolverAPI:
    """Engine-agnostic API for impact solving.

    Provides a unified interface for different physics engines, with
    optional recording and energy/COR/spin validation.
    """

    def __init__(
        self,
        model_type: ImpactModelType = ImpactModelType.RIGID_BODY,
        params: ImpactParameters | None = None,
    ) -> None:
        """Initialize impact solver.

        Args:
            model_type: Type of impact model to use
            params: Impact parameters (uses defaults if None)
        """
        if model_type is None:
            raise ValueError("model_type must be provided")
        self.model_type = model_type
        self.model = create_impact_model(model_type)
        self.params = params or ImpactParameters()
        self.recorder = ImpactRecorder()

    def _build_pre_state(
        self,
        clubhead_velocity: np.ndarray,
        clubhead_orientation: np.ndarray,
        ball_velocity: np.ndarray | None,
        ball_angular_velocity: np.ndarray | None,
        clubhead_mass: float,
        impact_offset: np.ndarray | None,
        clubhead_moi: float,
        clubhead_moi_tensor: np.ndarray | None,
    ) -> PreImpactState:
        """Assemble a PreImpactState, defaulting the ball to rest."""
        return PreImpactState(
            clubhead_velocity=np.asarray(clubhead_velocity, dtype=float),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.asarray(clubhead_orientation, dtype=float),
            ball_position=np.zeros(3),
            ball_velocity=(
                np.asarray(ball_velocity, dtype=float)
                if ball_velocity is not None
                else np.zeros(3)
            ),
            ball_angular_velocity=(
                np.asarray(ball_angular_velocity, dtype=float)
                if ball_angular_velocity is not None
                else np.zeros(3)
            ),
            clubhead_mass=clubhead_mass,
            clubhead_moi=clubhead_moi,
            impact_offset=(
                np.asarray(impact_offset, dtype=float)
                if impact_offset is not None
                else None
            ),
            clubhead_moi_tensor=clubhead_moi_tensor,
        )

    @precondition(
        lambda self, timestamp, *args, **kwargs: timestamp >= 0,
        "Timestamp must be non-negative",
    )
    def solve_impact(
        self,
        timestamp: float,
        clubhead_velocity: np.ndarray,
        clubhead_orientation: np.ndarray,
        ball_velocity: np.ndarray | None = None,
        ball_angular_velocity: np.ndarray | None = None,
        clubhead_mass: float = DRIVER_MASS_KG,
        impact_offset: np.ndarray | None = None,
        clubhead_moi: float = DRIVER_MOI_KG_M2,
        clubhead_moi_tensor: np.ndarray | None = None,
        record: bool = True,
    ) -> PostImpactState:
        """Solve impact and optionally record the event.

        Args:
            timestamp: Current simulation time [s]
            clubhead_velocity: Clubhead velocity [m/s] (3,)
            clubhead_orientation: Clubface normal [unitless] (3,)
            ball_velocity: Ball velocity (default: stationary) [m/s] (3,)
            ball_angular_velocity: Ball spin (default: zero) [rad/s] (3,)
            clubhead_mass: Clubhead mass [kg] (must be positive)
            impact_offset: Optional face offset from CG [m] (2,)
                [horizontal, vertical] — enables the MOI effective-mass
                reduction for off-center strikes
            clubhead_moi: Scalar clubhead MOI about CG [kg.m^2]
            clubhead_moi_tensor: Optional 3x3 MOI tensor (full 3-D
                effective-mass treatment)
            record: Whether to record this impact

        Returns:
            Post-impact state
        """
        if timestamp is None:
            raise ValueError("timestamp must be provided")
        if clubhead_mass <= 0:
            raise ValueError("Clubhead mass must be positive")
        pre_state = self._build_pre_state(
            clubhead_velocity,
            clubhead_orientation,
            ball_velocity,
            ball_angular_velocity,
            clubhead_mass,
            impact_offset,
            clubhead_moi,
            clubhead_moi_tensor,
        )
        post_state = self.model.solve(pre_state, self.params)
        if record:
            self.recorder.record_impact(
                timestamp, pre_state, post_state, self.params, self.model_type
            )
        return post_state

    def solve_with_gear_effect(
        self,
        timestamp: float,
        clubhead_velocity: np.ndarray,
        clubhead_orientation: np.ndarray,
        impact_offset: np.ndarray,
        ball_velocity: np.ndarray | None = None,
        clubhead_mass: float = DRIVER_MASS_KG,
        clubhead_moi: float = DRIVER_MOI_KG_M2,
        clubhead_moi_tensor: np.ndarray | None = None,
        face_normal_at_offset: FaceNormalAtOffset | None = None,
        record: bool = True,
    ) -> PostImpactState:
        """Solve an off-center impact with physics-based gear-effect spin.

        The impact offset is carried into the base solve (bug fix, see
        module docstring), the local bulge/roll normal (if a curvature
        callable is given) sets the impulse direction, and the head's
        rotation recoil adds gear-effect spin via :mod:`.gear_effect`.

        Args:
            timestamp: Current simulation time [s]
            clubhead_velocity: Clubhead velocity [m/s] (3,)
            clubhead_orientation: Nominal clubface normal (3,)
            impact_offset: Offset from face center [m] (2,)
                [horizontal (+ toe), vertical (+ high)]
            ball_velocity: Ball velocity [m/s] (3,)
            clubhead_mass: Clubhead mass [kg]
            clubhead_moi: Scalar clubhead MOI about CG [kg.m^2]
            clubhead_moi_tensor: Optional 3x3 MOI tensor
            face_normal_at_offset: Optional bulge/roll callable
                ``(toe_m, high_m) -> local normal`` from the club package
            record: Whether to record this impact

        Returns:
            Post-impact state with gear-effect spin and head recoil.
        """
        if timestamp is None:
            raise ValueError("timestamp must be provided")
        offset = np.asarray(impact_offset, dtype=float)
        contact_normal = resolve_contact_normal(
            offset, clubhead_orientation, face_normal_at_offset
        )

        # Base solve WITH the offset so the MOI effective-mass reduction
        # applies (previously the offset was silently dropped here).
        pre_state = self._build_pre_state(
            clubhead_velocity,
            contact_normal,
            ball_velocity,
            None,
            clubhead_mass,
            offset,
            clubhead_moi,
            clubhead_moi_tensor,
        )
        post_state = self.model.solve(pre_state, self.params)

        # Recover the normal impulse from the ball's velocity change.
        delta_v = post_state.ball_velocity - pre_state.ball_velocity
        normal_impulse = max(
            0.0, GOLF_BALL_MASS_KG * float(np.dot(delta_v, contact_normal))
        )

        gear = compute_gear_effect(
            impact_offset=offset,
            face_normal=np.asarray(clubhead_orientation, dtype=float),
            normal_impulse=normal_impulse,
            clubhead_moi=(
                clubhead_moi_tensor if clubhead_moi_tensor is not None else clubhead_moi
            ),
            cg_depth_m=self.params.cg_depth,
            friction_coefficient=self.params.friction_coefficient,
            face_normal_at_offset=face_normal_at_offset,
        )

        modified_post = PostImpactState(
            ball_velocity=post_state.ball_velocity,
            ball_angular_velocity=(
                post_state.ball_angular_velocity + gear.ball_spin_delta
            ),
            clubhead_velocity=post_state.clubhead_velocity,
            clubhead_angular_velocity=(
                post_state.clubhead_angular_velocity + gear.head_angular_velocity_delta
            ),
            contact_duration=post_state.contact_duration,
            energy_transfer=post_state.energy_transfer,
            impact_location=offset,
        )
        if record:
            self.recorder.record_impact(
                timestamp, pre_state, modified_post, self.params, self.model_type
            )
        return modified_post

    def get_energy_report(self) -> dict:
        """Get energy balance report for all recorded impacts."""
        if not self.recorder.events:
            raise RuntimeError("No impacts recorded")
        reports = [
            {
                "impact_id": event.impact_id,
                "timestamp": event.timestamp,
                "ke_pre": event.energy_balance["total_ke_pre"],
                "ke_post": event.energy_balance["total_ke_post"],
                "energy_lost": event.energy_balance["energy_lost"],
                "loss_ratio": event.energy_balance["energy_loss_ratio"],
                "ball_speed": event.energy_balance["ball_launch_speed"],
            }
            for event in self.recorder.events
        ]
        total_ke_pre = sum(r["ke_pre"] for r in reports)
        total_ke_post = sum(r["ke_post"] for r in reports)
        return {
            "impacts": reports,
            "total_ke_pre": total_ke_pre,
            "total_ke_post": total_ke_post,
            "total_energy_lost": total_ke_pre - total_ke_post,
            "overall_loss_ratio": (
                (total_ke_pre - total_ke_post) / total_ke_pre if total_ke_pre > 0 else 0
            ),
        }

    def validate_cor_behavior(
        self, tolerance: float = 0.05
    ) -> dict[str, bool | float | str | int]:
        """Validate COR behavior across recorded impacts.

        Args:
            tolerance: Acceptable deviation from expected COR

        Returns:
            Validation result with pass/fail and details
        """
        if tolerance is None:
            raise ValueError("tolerance must be provided")
        if not self.recorder.events:
            raise RuntimeError("No impacts recorded")

        expected_cor = self.params.cor
        measured_cors = []
        for event in self.recorder.events:
            v_club_pre = _norm(event.pre_state.clubhead_velocity)
            v_ball_pre = _norm(event.pre_state.ball_velocity)
            v_club_post = _norm(event.post_state.clubhead_velocity)
            v_ball_post = _norm(event.post_state.ball_velocity)
            # COR = (v_ball_post - v_club_post) / (v_club_pre - v_ball_pre)
            approach = v_club_pre - v_ball_pre
            if approach > 0.1:  # Avoid division by small number
                measured_cors.append((v_ball_post - v_club_post) / approach)

        if not measured_cors:
            raise RuntimeError("Could not compute COR")

        mean_cor = float(np.mean(measured_cors))
        deviation = abs(mean_cor - expected_cor)
        return {
            "valid": deviation <= tolerance,
            "expected_cor": expected_cor,
            "measured_cor_mean": mean_cor,
            "deviation": deviation,
            "tolerance": tolerance,
            "num_samples": len(measured_cors),
        }

    def validate_spin_behavior(
        self, max_spin_rpm: float = 10000
    ) -> dict[str, bool | float | str | int]:
        """Validate spin behavior is within physical limits.

        Args:
            max_spin_rpm: Maximum acceptable spin rate [RPM]

        Returns:
            Validation result with pass/fail and details
        """
        if max_spin_rpm is None:
            raise ValueError("max_spin_rpm must be provided")
        if not self.recorder.events:
            raise RuntimeError("No impacts recorded")

        max_spin_rad = max_spin_rpm * 2 * math.pi / 60  # Convert to rad/s
        spins = [
            _norm(event.post_state.ball_angular_velocity)
            for event in self.recorder.events
        ]
        max_observed = float(np.max(spins))
        return {
            "valid": max_observed <= max_spin_rad,
            "max_observed_rpm": max_observed * 60 / (2 * math.pi),
            "max_allowed_rpm": max_spin_rpm,
            "num_samples": len(spins),
        }

    def reset(self) -> None:
        """Reset solver state and clear recorded impacts."""
        self.recorder.reset()
