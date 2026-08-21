"""Registered execution provider for rotating-base transfer diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._numeric import FloatArray
from .contract import (
    EXPECTED_UPSTREAM_SOURCE_REVISION,
    MATCHING_RULES,
    MODEL_TIER,
    TORSO_PROFILES,
    RotatingBaseCase,
    RotatingBaseCaseMetrics,
)
from .integration import initial_state, rollout
from .types import (
    RotatingBaseConfig,
    RotatingBaseParams,
    RotatingBaseState,
    RotatingBaseTrace,
    TorsoTwoHandControl,
)

REGISTERED_TORSO_RATES_RAD_S = (1.5, 3.5, 5.5)
REGISTERED_TORSO_PROFILES_NM = {
    "accelerate": 55.0,
    "constant_rate": 0.0,
    "decelerate": -55.0,
}
REGISTERED_PEAK_GRIP_FORCE_CEILING_N = 100.0
REGISTERED_DURATION_S = 0.12
REGISTERED_STEP_S = 0.0005
REGISTERED_WRIST_RELEASE_S = 0.025


def _finite_array(name: str, value: object, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}")
    retained = array.copy()
    retained.setflags(write=False)
    return retained


@dataclass(frozen=True, slots=True)
class RotatingBaseRunRequest:
    """One exact member of the registered 18-case design."""

    torso_profile: str
    matching_rule: str
    initial_torso_rate_rad_s: float

    def __post_init__(self) -> None:
        if self.torso_profile not in TORSO_PROFILES:
            raise ValueError("torso_profile is outside the registered design")
        if self.matching_rule not in MATCHING_RULES:
            raise ValueError("matching_rule is outside the registered design")
        rate = self.initial_torso_rate_rad_s
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError(
                "initial_torso_rate_rad_s is outside the registered design"
            )
        if float(rate) not in REGISTERED_TORSO_RATES_RAD_S:
            raise ValueError(
                "initial_torso_rate_rad_s is outside the registered design"
            )
        object.__setattr__(self, "initial_torso_rate_rad_s", float(rate))

    @property
    def case_index(self) -> int:
        """Return the stable publication design index."""
        rule_index = MATCHING_RULES.index(self.matching_rule)
        profile_index = TORSO_PROFILES.index(self.torso_profile)
        rate_index = REGISTERED_TORSO_RATES_RAD_S.index(self.initial_torso_rate_rad_s)
        return rule_index * 9 + profile_index * 3 + rate_index


@dataclass(frozen=True, slots=True)
class RotatingBaseRunTrace:
    """Immutable UI-facing signals from one qualified execution."""

    time_s: FloatArray
    torso_rate_rad_s: FloatArray
    club_rate_rad_s: FloatArray
    clubhead_speed_m_s: FloatArray
    contact_power_on_club_w: FloatArray
    force_generated_couple_nm: FloatArray
    force_on_club_n: FloatArray
    distal_segment_kinetic_energy_j: FloatArray

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float).reshape(-1)
        if time.size < 2 or np.any(np.diff(time) <= 0.0):
            raise ValueError("time_s must be finite and strictly increasing")
        count = time.size
        object.__setattr__(self, "time_s", _finite_array("time_s", time, (count,)))
        for name in (
            "torso_rate_rad_s",
            "club_rate_rad_s",
            "clubhead_speed_m_s",
            "contact_power_on_club_w",
            "force_generated_couple_nm",
            "distal_segment_kinetic_energy_j",
        ):
            object.__setattr__(
                self, name, _finite_array(name, getattr(self, name), (count,))
            )
        object.__setattr__(
            self,
            "force_on_club_n",
            _finite_array("force_on_club_n", self.force_on_club_n, (count, 2, 2)),
        )


@dataclass(frozen=True, slots=True)
class RotatingBaseRunResult:
    """Source-pinned scalar and trace evidence for one registered request."""

    request: RotatingBaseRunRequest
    case: RotatingBaseCase
    trace: RotatingBaseRunTrace
    model_tier: str = MODEL_TIER
    source_revision: str = EXPECTED_UPSTREAM_SOURCE_REVISION


def _control_law(
    torso_nm: float,
) -> Callable[[float, RotatingBaseState], TorsoTwoHandControl]:
    def law(time_s: float, _state: RotatingBaseState) -> TorsoTwoHandControl:
        wrist_nm = -3.0 if time_s < REGISTERED_WRIST_RELEASE_S else 4.0
        return TorsoTwoHandControl(
            torso_nm=torso_nm,
            lead_arm_nm=7.0,
            trail_arm_nm=7.0,
            lead_wrist_nm=wrist_nm,
            trail_wrist_nm=wrist_nm,
        )

    return law


def _trapz(values: FloatArray, time: FloatArray) -> float:
    return float(np.trapezoid(values, time))


def _braking_and_impulse(trace: RotatingBaseTrace) -> tuple[float, float]:
    braking_work = -_trapz(np.minimum(trace.contact_power_on_club_w, 0.0), trace.time)
    resultant = np.sum(trace.force_on_club_n, axis=1)
    speed = np.maximum(trace.clubhead_speed_m_s, 1e-12)
    along_path = np.sum(
        resultant * trace.clubhead_velocity_m_s / speed[:, None], axis=1
    )
    impulse = -_trapz(np.minimum(along_path, 0.0), trace.time)
    return braking_work, impulse


def _bilateral_wrist_work(trace: RotatingBaseTrace) -> float:
    controls = np.asarray([control.as_array() for control in trace.controls])
    lead_rate = trace.qdot[:, 5] - (trace.qdot[:, 0] + trace.qdot[:, 1])
    trail_rate = trace.qdot[:, 5] - (trace.qdot[:, 0] + trace.qdot[:, 2])
    power = controls[:, 3] * lead_rate + controls[:, 4] * trail_rate
    return _trapz(power, trace.time)


def _exclusion_reasons(
    trace: RotatingBaseTrace, peak_force_n: float
) -> tuple[str, ...]:
    reasons: list[str] = []
    if peak_force_n > REGISTERED_PEAK_GRIP_FORCE_CEILING_N:
        reasons.append("registered_peak_grip_force_ceiling_exceeded")
    if np.max(trace.position_constraint_norm_m) >= 1e-7:
        reasons.append("position_constraint_closure_failed")
    if abs(trace.work_energy_closure_j) >= 0.08:
        reasons.append("work_energy_closure_failed")
    return tuple(reasons)


def _case_metrics(
    trace: RotatingBaseTrace, initial_club_rate: float
) -> RotatingBaseCaseMetrics:
    grip_magnitudes = np.linalg.norm(trace.force_on_club_n, axis=2)
    peak_force = float(np.max(grip_magnitudes))
    braking_work, negative_impulse = _braking_and_impulse(trace)
    force_couple_work = _trapz(
        trace.force_generated_couple_nm * trace.qdot[:, 5], trace.time
    )
    return RotatingBaseCaseMetrics(
        initial_club_rate_rad_s=initial_club_rate,
        final_torso_rate_rad_s=float(trace.qdot[-1, 0]),
        impact_speed_m_s=float(trace.clubhead_speed_m_s[-1]),
        clubhead_speed_gain_m_s=float(
            trace.clubhead_speed_m_s[-1] - trace.clubhead_speed_m_s[0]
        ),
        contact_work_on_club_j=_trapz(trace.contact_power_on_club_w, trace.time),
        braking_grip_work_j=braking_work,
        force_couple_work_j=force_couple_work,
        negative_along_path_impulse_ns=negative_impulse,
        bilateral_wrist_work_j=_bilateral_wrist_work(trace),
        total_control_work_j=_trapz(trace.control_power_w, trace.time),
        distal_energy_gain_j=float(
            trace.distal_segment_kinetic_energy_j[-1]
            - trace.distal_segment_kinetic_energy_j[0]
        ),
        peak_grip_force_n=peak_force,
        maximum_constraint_residual_m=float(np.max(trace.position_constraint_norm_m)),
        maximum_velocity_constraint_residual_m_s=float(
            np.max(trace.velocity_constraint_norm_m_s)
        ),
        maximum_contact_power_identity_residual_w=float(
            np.max(np.abs(trace.contact_power_identity_residual_w))
        ),
        work_energy_closure_j=trace.work_energy_closure_j,
    )


def _ui_trace(trace: RotatingBaseTrace) -> RotatingBaseRunTrace:
    return RotatingBaseRunTrace(
        time_s=trace.time,
        torso_rate_rad_s=trace.qdot[:, 0],
        club_rate_rad_s=trace.qdot[:, 5],
        clubhead_speed_m_s=trace.clubhead_speed_m_s,
        contact_power_on_club_w=trace.contact_power_on_club_w,
        force_generated_couple_nm=trace.force_generated_couple_nm,
        force_on_club_n=trace.force_on_club_n,
        distal_segment_kinetic_energy_j=trace.distal_segment_kinetic_energy_j,
    )


def run_registered_case(request: RotatingBaseRunRequest) -> RotatingBaseRunResult:
    """Execute one full-resolution registered rotating-base case."""
    if not isinstance(request, RotatingBaseRunRequest):
        raise TypeError("request must be a RotatingBaseRunRequest")
    params = RotatingBaseParams.publication_default()
    torso_rate = request.initial_torso_rate_rad_s
    club_rate = torso_rate + 1.0 if request.matching_rule == MATCHING_RULES[0] else 3.0
    state = initial_state(
        params, torso_rate_rad_s=torso_rate, club_rate_rad_s=club_rate
    )
    trace = rollout(
        state,
        _control_law(REGISTERED_TORSO_PROFILES_NM[request.torso_profile]),
        params,
        RotatingBaseConfig(duration_s=REGISTERED_DURATION_S, step_s=REGISTERED_STEP_S),
    )
    metrics = _case_metrics(trace, club_rate)
    reasons = _exclusion_reasons(trace, metrics.peak_grip_force_n)
    case = RotatingBaseCase(
        case_index=request.case_index,
        torso_profile=request.torso_profile,
        matching_rule=request.matching_rule,
        initial_torso_rate_rad_s=torso_rate,
        metrics=metrics,
        valid=not reasons,
        exclusion_reasons=reasons,
    )
    return RotatingBaseRunResult(request=request, case=case, trace=_ui_trace(trace))


__all__ = [
    "REGISTERED_TORSO_RATES_RAD_S",
    "RotatingBaseRunRequest",
    "RotatingBaseRunResult",
    "RotatingBaseRunTrace",
    "run_registered_case",
]
