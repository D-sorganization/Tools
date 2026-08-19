"""Counterfactual clubfitting engine (club-tester C4, #4553).

The comparator OEM fitting workflows run: hold the swing input fixed,
change one club at a time, and report what the *ball* does differently.
Each evaluation runs the full shipped pipeline —

    fitting document (C3) → shaft delivery deltas (C2)
      → delivery parameters → impact solve → flight simulation

— so every number in a report is produced by the same physics the
simulator GUIs display, never by a side model.

**Held-fixed semantics.** The grip kinematics are *declared inputs*: a
heavier head is evaluated at the same grip motion, so its ball-speed gain
from the mass ratio is reported without the (golfer-dependent) swing-speed
cost of the extra mass. That is the standard fitting-bay convention;
coupling head mass back into swing speed is a biomechanics question that
belongs to the C5 trajectory sources, not to this comparator.

Counterfactual bounds are validated hard (a fitting sweep must not wander
into regimes where the C2 quasi-static model refuses or the impact model
is unphysical), evaluation is deterministic, and the report serializes to
deterministic sorted-keys JSON so two identical runs are byte-identical.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.python.swing_sim.flight.frames import to_flight_frame
from shared.python.swing_sim.flight.launch import derive_launch_conditions
from shared.python.swing_sim.flight.pipeline import simulate
from shared.python.swing_sim.impact import (
    DeliveryParameters,
    ImpactModelType,
    ImpactSolverAPI,
    derive_delivery,
    to_pre_impact_state,
)

from ._validation import require_finite_float, require_identifier
from .fitting_document import ClubFittingDocument
from .shaft_delivery import (
    GripKinematics,
    ShaftDeliveryDeltas,
    ShaftTipMass,
    solve_shaft_delivery,
)
from .shaft_scaling import ShaftProfileScaling, scale_shaft_profile

FITTING_REPORT_FORMAT = "golf_club.fitting_report/1"

__all__ = [
    "FITTING_REPORT_FORMAT",
    "ClubOutcome",
    "CounterfactualSpec",
    "FittingReport",
    "compare_counterfactuals",
    "evaluate_club",
    "fitting_report_to_json",
]


@dataclass(frozen=True)
class CounterfactualSpec:
    """One bounded what-if applied to the baseline club.

    Bounds keep every counterfactual inside the validity envelopes of the
    downstream models; out-of-band requests are refused, not clamped.
    """

    label: str
    head_mass_scale: float = 1.0
    cg_back_delta_m: float = 0.0
    cg_toe_delta_m: float = 0.0
    loft_delta_deg: float = 0.0
    ei_scale: float = 1.0
    gj_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", require_identifier(self.label, "label"))
        for name in (
            "head_mass_scale",
            "cg_back_delta_m",
            "cg_toe_delta_m",
            "loft_delta_deg",
            "ei_scale",
            "gj_scale",
        ):
            object.__setattr__(
                self, name, require_finite_float(getattr(self, name), name)
            )
        if not 0.5 <= self.head_mass_scale <= 1.5:
            raise ValueError("head_mass_scale must lie in [0.5, 1.5]")
        if abs(self.cg_back_delta_m) > 0.02 or abs(self.cg_toe_delta_m) > 0.02:
            raise ValueError("cg deltas must lie within +/- 0.02 m")
        if abs(self.loft_delta_deg) > 4.0:
            raise ValueError("loft_delta_deg must lie within +/- 4 degrees")
        if not 0.5 <= self.ei_scale <= 2.0 or not 0.5 <= self.gj_scale <= 2.0:
            raise ValueError("stiffness scales must lie in [0.5, 2.0]")


@dataclass(frozen=True)
class ClubOutcome:
    """What one club delivered and what the ball did."""

    label: str
    delivered_loft_deg: float
    face_angle_deg: float
    lie_toe_down_deg: float
    clubhead_speed_mps: float
    ball_speed_mps: float
    launch_angle_deg: float
    backspin_rpm: float
    carry_m: float
    max_height_m: float
    flight_time_s: float
    lateral_m: float
    shaft: ShaftDeliveryDeltas


@dataclass(frozen=True)
class FittingReport:
    """Baseline plus per-counterfactual outcomes, all identity-carrying."""

    document_id: str
    grip: GripKinematics
    baseline: ClubOutcome
    counterfactuals: tuple[ClubOutcome, ...]


def _apply_counterfactual(
    document: ClubFittingDocument, spec: CounterfactualSpec
) -> tuple[Any, ShaftTipMass, float]:
    """Derived (profile, tip mass, loft) for one counterfactual."""
    profile = document.shaft_profile
    if spec.ei_scale != 1.0 or spec.gj_scale != 1.0:
        profile = scale_shaft_profile(
            profile,
            ShaftProfileScaling(
                ei_about_x_scale=spec.ei_scale,
                ei_about_y_scale=spec.ei_scale,
                gj_scale=spec.gj_scale,
            ),
            shaft_id=f"{profile.shaft_id}--{spec.label}",
        )
    base = document.tip_mass
    tip = ShaftTipMass(
        mass_kg=base.mass_kg * spec.head_mass_scale,
        cg_back_m=base.cg_back_m + spec.cg_back_delta_m,
        cg_toe_m=base.cg_toe_m + spec.cg_toe_delta_m,
        cg_drop_m=base.cg_drop_m,
    )
    return profile, tip, document.face.loft_deg + spec.loft_delta_deg


def evaluate_club(
    document: ClubFittingDocument,
    grip: GripKinematics,
    *,
    counterfactual: CounterfactualSpec | None = None,
    impact_offset_toe_mm: float = 0.0,
    impact_offset_high_mm: float = 0.0,
    flight_model: str = "waterloo_penner",
) -> ClubOutcome:
    """Run the full delivery→impact→flight pipeline for one club variant."""
    if not isinstance(document, ClubFittingDocument):
        raise TypeError("document must be ClubFittingDocument")
    if not isinstance(grip, GripKinematics):
        raise TypeError("grip must be GripKinematics")
    if counterfactual is not None and not isinstance(
        counterfactual, CounterfactualSpec
    ):
        raise TypeError("counterfactual must be CounterfactualSpec or None")
    for name, value in (
        ("impact_offset_toe_mm", impact_offset_toe_mm),
        ("impact_offset_high_mm", impact_offset_high_mm),
    ):
        require_finite_float(value, name)

    spec = counterfactual or CounterfactualSpec(label="baseline")
    profile, tip, static_loft_deg = _apply_counterfactual(document, spec)

    shaft = solve_shaft_delivery(profile, tip, grip)
    clubhead_speed = grip.omega_rad_s * grip.swing_radius_m + shaft.kick_speed_mps
    delivered_loft = static_loft_deg + shaft.dynamic_loft_add_deg
    face_angle = -shaft.face_closure_deg  # + = open in delivery convention

    params = DeliveryParameters(
        clubhead_speed_mps=clubhead_speed,
        club_path_deg=0.0,
        face_angle_deg=face_angle,
        attack_angle_deg=0.0,
        dynamic_loft_deg=delivered_loft,
        lie_deg=shaft.lie_toe_down_deg,
        impact_offset_toe_mm=impact_offset_toe_mm,
        impact_offset_high_mm=impact_offset_high_mm,
    )
    derived = derive_delivery(params)
    pre = to_pre_impact_state(params, clubhead_mass=tip.mass_kg)
    solver = ImpactSolverAPI(ImpactModelType.RIGID_BODY)
    post = solver.solve_impact(
        timestamp=0.0,
        clubhead_velocity=derived.clubhead_velocity,
        clubhead_orientation=derived.face_normal,
        clubhead_mass=tip.mass_kg,
        impact_offset=pre.impact_offset,
        record=False,
    )
    launch = derive_launch_conditions(
        to_flight_frame(post.ball_velocity),
        to_flight_frame(post.ball_angular_velocity),
    )
    flight = simulate(launch, model_name=flight_model)
    return ClubOutcome(
        label=spec.label,
        delivered_loft_deg=delivered_loft,
        face_angle_deg=face_angle,
        lie_toe_down_deg=shaft.lie_toe_down_deg,
        clubhead_speed_mps=clubhead_speed,
        ball_speed_mps=float(np.linalg.norm(post.ball_velocity)),
        launch_angle_deg=math.degrees(launch.launch_angle),
        backspin_rpm=launch.spin_rate,
        carry_m=flight.carry_distance,
        max_height_m=flight.max_height,
        flight_time_s=flight.flight_time,
        lateral_m=flight.lateral_deviation,
        shaft=shaft,
    )


def compare_counterfactuals(
    document: ClubFittingDocument,
    grip: GripKinematics,
    counterfactuals: tuple[CounterfactualSpec, ...],
    *,
    impact_offset_toe_mm: float = 0.0,
    impact_offset_high_mm: float = 0.0,
    flight_model: str = "waterloo_penner",
) -> FittingReport:
    """Evaluate the baseline and every counterfactual under one swing input."""
    if not isinstance(counterfactuals, tuple) or not all(
        isinstance(item, CounterfactualSpec) for item in counterfactuals
    ):
        raise TypeError("counterfactuals must be a tuple of CounterfactualSpec")
    labels = [item.label for item in counterfactuals]
    if len(set(labels)) != len(labels) or "baseline" in labels:
        raise ValueError("counterfactual labels must be unique and not 'baseline'")
    evaluate = lambda spec: evaluate_club(  # noqa: E731 - local binding
        document,
        grip,
        counterfactual=spec,
        impact_offset_toe_mm=impact_offset_toe_mm,
        impact_offset_high_mm=impact_offset_high_mm,
        flight_model=flight_model,
    )
    return FittingReport(
        document_id=document.document_id,
        grip=grip,
        baseline=evaluate(None),
        counterfactuals=tuple(evaluate(spec) for spec in counterfactuals),
    )


def _outcome_payload(outcome: ClubOutcome, baseline: ClubOutcome | None) -> dict:
    payload: dict[str, Any] = {
        "label": outcome.label,
        "delivered_loft_deg": outcome.delivered_loft_deg,
        "face_angle_deg": outcome.face_angle_deg,
        "lie_toe_down_deg": outcome.lie_toe_down_deg,
        "clubhead_speed_mps": outcome.clubhead_speed_mps,
        "ball_speed_mps": outcome.ball_speed_mps,
        "launch_angle_deg": outcome.launch_angle_deg,
        "backspin_rpm": outcome.backspin_rpm,
        "carry_m": outcome.carry_m,
        "max_height_m": outcome.max_height_m,
        "flight_time_s": outcome.flight_time_s,
        "lateral_m": outcome.lateral_m,
        "shaft": {
            "dynamic_loft_add_deg": outcome.shaft.dynamic_loft_add_deg,
            "face_closure_deg": outcome.shaft.face_closure_deg,
            "lie_toe_down_deg": outcome.shaft.lie_toe_down_deg,
            "kick_speed_mps": outcome.shaft.kick_speed_mps,
            "first_mode_hz": outcome.shaft.first_mode_hz,
            "model_name": outcome.shaft.model_name,
        },
    }
    if baseline is not None:
        payload["deltas_vs_baseline"] = {
            "carry_m": outcome.carry_m - baseline.carry_m,
            "ball_speed_mps": outcome.ball_speed_mps - baseline.ball_speed_mps,
            "launch_angle_deg": outcome.launch_angle_deg - baseline.launch_angle_deg,
            "backspin_rpm": outcome.backspin_rpm - baseline.backspin_rpm,
            "lateral_m": outcome.lateral_m - baseline.lateral_m,
        }
    return payload


def fitting_report_to_json(report: FittingReport) -> str:
    """Serialize deterministically; identical runs are byte-identical."""
    if not isinstance(report, FittingReport):
        raise TypeError("report must be FittingReport")
    payload = {
        "format": FITTING_REPORT_FORMAT,
        "document_id": report.document_id,
        "grip": {
            "omega_rad_s": report.grip.omega_rad_s,
            "alpha_rad_s2": report.grip.alpha_rad_s2,
            "swing_radius_m": report.grip.swing_radius_m,
            "downswing_duration_s": report.grip.downswing_duration_s,
            "release_recovery": report.grip.release_recovery,
        },
        "baseline": _outcome_payload(report.baseline, None),
        "counterfactuals": [
            _outcome_payload(outcome, report.baseline)
            for outcome in report.counterfactuals
        ],
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
