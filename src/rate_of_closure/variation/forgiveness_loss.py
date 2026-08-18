"""Transparent objective and constraints for chip-forgiveness decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .chip_forgiveness import ChipTrialCohort


def objective_id_for_target_carry(target_carry_m: float) -> str:
    """Return a stable objective identity without millimetre-rounding aliases."""
    target = float(target_carry_m)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("target_carry_m must be finite and >= 0")
    target_text = f"{target:.9f}".rstrip("0").rstrip(".")
    return f"chip-target-{target_text}m-balanced-v1"


@dataclass(frozen=True)
class ChipLossModel:
    """Normalized continuous loss plus explicit contact/failure penalties."""

    objective_id: str = "auto"
    target_carry_m: float = 27.432
    carry_tolerance_m: float = 2.0
    lateral_tolerance_m: float = 1.0
    maximum_turf_penetration_m: float = 0.05
    include_turf_penetration: bool = False
    missing_required_metric_penalty: float = 12.0
    unsupported_turf_penalty: float = 12.0
    ground_first_penalty: float = 4.0
    simultaneous_penalty: float = 2.0
    ground_only_miss_penalty: float = 12.0
    no_contact_miss_penalty: float = 12.0
    numerical_failure_penalty: float = 16.0

    def __post_init__(self) -> None:
        """Require a finite, nonnegative, scale-explicit objective."""
        if not isinstance(self.objective_id, str) or not self.objective_id.strip():
            raise ValueError("objective_id must be a nonempty string")
        if self.objective_id == "auto":
            object.__setattr__(
                self,
                "objective_id",
                objective_id_for_target_carry(self.target_carry_m),
            )
        if not isinstance(self.include_turf_penetration, bool):
            raise TypeError("include_turf_penetration must be a boolean")
        positive = (
            "carry_tolerance_m",
            "lateral_tolerance_m",
            "maximum_turf_penetration_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        finite = (
            "target_carry_m",
            "ground_first_penalty",
            "simultaneous_penalty",
            "ground_only_miss_penalty",
            "no_contact_miss_penalty",
            "numerical_failure_penalty",
            "missing_required_metric_penalty",
            "unsupported_turf_penalty",
        )
        for name in finite:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")

    def evaluate(
        self,
        cohort: ChipTrialCohort,
        metrics: Mapping[str, float | None],
        *,
        turf_contact_status: str | None = None,
    ) -> tuple[float, bool]:
        """Return loss and constraint state without imputing missing metrics."""
        if not isinstance(cohort, ChipTrialCohort):
            raise TypeError("cohort must be a ChipTrialCohort")
        loss = self._cohort_penalty(cohort)
        carry = metrics.get("carry_m")
        lateral = metrics.get("lateral_m")
        penetration = metrics.get("peak_turf_penetration_m")
        if carry is not None:
            loss += ((carry - self.target_carry_m) / self.carry_tolerance_m) ** 2
        if lateral is not None:
            loss += (lateral / self.lateral_tolerance_m) ** 2
        if self.include_turf_penetration and penetration is not None:
            loss += (penetration / self.maximum_turf_penetration_m) ** 2
        required_missing = cohort in {
            ChipTrialCohort.BALL_FIRST,
            ChipTrialCohort.BALL_ONLY,
            ChipTrialCohort.SIMULTANEOUS,
        } and (carry is None or lateral is None)
        if required_missing:
            loss += self.missing_required_metric_penalty
        turf_unsupported = turf_contact_status in {
            "outside_calibrated_domain",
            "step_limit",
            "cancelled",
        }
        if turf_unsupported:
            loss += self.unsupported_turf_penalty
        margin = metrics.get("ground_after_ball_margin_s")
        violated = cohort in {
            ChipTrialCohort.GROUND_FIRST,
            ChipTrialCohort.SIMULTANEOUS,
            ChipTrialCohort.GROUND_ONLY_MISS,
            ChipTrialCohort.NO_CONTACT_MISS,
            ChipTrialCohort.NUMERICAL_FAILURE,
        }
        violated = violated or (margin is not None and margin <= 0.0)
        violated = violated or (
            self.include_turf_penetration
            and penetration is not None
            and penetration > self.maximum_turf_penetration_m
        )
        violated = violated or required_missing or turf_unsupported
        return float(loss), violated

    def _cohort_penalty(self, cohort: ChipTrialCohort) -> float:
        """Return the explicit discrete penalty for one mutually exclusive cohort."""
        return {
            ChipTrialCohort.BALL_FIRST: 0.0,
            ChipTrialCohort.BALL_ONLY: 0.0,
            ChipTrialCohort.GROUND_FIRST: self.ground_first_penalty,
            ChipTrialCohort.SIMULTANEOUS: self.simultaneous_penalty,
            ChipTrialCohort.GROUND_ONLY_MISS: self.ground_only_miss_penalty,
            ChipTrialCohort.NO_CONTACT_MISS: self.no_contact_miss_penalty,
            ChipTrialCohort.NUMERICAL_FAILURE: self.numerical_failure_penalty,
        }[cohort]


__all__ = ["ChipLossModel", "objective_id_for_target_carry"]
