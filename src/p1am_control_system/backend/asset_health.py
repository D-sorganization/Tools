"""Deterministic synthetic asset statistics and maintenance advisories."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    # Type checkers must see the real 3.11 symbol; TYPE_CHECKING is always
    # true for them and always false at runtime, so this needs no version
    # test and never degrades StrEnum members to bare `str`.
    from enum import StrEnum
else:
    from enum_compat import StrEnum


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return value


class AdvisoryCode(StrEnum):
    CALIBRATION_DUE = "calibration_due"
    DRIFT = "drift"
    FLATLINE = "flatline"
    COMMAND_FEEDBACK_MISMATCH = "command_feedback_mismatch"
    NOISY_SIGNAL = "noisy_signal"


class AssetHealthPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    drift_limit: float = Field(default=2.0, gt=0)
    flatline_duration: timedelta = timedelta(minutes=5)
    flatline_span: float = Field(default=0.01, ge=0)
    mismatch_duration: timedelta = timedelta(seconds=30)
    noise_standard_deviation: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _positive_durations(self) -> AssetHealthPolicy:
        if self.flatline_duration <= timedelta(0):
            raise ValueError("flatline_duration must be positive")
        if self.mismatch_duration <= timedelta(0):
            raise ValueError("mismatch_duration must be positive")
        return self


class AssetObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    observed_at: datetime
    value: float
    reference: float
    command: bool
    feedback: bool
    running: bool

    @field_validator("observed_at")
    @classmethod
    def _timestamp_is_aware(cls, value: datetime) -> datetime:
        return _aware(value)

    @field_validator("value", "reference")
    @classmethod
    def _finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("observation values must be finite")
        return value


class AssetCounters(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime_seconds: float = Field(ge=0)
    start_count: int = Field(ge=0)


class DeviceStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_count: int = Field(gt=0)
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float = Field(ge=0)


class MaintenanceAdvisory(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: AdvisoryCode
    asset_id: str
    detected_at: datetime
    detail: str
    classification: Literal["maintenance_advisory"] = "maintenance_advisory"
    authoritative_trip: Literal[False] = False


class AssetHealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    generated_at: datetime
    counters: AssetCounters
    statistics: DeviceStatistics
    advisories: tuple[MaintenanceAdvisory, ...]
    data_classification: Literal["synthetic"] = "synthetic"


class AssetHealthService:
    def __init__(
        self,
        policy: AssetHealthPolicy,
        now: Callable[[], datetime],
    ) -> None:
        self._policy = policy
        self._now = now

    @staticmethod
    def _validate_observations(
        observations: Sequence[AssetObservation],
    ) -> tuple[AssetObservation, ...]:
        normalized = tuple(observations)
        if len(normalized) < 2:
            raise ValueError("at least two observations are required")
        if any(
            current.observed_at <= previous.observed_at
            for previous, current in zip(normalized, normalized[1:], strict=False)
        ):
            raise ValueError("observations must be strictly time ordered")
        return normalized

    @staticmethod
    def _counters(observations: tuple[AssetObservation, ...]) -> AssetCounters:
        runtime = sum(
            (current.observed_at - previous.observed_at).total_seconds()
            for previous, current in zip(observations, observations[1:], strict=False)
            if previous.running
        )
        starts = int(observations[0].running) + sum(
            int(current.running and not previous.running)
            for previous, current in zip(observations, observations[1:], strict=False)
        )
        return AssetCounters(runtime_seconds=runtime, start_count=starts)

    @staticmethod
    def _mismatch_span(observations: tuple[AssetObservation, ...]) -> timedelta:
        mismatched = [item for item in observations if item.command != item.feedback]
        if len(mismatched) < 2:
            return timedelta(0)
        trailing: list[AssetObservation] = []
        for item in reversed(observations):
            if item.command == item.feedback:
                break
            trailing.append(item)
        if len(trailing) < 2:
            return timedelta(0)
        return trailing[0].observed_at - trailing[-1].observed_at

    def assess(
        self,
        asset_id: str,
        observations: Sequence[AssetObservation],
        *,
        calibration_due_at: datetime,
    ) -> AssetHealthReport:
        if not asset_id.startswith("SYNTHETIC."):
            raise ValueError("asset_id must begin with SYNTHETIC.")
        normalized = self._validate_observations(observations)
        generated_at = _aware(self._now())
        calibration_due_at = _aware(calibration_due_at)
        values = [item.value for item in normalized]
        stats = DeviceStatistics(
            sample_count=len(values),
            minimum=min(values),
            maximum=max(values),
            mean=statistics.fmean(values),
            standard_deviation=statistics.pstdev(values),
        )
        advisories: list[MaintenanceAdvisory] = []

        def add(code: AdvisoryCode, detail: str) -> None:
            advisories.append(
                MaintenanceAdvisory(
                    code=code,
                    asset_id=asset_id,
                    detected_at=generated_at,
                    detail=detail,
                )
            )

        if generated_at >= calibration_due_at:
            add(AdvisoryCode.CALIBRATION_DUE, "Calibration due date has passed")
        latest = normalized[-1]
        if abs(latest.value - latest.reference) > self._policy.drift_limit:
            add(AdvisoryCode.DRIFT, "Value-to-reference deviation exceeds policy")
        duration = normalized[-1].observed_at - normalized[0].observed_at
        if (
            duration >= self._policy.flatline_duration
            and stats.maximum - stats.minimum <= self._policy.flatline_span
        ):
            add(AdvisoryCode.FLATLINE, "Signal span remains below flatline policy")
        if self._mismatch_span(normalized) >= self._policy.mismatch_duration:
            add(
                AdvisoryCode.COMMAND_FEEDBACK_MISMATCH,
                "Command and feedback remain inconsistent",
            )
        if stats.standard_deviation > self._policy.noise_standard_deviation:
            add(AdvisoryCode.NOISY_SIGNAL, "Signal variability exceeds noise policy")
        return AssetHealthReport(
            asset_id=asset_id,
            generated_at=generated_at,
            counters=self._counters(normalized),
            statistics=stats,
            advisories=tuple(advisories),
        )
