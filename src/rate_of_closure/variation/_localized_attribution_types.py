"""Strict noncausal presentation contract for paired localized interventions.

The contract accepts only explicitly retained baseline/perturbed observations.
It never derives attribution from Monte Carlo scatter or correlation. Current
Rate ensembles do not yet produce this authority; both UIs therefore fail
closed until a future executor supplies genuine isolated pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import cast

from shared.python.contracts import require

AUTHORITY_SCHEMA_ID = "rate-of-closure/localized-attribution-authority"
AUTHORITY_SCHEMA_VERSION = 1
VIEW_SCHEMA_ID = "rate-of-closure/localized-attribution-view"
VIEW_SCHEMA_VERSION = 1
INTERPRETATION = "paired-planted-intervention-noncausal"

_JOINT_BY_VARIABLE = {
    "swing_sim.swing.shoulder_commanded_torque_offset_nm": "joint.shoulder",
    "swing_sim.swing.wrist_commanded_torque_offset_nm": "joint.wrist",
}
_STATE_NAMES = {"position_x_m", "position_y_m", "position_z_m"}
_IMPACT_NAMES = {
    "impact_time_s",
    "clubhead_speed_mps",
    "spin_loft_deg",
    "face_to_path_deg",
    "spin_axis_tilt_deg",
}
_SHOT_NAMES = {
    "ball_speed_mph",
    "launch_angle_deg",
    "launch_azimuth_deg",
    "spin_rpm",
    "carry_m",
    "lateral_m",
    "max_height_m",
    "flight_time_s",
    "landing_angle_deg",
}
_STATUS_VALUES = {
    "evaluated_hit",
    "evaluated_no_impact",
    "numerical_failure",
}
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class TrialStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Typed outcome retained by one side of a pair."""

    EVALUATED_HIT = "evaluated_hit"
    EVALUATED_NO_IMPACT = "evaluated_no_impact"
    NUMERICAL_FAILURE = "numerical_failure"


class Availability(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Reason a paired response is available or unavailable."""

    AVAILABLE = "available"
    NO_IMPACT_UNAVAILABLE = "no_impact_unavailable"
    NUMERICAL_FAILURE = "numerical_failure"
    NONFINITE_UNAVAILABLE = "nonfinite_unavailable"


def _stable_text(value: object, label: str) -> str:
    require(isinstance(value, str), f"{label} must be a string", value)
    text = cast(str, value)
    require(text and text == text.strip(), f"{label} must be a stable ID", text)
    require(not any(ord(char) < 32 for char in text), f"{label} has controls", text)
    require(not text.startswith(_FORMULA_PREFIXES), f"{label} is unsafe", text)
    return text


def _finite(value: object, label: str) -> float:
    require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{label} must be a finite number",
        value,
    )
    result = float(cast(float, value))
    require(math.isfinite(result), f"{label} must be finite", result)
    return result


def _index(value: object, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a nonnegative integer",
        value,
    )
    return cast(int, value)


def _exact(raw: object, fields: set[str], label: str) -> dict[str, object]:
    require(isinstance(raw, dict), f"{label} must be an object", raw)
    record = cast(dict[str, object], raw)
    require(set(record) == fields, f"{label} has invalid fields", tuple(record))
    return record


@dataclass(frozen=True)
class AttributionSource:
    """One authored localized torque source."""

    spec_id: str
    variable_key: str
    joint_id: str
    time_window_s: tuple[float, float]
    unit: str

    def __post_init__(self) -> None:
        _stable_text(self.spec_id, "spec_id")
        variable = _stable_text(self.variable_key, "variable_key")
        require(
            variable in _JOINT_BY_VARIABLE, "unsupported localized variable", variable
        )
        require(
            self.joint_id == _JOINT_BY_VARIABLE[variable],
            "joint mismatch",
            self.joint_id,
        )
        require(self.unit == "N·m", "localized source unit must be N·m", self.unit)
        require(len(self.time_window_s) == 2, "window must contain start and end")
        start = _finite(self.time_window_s[0], "window start")
        end = _finite(self.time_window_s[1], "window end")
        require(0.0 <= start < end, "window must be finite half-open start < end")
        object.__setattr__(self, "time_window_s", (start, end))


@dataclass(frozen=True)
class AttributionTarget:
    """One selectable state, impact, or shot target."""

    target_id: str
    kind: str
    name: str
    unit: str
    time_s: float | None
    point_id: str | None
    coordinate_frame: str | None

    def __post_init__(self) -> None:
        _stable_text(self.target_id, "target_id")
        _stable_text(self.name, "target name")
        _stable_text(self.unit, "target unit")
        require(
            self.kind in {"state", "impact", "shot"}, "invalid target kind", self.kind
        )
        if self.kind == "state":
            require(self.name in _STATE_NAMES, "invalid state target", self.name)
            require(self.time_s is not None, "state target requires time_s")
            require(
                _finite(self.time_s, "target time") >= 0.0, "target time must be >= 0"
            )
            require(self.point_id is not None, "state target requires point_id")
            require(
                _stable_text(self.point_id, "point_id").startswith("swing."),
                "state point must be spatial swing.*",
            )
            require(
                self.coordinate_frame == "app_frame:x_target,y_up,z_right",
                "invalid state frame",
            )
            return
        names = _IMPACT_NAMES if self.kind == "impact" else _SHOT_NAMES
        require(self.name in names, f"invalid {self.kind} target", self.name)
        require(
            self.time_s is None
            and self.point_id is None
            and self.coordinate_frame is None,
            f"{self.kind} target cannot carry state locus",
        )


@dataclass(frozen=True)
class AttributionObservation:
    """One raw retained baseline/perturbed target response."""

    source_spec_id: str
    target_id: str
    baseline_trial_index: int
    perturbed_trial_index: int
    baseline_status: TrialStatus
    perturbed_status: TrialStatus
    baseline_source_value: float
    perturbed_source_value: float
    baseline_target_value: float | None
    perturbed_target_value: float | None
    response: float | None
    availability: Availability

    def __post_init__(self) -> None:
        _stable_text(self.source_spec_id, "source_spec_id")
        _stable_text(self.target_id, "target_id")
        baseline = _index(self.baseline_trial_index, "baseline_trial_index")
        perturbed = _index(self.perturbed_trial_index, "perturbed_trial_index")
        require(baseline != perturbed, "baseline and perturbed trials must differ")
        _finite(self.baseline_source_value, "baseline source value")
        _finite(self.perturbed_source_value, "perturbed source value")
        values = (self.baseline_target_value, self.perturbed_target_value)
        if self.availability is Availability.AVAILABLE:
            require(
                all(value is not None for value in values),
                "available pair requires target values",
            )
            require(self.response is not None, "available pair requires response")
            expected = _finite(values[1], "perturbed target") - _finite(
                values[0], "baseline target"
            )
            require(
                math.isclose(
                    _finite(self.response, "response"), expected, abs_tol=1e-12
                ),
                "response must equal perturbed minus baseline",
            )
        else:
            require(self.response is None, "unavailable pair response must be null")
            require(
                any(value is None for value in values),
                "unavailable pair requires a null target value",
            )


@dataclass(frozen=True)
class AttributionAuthority:
    """Strict collection of retained localized paired observations."""

    authority_id: str
    sources: tuple[AttributionSource, ...]
    targets: tuple[AttributionTarget, ...]
    observations: tuple[AttributionObservation, ...]
    interpretation: str = INTERPRETATION

    def __post_init__(self) -> None:
        _stable_text(self.authority_id, "authority_id")
        require(self.interpretation == INTERPRETATION, "invalid interpretation")
        source_ids = {source.spec_id for source in self.sources}
        target_ids = {target.target_id for target in self.targets}
        require(
            len(source_ids) == len(self.sources) and source_ids,
            "source IDs must be unique and nonempty",
        )
        require(
            len(target_ids) == len(self.targets) and target_ids,
            "target IDs must be unique and nonempty",
        )
        require(bool(self.observations), "observations must be nonempty")
        keys: set[tuple[str, str, int, int]] = set()
        targets = {target.target_id: target for target in self.targets}
        for observation in self.observations:
            require(
                observation.source_spec_id in source_ids, "unknown observation source"
            )
            require(observation.target_id in target_ids, "unknown observation target")
            _require_availability(observation, targets[observation.target_id])
            key = (
                observation.source_spec_id,
                observation.target_id,
                observation.baseline_trial_index,
                observation.perturbed_trial_index,
            )
            require(key not in keys, "duplicate attribution observation", key)
            keys.add(key)


@dataclass(frozen=True)
class AttributionViewDefinition:
    """Versioned persisted attribution selection."""

    authority_id: str
    source_spec_id: str
    target_id: str
    baseline_trial_index: int
    perturbed_trial_index: int

    def __post_init__(self) -> None:
        _stable_text(self.authority_id, "authority_id")
        _stable_text(self.source_spec_id, "source_spec_id")
        _stable_text(self.target_id, "target_id")
        require(
            _index(self.baseline_trial_index, "baseline_trial_index")
            != _index(self.perturbed_trial_index, "perturbed_trial_index"),
            "baseline and perturbed trials must differ",
        )


@dataclass(frozen=True)
class AttributionDenominator:
    """Typed availability ledger for one selected source/target."""

    total_pairs: int
    available_pairs: int
    typed_no_impact_pairs: int
    unavailable_no_impact_pairs: int
    failed_pairs: int
    nonfinite_pairs: int


@dataclass(frozen=True)
class AttributionView:
    """Resolved selection and its raw cohort denominator."""

    source: AttributionSource
    target: AttributionTarget
    selected: AttributionObservation
    observations: tuple[AttributionObservation, ...]
    denominator: AttributionDenominator


def _require_availability(
    observation: AttributionObservation, target: AttributionTarget
) -> None:
    statuses = {observation.baseline_status, observation.perturbed_status}
    if TrialStatus.NUMERICAL_FAILURE in statuses:
        expected = Availability.NUMERICAL_FAILURE
    elif target.kind != "state" and TrialStatus.EVALUATED_NO_IMPACT in statuses:
        expected = Availability.NO_IMPACT_UNAVAILABLE
    elif (
        observation.baseline_target_value is None
        or observation.perturbed_target_value is None
    ):
        expected = Availability.NONFINITE_UNAVAILABLE
    else:
        expected = Availability.AVAILABLE
    require(
        observation.availability is expected,
        "availability does not match typed outcomes",
        observation.availability,
    )
