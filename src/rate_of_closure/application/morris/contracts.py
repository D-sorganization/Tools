"""Strict versioned wire contracts for the Rate Morris authority."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from rate_of_closure.club.types import SPEC_BOUNDS

if TYPE_CHECKING:
    from rate_of_closure.simulation.records import SimulationConfig
    from shared.python.swing_sim.variation.morris_design import (
        MorrisDesign,
        MorrisFactor,
    )

MORRIS_REQUEST_SCHEMA_ID = "rate-of-closure/morris-request"
MORRIS_JOB_SCHEMA_ID = "rate-of-closure/morris-job"
MORRIS_AUTHORITY_SCHEMA_VERSION = 1
JobStatus = Literal["queued", "running", "completed", "cancelled", "failed"]

_REQUEST_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "request_id",
        "base",
        "factors",
        "trajectories",
        "levels",
        "seed",
        "minimum_effects",
        "worker_count",
    }
)
_BASE_FIELDS = frozenset(
    {
        "club_name",
        "support_mode",
        "tee_height_m",
        "plane_yaw_deg",
        "plane_side_tilt_deg",
        "plane_forward_tilt_deg",
        "pendulum_m1_kg",
        "pendulum_l1_m",
        "pendulum_lc1_m",
        "pendulum_i1_kg_m2",
        "pendulum_m2_kg",
        "pendulum_l2_m",
        "pendulum_lc2_m",
        "pendulum_i2_kg_m2",
        "damping_shoulder",
        "damping_wrist",
        "swing_duration_s",
        "flight_model",
        "impact_offset_toe_mm",
        "impact_offset_high_mm",
    }
)
_FACTOR_FIELDS = frozenset({"spec_id", "variable_key", "lower", "upper", "unit"})
_PENDULUM_POSITIVE = (
    "pendulum_m1_kg",
    "pendulum_l1_m",
    "pendulum_lc1_m",
    "pendulum_i1_kg_m2",
    "pendulum_m2_kg",
    "pendulum_l2_m",
    "pendulum_lc2_m",
    "pendulum_i2_kg_m2",
)
_DAMPING_KEYS = frozenset(
    {"swing_sim.swing.damping_shoulder", "swing_sim.swing.damping_wrist"}
)
_TOE_KEY = "swing_sim.impact.delivery.impact_offset_toe_mm"
_HIGH_KEY = "swing_sim.impact.delivery.impact_offset_high_mm"
_HEAD_MASS_KEY = "swing_sim.club.head_mass_kg"
_HEAD_MOI_KEY = "swing_sim.club.head_moi_kg_m2"
_TEE_KEY = "swing_sim.ball_setup.tee_height_m"
_MAX_MORRIS_SAMPLES = 100_000
_MAX_MORRIS_OBSERVATION_CELLS = 1_000_000
_MAX_MORRIS_WORKERS = 32


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer within [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class MorrisBaseRequest:
    """Primitive-only base simulation fields accepted on the wire."""

    values: dict[str, Any]

    def simulation_config(self) -> SimulationConfig:
        """Reconstruct the one pinned passive fixed-ball authority config."""
        from rate_of_closure.club.library import CLUB_LIBRARY
        from rate_of_closure.model import ImpactScenario
        from rate_of_closure.simulation.contact import ContactMode
        from rate_of_closure.simulation.records import SimulationConfig
        from shared.python.swing_sim.ball_setup import BallSetup, BallSupportMode
        from shared.python.swing_sim.types import PendulumParameters, PlaneOrientation

        value = self.values
        support = BallSupportMode(str(value["support_mode"]))
        scenario = ImpactScenario(
            clubhead_speed_mph=113.0,
            impact_offset_toe_mm=float(value["impact_offset_toe_mm"]),
            impact_offset_high_mm=float(value["impact_offset_high_mm"]),
        )
        plane = PlaneOrientation(
            float(value["plane_yaw_deg"]),
            float(value["plane_side_tilt_deg"]),
            float(value["plane_forward_tilt_deg"]),
        )
        parameters = PendulumParameters(
            *(
                float(value[name])
                for name in (
                    "pendulum_m1_kg",
                    "pendulum_l1_m",
                    "pendulum_lc1_m",
                    "pendulum_i1_kg_m2",
                    "pendulum_m2_kg",
                    "pendulum_l2_m",
                    "pendulum_lc2_m",
                    "pendulum_i2_kg_m2",
                    "damping_shoulder",
                    "damping_wrist",
                )
            )
        )
        return SimulationConfig(
            scenario=scenario,
            club=CLUB_LIBRARY[str(value["club_name"])],
            ball_setup=BallSetup(support, float(value["tee_height_m"])),
            source_kind="double_pendulum",
            plane=plane,
            flight_model=str(value["flight_model"]),
            swing_duration_s=float(value["swing_duration_s"]),
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
            pendulum_parameters=parameters,
        )


@dataclass(frozen=True)
class MorrisAuthorityRequest:
    """Validated v1 request and deterministic construction methods."""

    request_id: str
    base: MorrisBaseRequest
    factors: tuple[MorrisFactor, ...]
    trajectories: int
    levels: int
    seed: int
    minimum_effects: int
    worker_count: int

    def base_config(self) -> SimulationConfig:
        """Return the reconstructed immutable simulation config."""
        return self.base.simulation_config()

    def design(self) -> MorrisDesign:
        """Generate this request's deterministic Morris design."""
        from shared.python.swing_sim.variation.morris_design import (
            generate_morris_design,
        )

        return generate_morris_design(
            self.factors, self.trajectories, self.levels, self.seed
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the exact validated v1 request document."""
        return {
            "schema_id": MORRIS_REQUEST_SCHEMA_ID,
            "schema_version": MORRIS_AUTHORITY_SCHEMA_VERSION,
            "request_id": self.request_id,
            "base": dict(self.base.values),
            "factors": [
                {
                    "spec_id": factor.spec_id,
                    "variable_key": factor.variable_key,
                    "lower": factor.lower,
                    "upper": factor.upper,
                    "unit": factor.unit,
                }
                for factor in self.factors
            ],
            "trajectories": self.trajectories,
            "levels": self.levels,
            "seed": self.seed,
            "minimum_effects": self.minimum_effects,
            "worker_count": self.worker_count,
        }

    @property
    def total_samples(self) -> int:
        """Return the exact bounded design sample count."""
        return self.trajectories * (len(self.factors) + 1)


@dataclass(frozen=True)
class MorrisJobEnvelope:
    """Stable job status envelope; only completed jobs carry reports."""

    job_id: str
    request_id: str
    status: JobStatus
    completed_samples: int
    total_samples: int
    cancel_requested: bool = False
    report: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    @classmethod
    def running(
        cls,
        job_id: str,
        request_id: str,
        progress: tuple[int, int],
        cancel: bool,
    ) -> MorrisJobEnvelope:
        """Build a running envelope."""
        done, total = progress
        return cls(job_id, request_id, "running", done, total, cancel)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize exact v1 snake-case fields."""
        return {
            "schema_id": MORRIS_JOB_SCHEMA_ID,
            "schema_version": 1,
            "job_id": self.job_id,
            "request_id": self.request_id,
            "status": self.status,
            "completed_samples": self.completed_samples,
            "total_samples": self.total_samples,
            "cancel_requested": self.cancel_requested,
            "report": self.report,
            "error": self.error,
        }


def _parse_base(value: object) -> MorrisBaseRequest:
    from rate_of_closure.club.library import CLUB_LIBRARY
    from shared.python.swing_sim.flight.registry import FlightModelType

    item = dict(exact_mapping(value, _BASE_FIELDS, "Morris base"))
    for name in _BASE_FIELDS - {"club_name", "support_mode", "flight_model"}:
        item[name] = _finite(item[name], f"base {name}")
    if item["support_mode"] not in {"ground", "tee"}:
        raise ValueError("base support_mode is unsupported")
    if item["club_name"] not in CLUB_LIBRARY:
        raise ValueError("base club_name is not in the club library")
    if item["flight_model"] not in {model.value for model in FlightModelType}:
        raise ValueError("base flight_model is unsupported")
    _validate_base_physics(item)
    if item["support_mode"] == "ground" and item["tee_height_m"] != 0.0:
        raise ValueError("ground support requires tee_height_m == 0")
    result = MorrisBaseRequest(item)
    result.simulation_config()
    return result


def _validate_base_physics(item: dict[str, Any]) -> None:
    if item["tee_height_m"] < 0.0:
        raise ValueError("base tee_height_m must be nonnegative")
    if item["swing_duration_s"] <= 0.0:
        raise ValueError("base swing_duration_s must be positive")
    if any(item[name] <= 0.0 for name in _PENDULUM_POSITIVE):
        raise ValueError(
            "base pendulum masses, lengths, centers, and inertias must be positive"
        )
    if item["pendulum_lc1_m"] > item["pendulum_l1_m"]:
        raise ValueError("base pendulum_lc1_m must not exceed pendulum_l1_m")
    if item["pendulum_lc2_m"] > item["pendulum_l2_m"]:
        raise ValueError("base pendulum_lc2_m must not exceed pendulum_l2_m")
    if item["damping_shoulder"] < 0.0 or item["damping_wrist"] < 0.0:
        raise ValueError("base pendulum damping must be nonnegative")
    _bounded(item["impact_offset_toe_mm"], (-80.0, 80.0), "base toe offset")
    _bounded(item["impact_offset_high_mm"], (-40.0, 40.0), "base high offset")


def _bounded(value: float, bounds: tuple[float, float], name: str) -> None:
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} must be within [{bounds[0]}, {bounds[1]}]")


def _parse_factors(value: object, config: SimulationConfig) -> tuple[MorrisFactor, ...]:
    from rate_of_closure.variation.request_builder import apply_global_simulation_values

    if not isinstance(value, list) or not value:
        raise TypeError("factors must be a nonempty array")
    factors = tuple(_parse_factor(item) for item in value)
    if len({factor.spec_id for factor in factors}) != len(factors):
        raise ValueError("factor spec_id values must be unique")
    if len({factor.variable_key for factor in factors}) != len(factors):
        raise ValueError("factor variable_key values must be unique")
    for factor in factors:
        _validate_factor_endpoint(factor.variable_key, factor.lower, config)
        _validate_factor_endpoint(factor.variable_key, factor.upper, config)
        apply_global_simulation_values(config, {factor.variable_key: factor.lower})
        apply_global_simulation_values(config, {factor.variable_key: factor.upper})
    return factors


def _parse_factor(value: object) -> MorrisFactor:
    from rate_of_closure.variation.morris_rate_adapter import RATE_MORRIS_VARIABLE_KEYS
    from shared.python.swing_sim.variation.morris_design import MorrisFactor
    from shared.python.swing_sim.variation.spec import variable_registry

    item = exact_mapping(value, _FACTOR_FIELDS, "Morris factor")
    key = item["variable_key"]
    if not isinstance(key, str) or key not in RATE_MORRIS_VARIABLE_KEYS:
        raise ValueError("factor variable_key is unsupported")
    unit = variable_registry()[key].unit
    if item["unit"] != unit:
        raise ValueError("factor unit must match the registry unit")
    lower = _finite(item["lower"], "factor lower")
    upper = _finite(item["upper"], "factor upper")
    if lower >= upper:
        raise ValueError("factor bounds must satisfy lower < upper")
    return MorrisFactor(
        stable_id(item["spec_id"], "factor spec_id"),
        key,
        lower,
        upper,
        unit,
    )


def _validate_factor_endpoint(key: str, value: float, config: SimulationConfig) -> None:
    from shared.python.swing_sim.ball_setup import BallSupportMode

    if key in _DAMPING_KEYS and value < 0.0:
        raise ValueError("damping factor endpoints must be nonnegative")
    if key == _TOE_KEY:
        _bounded(value, (-80.0, 80.0), "toe factor endpoint")
    if key == _HIGH_KEY:
        _bounded(value, (-40.0, 40.0), "high factor endpoint")
    if key == _HEAD_MASS_KEY:
        _bounded(value, SPEC_BOUNDS["head_mass_kg"], "head mass factor endpoint")
    if key == _HEAD_MOI_KEY:
        _bounded(
            value,
            SPEC_BOUNDS["moi_about_shaft_kg_m2"],
            "head MOI factor endpoint",
        )
    if key == _TEE_KEY:
        if config.ball_setup.support_mode is not BallSupportMode.TEE:
            raise ValueError("tee_height_m factor requires tee support")
        if value < 0.0:
            raise ValueError("tee height factor endpoints must be nonnegative")


def parse_morris_request(value: object) -> MorrisAuthorityRequest:
    """Parse an exact v1 request and validate its full allocation."""
    item = exact_mapping(value, _REQUEST_FIELDS, "Morris request")
    if item["schema_id"] != MORRIS_REQUEST_SCHEMA_ID:
        raise ValueError("unsupported Morris request schema ID")
    if item["schema_version"] != MORRIS_AUTHORITY_SCHEMA_VERSION:
        raise ValueError("unsupported Morris request schema version")
    base = _parse_base(item["base"])
    factors = _parse_factors(item["factors"], base.simulation_config())
    trajectories = _integer(item["trajectories"], "trajectories", 1, 2**31 - 1)
    levels = _integer(item["levels"], "levels", 4, 10_000)
    if levels % 2:
        raise ValueError("levels must be even")
    total = trajectories * (len(factors) + 1)
    if total > _MAX_MORRIS_SAMPLES or total * 17 > _MAX_MORRIS_OBSERVATION_CELLS:
        raise ValueError("Morris sample allocation exceeds resource limits")
    return MorrisAuthorityRequest(
        stable_id(item["request_id"], "request_id"),
        base,
        factors,
        trajectories,
        levels,
        _integer(item["seed"], "seed", 0, 2**32 - 1),
        _integer(item["minimum_effects"], "minimum_effects", 2, trajectories),
        _integer(item["worker_count"], "worker_count", 1, _MAX_MORRIS_WORKERS),
    )


__all__ = [
    "MORRIS_AUTHORITY_SCHEMA_VERSION",
    "MORRIS_JOB_SCHEMA_ID",
    "MORRIS_REQUEST_SCHEMA_ID",
    "MorrisAuthorityRequest",
    "MorrisJobEnvelope",
    "parse_morris_request",
]
