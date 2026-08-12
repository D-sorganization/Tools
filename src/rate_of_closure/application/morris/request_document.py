"""Canonical UI-neutral construction of validated Morris requests."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rate_of_closure.club.types import SPEC_BOUNDS
from shared.python.swing_sim.ball_setup import BallSupportMode

if TYPE_CHECKING:
    from rate_of_closure.simulation.records import SimulationConfig

from .contracts import (
    MORRIS_AUTHORITY_SCHEMA_VERSION,
    MORRIS_REQUEST_SCHEMA_ID,
    MorrisAuthorityRequest,
    parse_morris_request,
)

CANONICAL_MORRIS_FACTOR_KEYS = (
    "swing_sim.swing.yaw_deg",
    "swing_sim.swing.side_tilt_deg",
    "swing_sim.swing.forward_tilt_deg",
    "swing_sim.swing.damping_shoulder",
    "swing_sim.swing.damping_wrist",
    "swing_sim.impact.delivery.impact_offset_toe_mm",
    "swing_sim.impact.delivery.impact_offset_high_mm",
    "swing_sim.club.head_mass_kg",
    "swing_sim.club.head_moi_kg_m2",
    "swing_sim.ball_setup.tee_height_m",
)
_TEE_KEY = CANONICAL_MORRIS_FACTOR_KEYS[-1]
_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9._:-]+")
_KNOWN_BOUNDS = {
    CANONICAL_MORRIS_FACTOR_KEYS[3]: (0.0, math.inf),
    CANONICAL_MORRIS_FACTOR_KEYS[4]: (0.0, math.inf),
    CANONICAL_MORRIS_FACTOR_KEYS[5]: (-80.0, 80.0),
    CANONICAL_MORRIS_FACTOR_KEYS[6]: (-40.0, 40.0),
    CANONICAL_MORRIS_FACTOR_KEYS[7]: SPEC_BOUNDS["head_mass_kg"],
    CANONICAL_MORRIS_FACTOR_KEYS[8]: SPEC_BOUNDS["moi_about_shaft_kg_m2"],
    _TEE_KEY: (0.0, math.inf),
}


@dataclass(frozen=True)
class MorrisFactorDraft:
    """One editable bounded factor before authoritative request validation."""

    variable_key: str
    enabled: bool
    lower: float
    upper: float


def _base_value(config: SimulationConfig, key: str) -> float:
    values = {
        CANONICAL_MORRIS_FACTOR_KEYS[0]: config.plane.yaw_deg,
        CANONICAL_MORRIS_FACTOR_KEYS[1]: config.plane.side_tilt_deg,
        CANONICAL_MORRIS_FACTOR_KEYS[2]: config.plane.forward_tilt_deg,
        CANONICAL_MORRIS_FACTOR_KEYS[3]: config.pendulum_parameters.d1,
        CANONICAL_MORRIS_FACTOR_KEYS[4]: config.pendulum_parameters.d2,
        CANONICAL_MORRIS_FACTOR_KEYS[5]: config.scenario.impact_offset_toe_mm,
        CANONICAL_MORRIS_FACTOR_KEYS[6]: config.scenario.impact_offset_high_mm,
        CANONICAL_MORRIS_FACTOR_KEYS[7]: config.club.head_mass_kg,
        CANONICAL_MORRIS_FACTOR_KEYS[8]: config.club.moi_about_shaft_kg_m2,
        _TEE_KEY: config.ball_setup.tee_height_m,
    }
    return float(values[key])


def suggested_factor_drafts(config: SimulationConfig) -> tuple[MorrisFactorDraft, ...]:
    """Return ordered registry-derived bounds applicable to ``config``."""
    from rate_of_closure.simulation.records import SimulationConfig

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    keys: tuple[str, ...] = CANONICAL_MORRIS_FACTOR_KEYS
    if config.ball_setup.support_mode is BallSupportMode.GROUND:
        keys = keys[:-1]
    result = tuple(_suggested_draft(config, key) for key in keys)
    if any(item.lower >= item.upper for item in result):
        raise ValueError("suggested factor bounds collapsed after physical clamping")
    return result


def _suggested_draft(config: SimulationConfig, key: str) -> MorrisFactorDraft:
    from shared.python.swing_sim.variation.spec import variable_registry

    definition = variable_registry()[key]
    center = _base_value(config, key)
    lower = center - 2.0 * definition.typical_scale
    upper = center + 2.0 * definition.typical_scale
    if key in _KNOWN_BOUNDS:
        minimum, maximum = _KNOWN_BOUNDS[key]
        lower = max(lower, minimum)
        upper = min(upper, maximum)
    return MorrisFactorDraft(key, True, lower, upper)


def _base_document(config: SimulationConfig) -> dict[str, object]:
    parameters = config.pendulum_parameters
    return {
        "club_name": config.club.name,
        "support_mode": config.ball_setup.support_mode.value,
        "tee_height_m": config.ball_setup.tee_height_m,
        "plane_yaw_deg": config.plane.yaw_deg,
        "plane_side_tilt_deg": config.plane.side_tilt_deg,
        "plane_forward_tilt_deg": config.plane.forward_tilt_deg,
        "pendulum_m1_kg": parameters.m1,
        "pendulum_l1_m": parameters.l1,
        "pendulum_lc1_m": parameters.lc1,
        "pendulum_i1_kg_m2": parameters.i1,
        "pendulum_m2_kg": parameters.m2,
        "pendulum_l2_m": parameters.l2,
        "pendulum_lc2_m": parameters.lc2,
        "pendulum_i2_kg_m2": parameters.i2,
        "damping_shoulder": parameters.d1,
        "damping_wrist": parameters.d2,
        "swing_duration_s": config.swing_duration_s,
        "flight_model": config.flight_model,
        "impact_offset_toe_mm": config.scenario.impact_offset_toe_mm,
        "impact_offset_high_mm": config.scenario.impact_offset_high_mm,
    }


def spec_id_for_key(key: str) -> str:
    """Return the portable authority spec ID derived from a registry key."""
    result = _ID_CHARACTERS.sub("-", key).strip("-.")
    if not result or len(result) > 128:
        raise ValueError("factor variable_key cannot form a stable spec_id")
    return result


def _factor_documents(
    config: SimulationConfig, drafts: tuple[MorrisFactorDraft, ...]
) -> list[dict[str, object]]:
    from shared.python.swing_sim.variation.spec import variable_registry

    registry = variable_registry()
    for draft in drafts:
        if type(draft.enabled) is not bool:
            raise TypeError("factor enabled must be boolean")
        if (
            not isinstance(draft.variable_key, str)
            or not draft.variable_key
            or draft.variable_key != draft.variable_key.strip()
        ):
            raise TypeError("factor variable_key must be a nonempty trimmed string")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (draft.lower, draft.upper)
        ):
            raise TypeError("factor bounds must be finite numbers")
    enabled = tuple(draft for draft in drafts if draft.enabled)
    if not enabled:
        raise ValueError("at least one Morris factor must be enabled")
    if len({draft.variable_key for draft in enabled}) != len(enabled):
        raise ValueError("enabled Morris factor keys must be unique")
    documents: list[dict[str, object]] = []
    order = {key: index for index, key in enumerate(CANONICAL_MORRIS_FACTOR_KEYS)}
    for draft in sorted(
        enabled, key=lambda item: order.get(item.variable_key, len(order))
    ):
        if draft.variable_key not in CANONICAL_MORRIS_FACTOR_KEYS:
            raise ValueError("factor variable_key is unsupported")
        if draft.variable_key == _TEE_KEY and (
            config.ball_setup.support_mode is not BallSupportMode.TEE
        ):
            raise ValueError("tee_height_m factor requires tee support")
        if draft.lower >= draft.upper:
            raise ValueError("factor bounds must satisfy lower < upper")
        documents.append(
            {
                "spec_id": spec_id_for_key(draft.variable_key),
                "variable_key": draft.variable_key,
                "lower": draft.lower,
                "upper": draft.upper,
                "unit": registry[draft.variable_key].unit,
            }
        )
    return documents


def build_morris_request(
    config: SimulationConfig,
    drafts: tuple[MorrisFactorDraft, ...],
    *,
    request_id: str,
    trajectories: int = 12,
    levels: int = 4,
    seed: int = 0,
    minimum_effects: int = 4,
    worker_count: int = 1,
) -> MorrisAuthorityRequest:
    """Build and canonically validate an exact request without losing semantics."""
    from rate_of_closure.simulation.records import SimulationConfig

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if not isinstance(drafts, tuple) or not all(
        isinstance(item, MorrisFactorDraft) for item in drafts
    ):
        raise TypeError("drafts must be a tuple of MorrisFactorDraft values")
    document = {
        "schema_id": MORRIS_REQUEST_SCHEMA_ID,
        "schema_version": MORRIS_AUTHORITY_SCHEMA_VERSION,
        "request_id": request_id,
        "base": _base_document(config),
        "factors": _factor_documents(config, drafts),
        "trajectories": trajectories,
        "levels": levels,
        "seed": seed,
        "minimum_effects": minimum_effects,
        "worker_count": worker_count,
    }
    request = parse_morris_request(document)
    if request.base_config() != config:
        raise ValueError("config differs from pinned authority semantics")
    return request


__all__ = [
    "CANONICAL_MORRIS_FACTOR_KEYS",
    "MorrisFactorDraft",
    "build_morris_request",
    "spec_id_for_key",
    "suggested_factor_drafts",
]
