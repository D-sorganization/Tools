"""Canonical identity and request/result binding for paired attribution."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import cast

import numpy as np

from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.variation.ensemble_request_identity import (
    config_identity_sha256,
)
from rate_of_closure.variation.localized_attribution import AttributionTarget
from rate_of_closure.variation.simulation_types import (
    SimulationEnsembleRequest,
    SimulationEnsembleResult,
)
from shared.python.contracts import require
from shared.python.swing_sim.variation import VariationPlan


def finite_value(value: object, label: str) -> float:
    """Return one strict finite real while rejecting bool/complex coercion."""
    require(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
        f"{label} must be a real number excluding booleans",
        value,
    )
    result = float(cast(float, value))
    require(math.isfinite(result), f"{label} must be finite", result)
    return result


def stable_id(value: object, label: str) -> str:
    """Return one bounded trimmed control-free stable identifier."""
    require(isinstance(value, str), f"{label} must be a string", value)
    result = cast(str, value)
    require(
        bool(result)
        and result == result.strip()
        and len(result) <= 256
        and not any(ord(char) < 32 for char in result),
        f"{label} must be a stable ID",
        value,
    )
    return result


def canonical_design_identity(
    design_id: str,
    base_config: SimulationConfig,
    source_plan: VariationPlan,
    targets: tuple[AttributionTarget, ...],
    intervention_deltas_nm: Mapping[str, float],
    request_identity: str,
) -> str:
    """Hash every ordered design semantic plus its exact execution request."""
    payload = {
        "schema": "rate-of-closure/localized-attribution-design@1",
        "design_id": design_id,
        "base_config": config_identity_sha256(base_config),
        "source_plan": source_plan.to_json_dict(),
        "targets": [target.__dict__ for target in targets],
        "intervention_deltas_nm": dict(sorted(intervention_deltas_nm.items())),
        "request_identity": request_identity,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_result_matches_request(
    result: SimulationEnsembleResult, request: SimulationEnsembleRequest
) -> None:
    """Bind the retained result to the exact plan and explicit design rows."""
    require(
        result.variation.plan.to_json_dict() == request.plan.to_json_dict(),
        "result plan must match the retained request",
    )
    require(
        np.array_equal(result.variation.inputs, request.sampled_inputs),
        "result inputs must match the retained request",
    )


__all__ = [
    "canonical_design_identity",
    "finite_value",
    "require_result_matches_request",
    "stable_id",
]
