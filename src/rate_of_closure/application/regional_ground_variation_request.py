"""Canonical bounded persistence for seeded regional-ground variation requests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rate_of_closure.variation.regional_ground_variation import (
    GroundRegionalVariationRequest,
    register_ground_variation_variables,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground.regional_plan_wire import (
    regional_material_plan_request_from_dict,
)
from shared.python.swing_sim.ground.strict_json import strict_json_object
from shared.python.swing_sim.variation.execution_metadata import plan_sha256
from shared.python.swing_sim.variation.spec import SCHEMA_VERSION, VariationPlan

from ._workspace_validation import exact_mapping
from .atomic_text_files import write_utf8_text_atomic
from .bounded_text_files import read_bounded_utf8

REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA = (
    "rate-of-closure/regional-ground-variation-request/v2"
)
REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA_VERSION = 2
MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES = 1_048_576

_ROOT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "variation_plan",
        "variation_plan_sha256",
        "regional_plan",
        "result_id",
        "source_provenance",
        "max_rows",
        "series_id",
    }
)
_VARIATION_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "base_variables",
        "noise",
        "n_runs",
        "seed",
        "flight_model",
        "groups",
    }
)
_NOISE_FIELDS = frozenset(
    {
        "variable_key",
        "distribution",
        "scale",
        "lower",
        "upper",
        "spec_id",
        "time_window_s",
        "point_ids",
    }
)
_GROUP_FIELDS = frozenset({"group_id", "spec_ids", "matrix_kind", "matrix"})


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _text(value: object, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    return value


def _number(value: object, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a JSON number")
    if not math.isfinite(cast(int | float, value)):
        raise ValueError(f"{name} must be finite")


def _integer(value: object, name: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _noise_shape(value: object, index: int) -> None:
    name = f"variation_plan noise[{index}]"
    data = exact_mapping(value, _NOISE_FIELDS, name)
    for field in ("variable_key", "distribution", "spec_id"):
        _text(data[field], f"{name} {field}")
    _number(data["scale"], f"{name} scale")
    _number(data["lower"], f"{name} lower", nullable=True)
    _number(data["upper"], f"{name} upper", nullable=True)
    window = data["time_window_s"]
    if window is not None:
        values = _sequence(window, f"{name} time_window_s")
        if len(values) != 2:
            raise ValueError(f"{name} time_window_s must contain two numbers")
        for item in values:
            _number(item, f"{name} time_window_s")
    for point in _sequence(data["point_ids"], f"{name} point_ids"):
        _text(point, f"{name} point_id")


def _group_shape(value: object, index: int) -> None:
    name = f"variation_plan groups[{index}]"
    data = exact_mapping(value, _GROUP_FIELDS, name)
    _text(data["group_id"], f"{name} group_id")
    _text(data["matrix_kind"], f"{name} matrix_kind")
    for spec_id in _sequence(data["spec_ids"], f"{name} spec_ids"):
        _text(spec_id, f"{name} spec_id")
    for row in _sequence(data["matrix"], f"{name} matrix"):
        for item in _sequence(row, f"{name} matrix row"):
            _number(item, f"{name} matrix value")


def _base_shape(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("variation_plan base_variables must be a JSON object")
    for key, item in value.items():
        _text(key, "variation_plan base_variables key")
        _number(item, f"variation_plan base_variables[{key!r}]")


def _variation_plan(value: object) -> VariationPlan:
    data = exact_mapping(value, _VARIATION_FIELDS, "variation_plan")
    version = _integer(data["schema_version"], "variation_plan schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported variation_plan schema_version {version!r}")
    _text(data["mode"], "variation_plan mode")
    _text(data["flight_model"], "variation_plan flight_model")
    _base_shape(data["base_variables"])
    _integer(data["n_runs"], "variation_plan n_runs")
    _integer(data["seed"], "variation_plan seed", nonnegative=True)
    for index, item in enumerate(_sequence(data["noise"], "variation_plan noise")):
        _noise_shape(item, index)
    for index, item in enumerate(_sequence(data["groups"], "variation_plan groups")):
        _group_shape(item, index)
    register_ground_variation_variables()
    return VariationPlan.from_json_dict(data)


def _request_payload(request: GroundRegionalVariationRequest) -> dict[str, Any]:
    if type(request) is not GroundRegionalVariationRequest:
        raise TypeError("request must be an exact GroundRegionalVariationRequest")
    return {
        "schema": REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA,
        "schema_version": REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA_VERSION,
        "variation_plan": request.plan.to_json_dict(),
        "variation_plan_sha256": plan_sha256(request.plan),
        "regional_plan": request.regional_plan.to_dict(),
        "result_id": request.result_id,
        "source_provenance": request.source_provenance,
        "max_rows": request.max_rows,
        "series_id": request.series_id,
    }


def _bounded_utf8(text: object) -> str:
    if type(text) is not str:
        raise TypeError("regional-ground variation request JSON must be text")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "regional-ground variation request must be valid UTF-8"
        ) from exc
    if len(encoded) > MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES:
        raise ValueError("regional-ground variation request exceeds maximum wire size")
    return text


def regional_ground_variation_request_to_json(
    request: GroundRegionalVariationRequest,
) -> str:
    """Return deterministic, browser-portable canonical request JSON."""
    text = str(canonical_numeric_json(_request_payload(request)))
    return _bounded_utf8(text)


def regional_ground_variation_request_from_json(
    text: str,
) -> GroundRegionalVariationRequest:
    """Parse one bounded exact request without executing any physics."""
    payload = strict_json_object(_bounded_utf8(text))
    canonical_numeric_json(payload)
    data = exact_mapping(payload, _ROOT_FIELDS, "regional-ground variation request")
    if data["schema"] != REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA:
        raise ValueError("unsupported regional-ground variation request schema")
    version = _integer(data["schema_version"], "schema_version")
    if version != REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}")
    result_id = cast(str, _text(data["result_id"], "result_id"))
    provenance = cast(str, _text(data["source_provenance"], "source_provenance"))
    series_id = _text(data["series_id"], "series_id", nullable=True)
    max_rows = _integer(data["max_rows"], "max_rows")
    variation_plan = _variation_plan(data["variation_plan"])
    if data["variation_plan_sha256"] != plan_sha256(variation_plan):
        raise ValueError("regional-ground variation plan digest mismatch")
    return GroundRegionalVariationRequest(
        variation_plan,
        regional_material_plan_request_from_dict(data["regional_plan"]),
        result_id,
        provenance,
        max_rows,
        series_id,
    )


def read_regional_ground_variation_request(
    source: str | Path,
) -> GroundRegionalVariationRequest:
    """Read one bounded UTF-8 snapshot and completely validate it."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(
            f"regional-ground variation request does not exist: {path}"
        )
    text = read_bounded_utf8(
        path,
        MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES,
        "regional-ground variation request",
    )
    return regional_ground_variation_request_from_json(text)


def write_regional_ground_variation_request_atomic(
    request: GroundRegionalVariationRequest,
    destination: str | Path | None,
) -> bool:
    """Atomically replace a native request file, or return false on cancel."""
    if destination is None:
        return False
    text = regional_ground_variation_request_to_json(request)
    write_succeeded: bool = write_utf8_text_atomic(
        text,
        destination,
        document_name="regional-ground variation request",
    )
    return write_succeeded


__all__ = [
    "MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES",
    "REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA",
    "REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA_VERSION",
    "read_regional_ground_variation_request",
    "regional_ground_variation_request_from_json",
    "regional_ground_variation_request_to_json",
    "write_regional_ground_variation_request_atomic",
]
