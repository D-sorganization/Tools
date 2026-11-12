"""In-memory CalculatorSpec registry."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from .catalog import CATALOG
from .spec import CalculatorSpec, FieldMetadata

_BRACKET_UNITS_RE = re.compile(r"\[([^\]]+)\]\s*$")


@dataclass(frozen=True)
class RegisteredCalculator:
    """Spec bound to a concrete REST POST path."""

    spec: CalculatorSpec
    http_path: str


CALCULATOR_REGISTRY: dict[str, RegisteredCalculator] = {}

_OPENAPI_SCHEMA_CACHE: dict[str, dict[str, dict[str, Any] | None]] = {}


def _join_path(prefix: str, subpath: str) -> str:
    base = prefix.rstrip("/") or ""
    suffix = subpath.strip("/")
    if not suffix:
        return base or "/"
    return f"{base}/{suffix}" if base else f"/{suffix}"


def _field_metadata(
    model: type[BaseModel], units: dict[str, str]
) -> tuple[FieldMetadata, ...]:
    schema = model.model_json_schema()
    props = schema.get("properties", {})
    items: list[FieldMetadata] = []
    for name, prop in props.items():
        valid_range = {
            key: prop[key]
            for key in (
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "minLength",
                "maxLength",
                "pattern",
            )
            if key in prop
        }
        description = prop.get("description")
        unit = units.get(name)
        if unit is None and isinstance(description, str):
            match = _BRACKET_UNITS_RE.search(description.strip())
            if match:
                unit = match.group(1)
        items.append(
            FieldMetadata(
                name=name,
                units=unit,
                valid_range=valid_range,
                description=description,
            )
        )
    return tuple(items)


def bind_calculator_route(
    spec: CalculatorSpec,
    *,
    router: APIRouter,
) -> None:
    """Register a calculator and assert its POST route exists on ``router``."""
    path = _join_path(str(getattr(router, "prefix", "")), spec.route_subpath)
    if spec.id in CALCULATOR_REGISTRY:
        raise ValueError(f"duplicate calculator id: {spec.id!r}")

    prefix = str(getattr(router, "prefix", "") or "")
    found = False
    for route in router.routes:
        methods = getattr(route, "methods", None) or set()
        raw_path = str(getattr(route, "path", "") or "")
        if raw_path.startswith("/api/calc/"):
            full = raw_path
        elif raw_path in ("", "/"):
            full = _join_path(prefix, "")
        elif raw_path.startswith("/"):
            full = _join_path(prefix, raw_path.lstrip("/"))
        else:
            full = _join_path(prefix, raw_path)
        if "POST" not in methods or full != path:
            continue
        endpoint = getattr(route, "endpoint", None)
        endpoint_name = getattr(endpoint, "__name__", None)
        if endpoint is spec.fn or endpoint_name == spec.fn.__name__:
            found = True
            break
    if not found:
        observed = [
            {
                "methods": sorted(getattr(route, "methods", None) or []),
                "path": getattr(route, "path", None),
                "endpoint": getattr(getattr(route, "endpoint", None), "__name__", None),
            }
            for route in router.routes
        ]
        raise ValueError(
            f"{spec.id}: handler {spec.fn.__name__!r} is not bound to POST {path!r}; "
            f"router prefix={getattr(router, 'prefix', None)!r}, routes={observed!r}"
        )

    CALCULATOR_REGISTRY[spec.id] = RegisteredCalculator(spec=spec, http_path=path)


def registered_calculators() -> Iterator[CalculatorSpec]:
    """Yield registered specs in stable id order."""
    for key in sorted(CALCULATOR_REGISTRY):
        yield CALCULATOR_REGISTRY[key].spec


def describe_calculator(calculator_id: str) -> dict[str, Any]:
    """Return units, valid ranges, and worked example for one calculator."""
    entry = CALCULATOR_REGISTRY[calculator_id]
    spec = entry.spec
    fields = _field_metadata(spec.request_model, dict(spec.units))
    cached = _OPENAPI_SCHEMA_CACHE.get(calculator_id, {})
    request_schema = cached.get("request") or spec.request_model.model_json_schema()
    response_schema = cached.get("response")
    if response_schema is None and spec.response_model is not None:
        response_schema = spec.response_model.model_json_schema()
    payload: dict[str, Any] = {
        "id": spec.id,
        "title": spec.title,
        "category": spec.category,
        "maturity": spec.maturity,
        "formula_ids": list(spec.formula_ids),
        "http_path": entry.http_path,
        "request_fields": [
            {
                "name": f.name,
                "units": f.units,
                "valid_range": f.valid_range,
                "description": f.description,
            }
            for f in fields
        ],
        "worked_example": {
            "request": spec.worked_example.request,
            "response_summary": spec.worked_example.response_summary,
        },
        "request_schema": request_schema,
    }
    if response_schema is not None:
        payload["response_schema"] = response_schema
    return payload


def materialize_openapi_schemas(openapi: dict[str, Any]) -> None:
    """Cache request/response JSON Schemas from a generated OpenAPI document."""
    schemas = openapi.get("components", {}).get("schemas", {})
    for calculator_id, entry in CALCULATOR_REGISTRY.items():
        spec = entry.spec
        req_name = spec.request_model.__name__
        request = schemas.get(req_name)
        response = None
        if spec.response_model is not None:
            response = schemas.get(spec.response_model.__name__)
        _OPENAPI_SCHEMA_CACHE[calculator_id] = {
            "request": request,
            "response": response,
        }


def http_path_for(calculator_id: str) -> str:
    """Return the POST path for a registered calculator."""
    return CALCULATOR_REGISTRY[calculator_id].http_path


def register_from_catalog(
    catalog_id: str,
    *,
    router: APIRouter,
    fn: Any,
) -> None:
    """Build a :class:`CalculatorSpec` from :mod:`catalog` and register it."""
    if catalog_id not in CATALOG:
        raise KeyError(f"unknown catalog id: {catalog_id!r}")
    meta = CATALOG[catalog_id]
    spec = CalculatorSpec(fn=fn, **meta)
    bind_calculator_route(spec, router=router)
