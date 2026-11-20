"""CalculatorSpec registry contract tests (issue #6020)."""

from __future__ import annotations

import inspect
from typing import Any, get_args, get_origin

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("calc_backend")

from calc_backend.app import app
from calc_backend.registry.catalog import CATALOG
from calc_backend.registry.store import (
    CALCULATOR_REGISTRY,
    describe_calculator,
    http_path_for,
    registered_calculators,
)

_BOOL_LIKE_DEFAULT_TRUE = (True, "true", "True")


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    name = ref.rsplit("/", 1)[-1]
    return root["components"]["schemas"][name]


def _response_models() -> list[type]:
    from pydantic import BaseModel

    models: list[type] = []
    for spec in registered_calculators():
        model = spec.response_model
        if model is not None and issubclass(model, BaseModel):
            models.append(model)
    return models


def test_registry_covers_catalog() -> None:
    assert set(CALCULATOR_REGISTRY) == set(CATALOG)


def test_describe_returns_units_ranges_and_example() -> None:
    spec = next(registered_calculators())
    payload = describe_calculator(spec.id)
    assert payload["worked_example"]["request"]
    assert payload["request_fields"]
    assert any(field.get("units") for field in payload["request_fields"])


def test_openapi_request_schema_matches_registry() -> None:
    app.openapi_schema = None
    openapi = app.openapi()
    for spec in registered_calculators():
        path_item = openapi["paths"][http_path_for(spec.id)]
        post = path_item["post"]
        body = post["requestBody"]["content"]["application/json"]["schema"]
        openapi_schema = _resolve_ref(body, openapi)
        registry_schema = describe_calculator(spec.id)["request_schema"]
        assert openapi_schema == registry_schema, spec.id


def test_openapi_response_schema_matches_registry_when_typed() -> None:
    app.openapi_schema = None
    openapi = app.openapi()
    for spec in registered_calculators():
        if spec.response_model is None:
            continue
        path_item = openapi["paths"][http_path_for(spec.id)]
        post = path_item["post"]
        success = post["responses"]["200"]["content"]["application/json"]["schema"]
        if "$ref" not in success:
            continue
        openapi_schema = _resolve_ref(success, openapi)
        registry_schema = describe_calculator(spec.id).get("response_schema")
        assert registry_schema is not None
        assert openapi_schema == registry_schema, spec.id


def test_result_models_avoid_default_true_success_flags() -> None:
    for model in _response_models():
        for name, field in model.model_fields.items():
            if name not in {"success", "converged", "ok"}:
                continue
            default = field.default
            if default in _BOOL_LIKE_DEFAULT_TRUE:
                pytest.fail(f"{model.__name__}.{name} defaults to true")
            annotation = field.annotation
            origin = get_origin(annotation)
            if origin is not None:
                args = get_args(annotation)
                if any(arg is bool for arg in args) and default is True:
                    pytest.fail(f"{model.__name__}.{name} defaults to true")


def test_catalog_specs_reference_callable_handlers() -> None:
    for entry in CALCULATOR_REGISTRY.values():
        assert callable(entry.spec.fn)
        assert inspect.isfunction(entry.spec.fn)


def test_all_worked_examples_validate_against_request_models() -> None:
    for spec in registered_calculators():
        example = spec.worked_example.request
        assert example, f"{spec.id}: worked_example.request must be non-empty"
        unknown = set(example) - set(spec.request_model.model_fields)
        assert not unknown, (
            f"{spec.id}: worked_example.request has unknown fields {sorted(unknown)}"
        )
        spec.request_model.model_validate(example)


def test_all_catalog_units_reference_real_request_fields() -> None:
    for spec in registered_calculators():
        unknown = set(spec.units) - set(spec.request_model.model_fields)
        assert not unknown, (
            f"{spec.id}: units references unknown fields {sorted(unknown)}"
        )


def test_registry_covers_every_calculator_post_route() -> None:
    app.openapi_schema = None
    openapi = app.openapi()
    post_paths = {
        path
        for path, item in openapi["paths"].items()
        if path.startswith("/api/calc/") and "post" in item
    }
    registered_paths = {http_path_for(spec.id) for spec in registered_calculators()}
    assert registered_paths == post_paths
