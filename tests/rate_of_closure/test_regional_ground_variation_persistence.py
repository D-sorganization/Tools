"""Strict persistence contracts for seeded regional-ground variation requests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.application.regional_ground_variation_request import (
    MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES,
    REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA,
    read_regional_ground_variation_request,
    regional_ground_variation_request_from_json,
    regional_ground_variation_request_to_json,
    write_regional_ground_variation_request_atomic,
)
from rate_of_closure.application.regional_surface_plan import (
    illustrative_regional_surface_plan_draft,
    validate_regional_surface_plan_draft,
)
from rate_of_closure.variation.regional_ground_variation import (
    GROUND_NORMAL_RESTITUTION_KEY,
    GROUND_ROLLING_RESISTANCE_KEY,
    GroundRegionalVariationRequest,
    register_ground_variation_variables,
)
from shared.python.swing_sim.flight.tests._regional_ground_pipeline_support import (
    _plan,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan
from shared.python.swing_sim.variation import registry as variation_registry

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(autouse=True)
def _isolated_ground_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the Rate-owned extension from leaking into shared registry tests."""
    monkeypatch.setattr(
        variation_registry, "_REGISTRY", dict(variation_registry.variable_registry())
    )
    register_ground_variation_variables()


def _request() -> GroundRegionalVariationRequest:
    plan = VariationPlan(
        mode="launch",
        base_variables={
            GROUND_NORMAL_RESTITUTION_KEY: 0.4,
            GROUND_ROLLING_RESISTANCE_KEY: 0.04,
        },
        noise=(
            NoiseSpec(
                GROUND_ROLLING_RESISTANCE_KEY,
                distribution="uniform",
                scale=0.02,
                lower=0.02,
                upper=0.08,
                spec_id="ground-rolling-resistance",
            ),
        ),
        n_runs=4,
        seed=1729,
    )
    return GroundRegionalVariationRequest(
        plan,
        _plan(),
        "seeded-ground-study",
        "pytest/exact-parent-27d2a68d",
        8,
        "driver",
    )


def _payload() -> dict[str, object]:
    return json.loads(regional_ground_variation_request_to_json(_request()))


def _editor_request() -> GroundRegionalVariationRequest:
    request = _request()
    regional = validate_regional_surface_plan_draft(
        illustrative_regional_surface_plan_draft()
    )
    plan = replace(
        request.plan,
        base_variables={
            GROUND_NORMAL_RESTITUTION_KEY: regional.base_surface.normal_restitution,
            GROUND_ROLLING_RESISTANCE_KEY: regional.base_surface.rolling_resistance,
        },
    )
    return replace(request, plan=plan, regional_plan=regional)


def test_canonical_round_trip_is_exact_deterministic_and_composed() -> None:
    request = _request()

    first = regional_ground_variation_request_to_json(request)
    second = regional_ground_variation_request_to_json(request)
    payload = json.loads(first)

    assert first == second
    assert "\n" not in first
    assert regional_ground_variation_request_from_json(first) == request
    assert payload["schema"] == REGIONAL_GROUND_VARIATION_REQUEST_SCHEMA
    assert payload["variation_plan"] == request.plan.to_json_dict()
    assert payload["regional_plan"] == request.regional_plan.to_dict()


def test_python_serializer_matches_the_react_golden_bytes() -> None:
    fixture = (
        Path(__file__).parents[2]
        / "src"
        / "rate_of_closure"
        / "web"
        / "src"
        / "model"
        / "__fixtures__"
        / "regional_ground_variation_request_golden_v1.json"
    )

    assert fixture.read_text(encoding="utf-8").removesuffix("\n") == (
        regional_ground_variation_request_to_json(_editor_request())
    )


def test_native_file_round_trip_writes_exact_canonical_bytes(tmp_path: Path) -> None:
    request = _request()
    target = tmp_path / "ground-variation-request.json"

    assert write_regional_ground_variation_request_atomic(request, target)

    assert target.read_bytes() == regional_ground_variation_request_to_json(
        request
    ).encode("utf-8")
    assert read_regional_ground_variation_request(target) == request


def test_cancelled_write_is_a_no_op(tmp_path: Path) -> None:
    assert write_regional_ground_variation_request_atomic(_request(), None) is False
    assert list(tmp_path.iterdir()) == []


def test_replace_failure_preserves_last_known_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rate_of_closure.application import atomic_text_files

    target = tmp_path / "ground-variation-request.json"
    target.write_text("last-known-good", encoding="utf-8")
    monkeypatch.setattr(
        atomic_text_files.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        write_regional_ground_variation_request_atomic(_request(), target)

    assert target.read_text(encoding="utf-8") == "last-known-good"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda value: value.update(schema="unsupported/v2"), "schema"),
        (lambda value: value.update(schema_version=1), "schema_version"),
        (lambda value: value.update(unknown=True), "fields mismatch"),
        (
            lambda value: value["variation_plan"].update(schema_version=1),
            "variation_plan schema_version",
        ),
        (
            lambda value: value["variation_plan"].update(unknown=True),
            "variation_plan fields mismatch",
        ),
        (
            lambda value: value["regional_plan"].update(unknown=True),
            "regional material plan request fields",
        ),
        (
            lambda value: value["variation_plan"].update(seed=1730),
            "plan digest mismatch",
        ),
    ],
    ids=(
        "schema",
        "schema-version",
        "outer-extra",
        "variation-version",
        "variation-extra",
        "regional-extra",
        "plan-substitution",
    ),
)
def test_schema_versions_and_fields_fail_closed(mutator, message: str) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises((TypeError, ValueError), match=message):
        regional_ground_variation_request_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda value: value.update(max_rows=True), "max_rows"),
        (lambda value: value.update(result_id=""), "result_id"),
        (lambda value: value.update(series_id=7), "series_id"),
        (
            lambda value: value["variation_plan"].update(n_runs=True),
            "n_runs",
        ),
        (
            lambda value: value["variation_plan"]["noise"][0].update(scale=True),
            "noise.*scale",
        ),
        (
            lambda value: value["variation_plan"]["base_variables"].update(
                {GROUND_ROLLING_RESISTANCE_KEY: True}
            ),
            "base_variables",
        ),
    ],
    ids=("bool-cap", "blank-id", "series-type", "bool-runs", "bool-scale", "bool-base"),
)
def test_identifier_cap_and_numeric_types_fail_closed(mutator, message: str) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises((TypeError, ValueError), match=message):
        regional_ground_variation_request_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "text, message",
    [
        ('{"schema":"one","schema":"two"}', "duplicate"),
        ('{"value":NaN}', "finite"),
        ('{"value":"\\ud800"}', "surrogate"),
    ],
    ids=("duplicate", "nonfinite", "surrogate"),
)
def test_json_safety_failures_are_rejected(text: str, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        regional_ground_variation_request_from_json(text)


def test_unsafe_cross_runtime_integer_is_rejected() -> None:
    payload = _payload()
    payload["max_rows"] = 9_007_199_254_740_992

    with pytest.raises(ValueError, match="safe range"):
        regional_ground_variation_request_from_json(json.dumps(payload))


def test_utf8_wire_bound_and_invalid_file_encoding_fail_closed(tmp_path: Path) -> None:
    oversized = "é" * (MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES // 2 + 1)
    with pytest.raises(ValueError, match="maximum wire size"):
        regional_ground_variation_request_from_json(oversized)

    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        read_regional_ground_variation_request(invalid)


def test_serializer_and_file_reader_require_exact_valid_inputs(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="exact GroundRegionalVariationRequest"):
        regional_ground_variation_request_to_json(object())  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_regional_ground_variation_request(tmp_path / "missing.json")
