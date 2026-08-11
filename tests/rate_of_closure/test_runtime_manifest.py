from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rate_of_closure.runtime_manifest import (
    CalculationRuntimeManifest,
    create_runtime_manifest,
    runtime_manifest_from_json,
    stable_runtime_manifest_json,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "runtime_manifest_parity_v1.json"
)


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _manifest_payload() -> dict[str, Any]:
    return copy.deepcopy(_fixture()["manifest"])


def test_runtime_manifest_matches_shared_canonical_fixture() -> None:
    fixture = _fixture()
    manifest = CalculationRuntimeManifest.model_validate(fixture["manifest"])

    assert manifest.to_wire() == fixture["manifest"]
    assert stable_runtime_manifest_json(manifest) == fixture["expected_canonical_json"]
    assert runtime_manifest_from_json(manifest.to_json()) == manifest


def test_factory_requires_only_explicit_inputs_and_freezes_nested_records() -> None:
    parsed = CalculationRuntimeManifest.model_validate(_manifest_payload())
    rebuilt = create_runtime_manifest(
        surface_id=parsed.surface_id,
        build=parsed.build,
        calculations=parsed.calculations,
        provenance=parsed.provenance,
    )

    assert rebuilt == parsed
    with pytest.raises(ValidationError, match="frozen"):
        rebuilt.surface_id = parsed.surface_id  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        rebuilt.build.build_id = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("description", "mutate"),
    [
        ("unknown top-level field", lambda value: value.update(extra=True)),
        (
            "unknown nested field",
            lambda value: value["build"].update(extra=True),
        ),
        (
            "unsupported schema",
            lambda value: value.update(
                schema_version="calculation-runtime-manifest/v2"
            ),
        ),
        ("unknown surface", lambda value: value.update(surface_id="tools.cli")),
        (
            "non-SHA revision",
            lambda value: value["build"].update(tools_commit="working-tree"),
        ),
        (
            "placeholder build identity",
            lambda value: value["build"].update(build_id="todo"),
        ),
        (
            "duplicate calculation domain",
            lambda value: value["calculations"][2].update(domain="flight"),
        ),
        (
            "out-of-order calculation domains",
            lambda value: value["calculations"].reverse(),
        ),
        (
            "available calculation reason",
            lambda value: value["calculations"][0].update(reason="fallback"),
        ),
        (
            "available calculation missing authority",
            lambda value: value["calculations"][0].update(
                implementation_authority=None
            ),
        ),
        (
            "unavailable calculation leaks model identity",
            lambda value: value["calculations"][2].update(model_id="unqualified"),
        ),
        (
            "unavailable calculation omits reason",
            lambda value: value["calculations"][2].update(reason=None),
        ),
        (
            "placeholder unavailable reason",
            lambda value: value["calculations"][2].update(reason="Unknown"),
        ),
        (
            "duplicate numerical option",
            lambda value: value["calculations"][0]["numerical_options"].append(
                copy.deepcopy(value["calculations"][0]["numerical_options"][0])
            ),
        ),
        (
            "numeric option without unit",
            lambda value: value["calculations"][1]["numerical_options"][0].update(
                unit=None
            ),
        ),
        (
            "text option with unit",
            lambda value: value["calculations"][0]["numerical_options"][0].update(
                unit="1"
            ),
        ),
        (
            "duplicate evidence identity",
            lambda value: value["provenance"]["evidence_ids"].append("issue-4261"),
        ),
        (
            "surrogate provenance text",
            lambda value: value["provenance"].update(source_reference="fixture-\ud800"),
        ),
    ],
)
def test_runtime_manifest_rejects_adversarial_records(
    description: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    assert description
    payload = _manifest_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=".+"):
        CalculationRuntimeManifest.model_validate(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_runtime_manifest_rejects_nonfinite_options(value: float) -> None:
    payload = _manifest_payload()
    payload["calculations"][1]["numerical_options"][0]["value"] = value

    with pytest.raises(ValidationError, match="finite"):
        CalculationRuntimeManifest.model_validate(payload)


def test_runtime_manifest_rejects_unsafe_integer_and_duplicate_json_fields() -> None:
    payload = _manifest_payload()
    payload["calculations"][1]["numerical_options"][0].update(
        value=9_007_199_254_740_992,
        unit="1",
    )
    with pytest.raises(ValidationError, match="safe integer"):
        CalculationRuntimeManifest.model_validate(payload)

    duplicate = '{"schema_version":"first","schema_version":"second"}'
    with pytest.raises(ValueError, match="duplicate JSON field"):
        runtime_manifest_from_json(duplicate)
