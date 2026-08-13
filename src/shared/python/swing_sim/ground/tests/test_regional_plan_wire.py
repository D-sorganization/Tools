"""Strict cross-runtime wire contracts for coplanar regional material plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shared.python.swing_sim.ground import (
    GroundProvenance,
    GroundRegionalMaterialPlanRequest,
    GroundRegionalMaterialPlanResult,
    build_regional_material_plan_result,
    regional_material_plan_request_from_json,
    regional_material_plan_result_from_json,
    regional_plan_to_surface_resolver,
)

FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "ground_regional_plan_golden_v1.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_shared_fixture_round_trips_with_exact_digests() -> None:
    fixture = _fixture()
    request = GroundRegionalMaterialPlanRequest.from_dict(fixture["request"])
    result = GroundRegionalMaterialPlanResult.from_dict(fixture["result"])

    assert request.to_dict() == fixture["request"]
    assert result.to_dict() == fixture["result"]
    assert _digest(request.to_json()) == fixture["request_sha256"]
    assert _digest(result.to_json()) == fixture["result_sha256"]


def test_builder_preserves_material_evidence_and_binds_runtime_resolver() -> None:
    fixture = _fixture()
    request = GroundRegionalMaterialPlanRequest.from_dict(fixture["request"])
    producer = GroundProvenance(
        "tools-ground-plan-validator",
        "1.0.0",
        "fixture-revision",
        fixture["request_sha256"],
    )

    result = build_regional_material_plan_result(request, producer)
    resolver = regional_plan_to_surface_resolver(request)

    assert result.request_sha256 == fixture["request_sha256"]
    assert result.base_surface == request.base_surface
    assert result.ordered_regions == tuple(
        sorted(
            request.regions, key=lambda region: (-region.precedence, region.region_id)
        )
    )
    assert resolver.surface == request.base_surface
    assert tuple(region.region_id for region in resolver.regions) == tuple(
        region.region_id for region in request.regions
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: {**data, "unexpected": True}, "fields"),
        (lambda data: {**data, "schema_version": "ground-plan/v2"}, "schema"),
        (lambda data: {**data, "regions": []}, "at least one"),
        (
            lambda data: {
                **data,
                "regions": [data["regions"][0], data["regions"][0]],
            },
            "region_id values must be unique",
        ),
        (
            lambda data: {
                **data,
                "regions": [
                    data["regions"][0],
                    {
                        **data["regions"][1],
                        "precedence": data["regions"][0]["precedence"],
                    },
                ],
            },
            "precedence values must be unique",
        ),
    ],
)
def test_request_parser_fails_closed_on_schema_and_identity_errors(
    mutation: object,
    message: str,
) -> None:
    request = _fixture()["request"]
    assert isinstance(request, dict)
    changed = mutation(request)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        GroundRegionalMaterialPlanRequest.from_dict(changed)


def test_request_rejects_non_coplanar_regions_and_out_of_domain_bounds() -> None:
    request = _fixture()["request"]
    assert isinstance(request, dict)
    regions = request["regions"]
    assert isinstance(regions, list)

    tilted = json.loads(json.dumps(request))
    tilted["regions"][0]["surface"]["normal_unit"] = [0.0, 0.8, 0.6]
    with pytest.raises(ValueError, match="coplanar static geometry"):
        GroundRegionalMaterialPlanRequest.from_dict(tilted)

    outside = json.loads(json.dumps(request))
    outside["regions"][0]["upper_coordinate_m"] = 1_001.0
    with pytest.raises(ValueError, match="inside the base domain"):
        GroundRegionalMaterialPlanRequest.from_dict(outside)


def test_request_rejects_nonstatic_duplicate_and_invalid_interval_evidence() -> None:
    request = _fixture()["request"]
    assert isinstance(request, dict)

    moving = json.loads(json.dumps(request))
    moving["base_surface"]["surface_velocity_m_s"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="static surfaces"):
        GroundRegionalMaterialPlanRequest.from_dict(moving)

    duplicate_surface = json.loads(json.dumps(request))
    duplicate_surface["regions"][1]["surface"]["surface_id"] = duplicate_surface[
        "regions"
    ][0]["surface"]["surface_id"]
    with pytest.raises(ValueError, match="surface_id values must be unique"):
        GroundRegionalMaterialPlanRequest.from_dict(duplicate_surface)

    empty_interval = json.loads(json.dumps(request))
    empty_interval["regions"][0]["upper_coordinate_m"] = empty_interval["regions"][0][
        "lower_coordinate_m"
    ]
    with pytest.raises(ValueError, match="below upper_coordinate_m"):
        GroundRegionalMaterialPlanRequest.from_dict(empty_interval)


def test_json_entrypoints_reject_duplicate_keys_and_oversized_documents() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        regional_material_plan_request_from_json(
            '{"schema_version":"ground-regional-material-plan-request/v1",'
            '"schema_version":"ground-regional-material-plan-request/v1"}'
        )
    with pytest.raises(ValueError, match="maximum wire size"):
        regional_material_plan_request_from_json("{" + " " * 1_048_577 + "}")


def test_result_rejects_fabricated_or_reordered_material_data() -> None:
    result = _fixture()["result"]
    assert isinstance(result, dict)
    changed_surface = json.loads(json.dumps(result))
    changed_surface["ordered_regions"][0]["surface"]["surface_id"] = "invented"
    with pytest.raises(ValueError, match="surface identity"):
        GroundRegionalMaterialPlanResult.from_dict(changed_surface)

    reordered = json.loads(json.dumps(result))
    reordered["ordered_regions"].reverse()
    with pytest.raises(ValueError, match="canonical precedence order"):
        GroundRegionalMaterialPlanResult.from_dict(reordered)

    wrong_digest = json.loads(json.dumps(result))
    wrong_digest["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="request_sha256"):
        GroundRegionalMaterialPlanResult.from_dict(wrong_digest)

    text = json.dumps(result, separators=(",", ":"))
    assert regional_material_plan_result_from_json(text).to_dict() == result
