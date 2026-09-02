"""Strict comparison-aware ground playback workspace-v2 contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rate_of_closure.simulation.ground_playback import load_ground_result_json
from rate_of_closure.simulation.ground_playback_workspace import (
    GROUND_PLAYBACK_WORKSPACE_SCHEMA,
    GroundPlaybackState,
    GroundPlaybackViewState,
    GroundPlaybackWorkspace,
    ground_workspace_from_json,
    ground_workspace_to_json,
)
from rate_of_closure.simulation.ground_playback_workspace_v2 import (
    GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
    GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
    GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
    GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2,
    GroundPlaybackComparisonState,
    GroundPlaybackWorkspaceV2,
    ground_workspace_from_versioned_json,
    ground_workspace_v2_from_json,
    ground_workspace_v2_to_json,
    load_ground_workspace_versioned_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE_DIRECTORY = (
    Path(__file__).parents[2] / "src/rate_of_closure/web/src/model/__fixtures__"
)
RESULT_FIXTURE = FIXTURE_DIRECTORY / "ground_reference_pipeline_golden_v1.json"
WORKSPACE_FIXTURE = FIXTURE_DIRECTORY / "ground_playback_workspace_golden_v2.json"
WORKSPACE_SHA256 = "28b94af16a05315a9d1067bda894a3817dce5849a9562e1ffef7d0d8caecd654"


def _result_payload() -> dict[str, object]:
    payload = json.loads(RESULT_FIXTURE.read_text(encoding="utf-8"))
    return payload["result"]


def _comparison_payload(*, time_offset_s: float = 0.2) -> dict[str, object]:
    payload = _result_payload()
    payload["request_id"] = "comparison-run"
    payload["provenance"]["input_sha256"] = "b" * 64  # type: ignore[index]
    for point in payload["trajectory"]:  # type: ignore[union-attr]
        point["time_s"] += time_offset_s
    for event in payload["events"]:  # type: ignore[union-attr]
        event["time_s"] += time_offset_s
    payload["termination"]["time_s"] += time_offset_s  # type: ignore[index]
    return payload


def _workspace_v2(*, visible: bool = False) -> GroundPlaybackWorkspaceV2:
    primary = load_ground_result_json(json.dumps(_result_payload()))
    comparison = load_ground_result_json(json.dumps(_comparison_payload()))
    return GroundPlaybackWorkspaceV2(
        result=primary,
        comparison=GroundPlaybackComparisonState(comparison, visible),
        playback=GroundPlaybackState(
            time_s=1.60466094435,
            speed=2.0,
            loop=True,
        ),
        view=GroundPlaybackViewState(yaw_deg=-37.5, pitch_deg=18.0, zoom=1.75),
    )


def test_v2_round_trip_preserves_hidden_comparison_and_union_time() -> None:
    workspace = _workspace_v2(visible=False)

    encoded = ground_workspace_v2_to_json(workspace)
    restored = ground_workspace_v2_from_json(encoded)

    assert restored == workspace
    assert ground_workspace_v2_to_json(restored) == encoded
    payload = json.loads(encoded)
    assert payload["schema_version"] == GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2
    assert set(payload) == {
        "schema_version",
        "result",
        "comparison",
        "playback",
        "view",
    }
    assert payload["comparison"]["visible"] is False
    assert payload["playback"]["time_s"] > workspace.result.trajectory[-1].time_s


def test_v1_dispatch_migrates_one_way_without_changing_v1_apis() -> None:
    result = load_ground_result_json(json.dumps(_result_payload()))
    legacy = GroundPlaybackWorkspace(
        result=result,
        playback=GroundPlaybackState(time_s=1.205, speed=1.0, loop=False),
        view=GroundPlaybackViewState(yaw_deg=0.0, pitch_deg=22.0, zoom=1.0),
    )
    legacy_text = ground_workspace_to_json(legacy)

    migrated = ground_workspace_from_versioned_json(legacy_text)
    load = load_ground_workspace_versioned_json(legacy_text)

    assert legacy.schema_version == GROUND_PLAYBACK_WORKSPACE_SCHEMA
    assert json.loads(legacy_text)["schema_version"] == GROUND_PLAYBACK_WORKSPACE_SCHEMA
    assert migrated.schema_version == GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2
    assert migrated.result == legacy.result
    assert migrated.comparison is None
    assert migrated.playback == legacy.playback
    assert migrated.view == legacy.view
    assert load.workspace == migrated
    assert load.source_schema_version == GROUND_PLAYBACK_WORKSPACE_SCHEMA
    assert load.migrated_from_v1
    assert json.loads(ground_workspace_v2_to_json(migrated))["comparison"] is None
    with pytest.raises(ValueError, match="expected.*schema v2"):
        ground_workspace_v2_from_json(legacy_text)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(extra=True), "fields do not match v2"),
        (
            lambda payload: payload["comparison"].update(extra=True),
            "comparison fields do not match v2",
        ),
        (
            lambda payload: payload["comparison"].update(visible=1),
            "comparison visible must be a boolean",
        ),
        (
            lambda payload: payload["playback"].update(time_s=99.0),
            "union timeline",
        ),
    ],
)
def test_v2_rejects_unknown_nested_and_invalid_state(mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    payload = json.loads(ground_workspace_v2_to_json(_workspace_v2()))
    mutate(payload)

    with pytest.raises((TypeError, ValueError), match=message):
        ground_workspace_v2_from_json(json.dumps(payload))


def test_v2_rejects_duplicates_and_enforces_all_document_bounds() -> None:
    encoded = ground_workspace_v2_to_json(_workspace_v2())
    duplicate = encoded.replace(
        f'"schema_version":"{GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2}"',
        f'"schema_version":"{GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2}",'
        '"schema_version":"duplicate"',
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        ground_workspace_v2_from_json(duplicate)
    with pytest.raises(ValueError, match="size limit"):
        ground_workspace_v2_from_json(encoded, max_bytes=10)
    with pytest.raises(ValueError, match="size limit"):
        ground_workspace_v2_to_json(_workspace_v2(), max_bytes=10)
    with pytest.raises(ValueError, match="per-result point limit"):
        ground_workspace_v2_from_json(encoded, max_points_per_result=1)
    with pytest.raises(ValueError, match="combined point limit"):
        ground_workspace_v2_from_json(
            encoded,
            max_points_per_result=100,
            max_combined_points=3,
        )
    assert GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2 == 11 * 1024 * 1024


def test_v1_and_v2_normalize_oversized_numeric_tokens_to_value_errors() -> None:
    oversized = 10**400
    legacy = GroundPlaybackWorkspace(
        result=load_ground_result_json(json.dumps(_result_payload())),
        playback=GroundPlaybackState(time_s=1.205, speed=1.0, loop=False),
        view=GroundPlaybackViewState(yaw_deg=0.0, pitch_deg=22.0, zoom=1.0),
    )
    legacy_payload = json.loads(ground_workspace_to_json(legacy))
    legacy_payload["playback"]["time_s"] = oversized
    v2_payload = json.loads(ground_workspace_v2_to_json(_workspace_v2()))
    v2_payload["playback"]["time_s"] = oversized

    with pytest.raises(ValueError, match="time_s must be finite"):
        ground_workspace_from_json(json.dumps(legacy_payload))
    with pytest.raises(ValueError, match="time_s must be finite"):
        ground_workspace_v2_from_json(json.dumps(v2_payload))


def test_v2_public_limit_overrides_cannot_raise_hard_contract_caps() -> None:
    encoded = ground_workspace_v2_to_json(_workspace_v2())

    with pytest.raises(ValueError, match="max_bytes.*hard cap"):
        ground_workspace_v2_from_json(
            encoded,
            max_bytes=GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2 + 1,
        )
    with pytest.raises(ValueError, match="max_points_per_result.*hard cap"):
        ground_workspace_v2_from_json(
            encoded,
            max_points_per_result=GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT + 1,
        )
    with pytest.raises(ValueError, match="max_combined_points.*hard cap"):
        ground_workspace_v2_from_json(
            encoded,
            max_combined_points=GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED + 1,
        )


def test_v2_shared_golden_is_byte_identical_and_sha_pinned() -> None:
    golden = WORKSPACE_FIXTURE.read_text(encoding="utf-8")

    assert ground_workspace_v2_to_json(ground_workspace_v2_from_json(golden)) == golden
    assert hashlib.sha256(golden.encode("utf-8")).hexdigest() == WORKSPACE_SHA256
    load = load_ground_workspace_versioned_json(golden)
    assert load.source_schema_version == GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2
    assert not load.migrated_from_v1


def test_supported_speeds_match_the_shared_playback_transport() -> None:
    """The workspace wire's speed whitelist must equal the transport's speeds.

    ``SUPPORTED_PLAYBACK_SPEEDS`` deliberately stays its own constant: it
    validates a versioned, fail-closed persisted document, so what it accepts
    is a wire contract that must not silently follow a runtime refactor.
    ``PLAYBACK_SPEEDS`` is what every playback surface (Qt
    ``PlaybackTransportControls``, React ``PlaybackTransportBar``) actually
    offers. Today the two are equal by coincidence; this gate is what makes
    the equality a fact. If it fails, either the transport grew a speed the
    wire must learn to accept (a workspace wire-version question - raise it
    on the workspace's issue, do not just widen the whitelist) or the wire
    accepts a speed no player offers.
    """
    from rate_of_closure.simulation.ground_playback_workspace import (
        SUPPORTED_PLAYBACK_SPEEDS,
    )
    from rate_of_closure.simulation.playback_transport import PLAYBACK_SPEEDS

    assert tuple(SUPPORTED_PLAYBACK_SPEEDS) == tuple(PLAYBACK_SPEEDS)
