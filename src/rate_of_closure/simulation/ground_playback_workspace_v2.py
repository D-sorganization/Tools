"""Strict comparison-aware persistence for ground playback workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from rate_of_closure.simulation.ground_playback import GroundPlaybackTimeline
from rate_of_closure.simulation.ground_playback_workspace import (
    GROUND_PLAYBACK_WORKSPACE_SCHEMA,
    GroundPlaybackState,
    GroundPlaybackViewState,
    GroundPlaybackWorkspace,
    ground_workspace_from_json,
)
from rate_of_closure.simulation.ground_playback_workspace_common import (
    exact_workspace_fields,
    workspace_object,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground import GroundSimulationResult
from shared.python.swing_sim.ground.contract_wire import record_from_dict
from shared.python.swing_sim.ground.strict_json import strict_json_object

GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2 = "rate-of-closure-ground-playback-workspace/v2"
GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2 = 11 * 1024 * 1024
GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT = 100_000
GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED = 200_000


def _positive_limit(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_limit(value: object, name: str, hard_cap: int) -> int:
    normalized = _positive_limit(value, name)
    if normalized > hard_cap:
        raise ValueError(f"{name} cannot exceed the {hard_cap} hard cap")
    return normalized


def _validate_point_limits(
    result: GroundSimulationResult,
    comparison: GroundSimulationResult | None,
    *,
    max_points_per_result: int,
    max_combined_points: int,
) -> None:
    per_result = _bounded_limit(
        max_points_per_result,
        "max_points_per_result",
        GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
    )
    combined = _bounded_limit(
        max_combined_points,
        "max_combined_points",
        GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
    )
    counts = [len(result.trajectory)]
    if comparison is not None:
        counts.append(len(comparison.trajectory))
    if any(count > per_result for count in counts):
        raise ValueError("ground workspace exceeds the per-result point limit")
    if sum(counts) > combined:
        raise ValueError("ground workspace exceeds the combined point limit")


@dataclass(frozen=True)
class GroundPlaybackComparisonState:
    """One exact comparison result and its independent overlay visibility."""

    result: GroundSimulationResult
    visible: bool

    def __post_init__(self) -> None:
        if type(self.result) is not GroundSimulationResult:
            raise TypeError(
                "comparison result must use the exact GroundSimulationResult type"
            )
        GroundPlaybackTimeline(self.result)
        if type(self.visible) is not bool:
            raise TypeError("comparison visible must be a boolean")


@dataclass(frozen=True)
class GroundPlaybackWorkspaceV2:
    """Validated primary, optional comparison, playback, and camera state."""

    result: GroundSimulationResult
    comparison: GroundPlaybackComparisonState | None
    playback: GroundPlaybackState
    view: GroundPlaybackViewState
    schema_version: str = GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2

    def __post_init__(self) -> None:
        if type(self.result) is not GroundSimulationResult:
            raise TypeError("result must use the exact GroundSimulationResult type")
        primary = GroundPlaybackTimeline(self.result)
        if (
            self.comparison is not None
            and type(self.comparison) is not GroundPlaybackComparisonState
        ):
            raise TypeError(
                "comparison must use the exact GroundPlaybackComparisonState "
                "type or None"
            )
        if type(self.playback) is not GroundPlaybackState:
            raise TypeError("playback must use the exact GroundPlaybackState type")
        if type(self.view) is not GroundPlaybackViewState:
            raise TypeError("view must use the exact GroundPlaybackViewState type")
        if self.schema_version != GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2:
            raise ValueError("unsupported ground playback workspace schema_version")
        comparison_timeline = (
            GroundPlaybackTimeline(self.comparison.result)
            if self.comparison is not None
            else None
        )
        start_time_s = min(
            primary.start_time_s,
            (
                comparison_timeline.start_time_s
                if comparison_timeline is not None
                else primary.start_time_s
            ),
        )
        end_time_s = max(
            primary.end_time_s,
            (
                comparison_timeline.end_time_s
                if comparison_timeline is not None
                else primary.end_time_s
            ),
        )
        if not start_time_s <= self.playback.time_s <= end_time_s:
            raise ValueError("playback time_s must lie within the union timeline")
        _validate_point_limits(
            self.result,
            self.comparison.result if self.comparison is not None else None,
            max_points_per_result=GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
            max_combined_points=GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
        )


@dataclass(frozen=True)
class GroundPlaybackWorkspaceLoad:
    """Normalized v2 workspace plus its exact source-schema provenance."""

    workspace: GroundPlaybackWorkspaceV2
    source_schema_version: str

    def __post_init__(self) -> None:
        if type(self.workspace) is not GroundPlaybackWorkspaceV2:
            raise TypeError(
                "workspace must use the exact GroundPlaybackWorkspaceV2 type"
            )
        if self.source_schema_version not in {
            GROUND_PLAYBACK_WORKSPACE_SCHEMA,
            GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2,
        }:
            raise ValueError("unsupported source workspace schema_version")

    @property
    def migrated_from_v1(self) -> bool:
        """Return whether this load performed the documented v1-to-v2 migration."""
        source_schema: str = self.source_schema_version
        legacy_schema: str = GROUND_PLAYBACK_WORKSPACE_SCHEMA
        return source_schema == legacy_schema


def _comparison_payload(
    comparison: GroundPlaybackComparisonState | None,
) -> dict[str, Any] | None:
    if comparison is None:
        return None
    return {"result": comparison.result.to_dict(), "visible": comparison.visible}


def _workspace_payload(workspace: GroundPlaybackWorkspaceV2) -> dict[str, Any]:
    return {
        "comparison": _comparison_payload(workspace.comparison),
        "playback": {
            "loop": workspace.playback.loop,
            "speed": workspace.playback.speed,
            "time_s": workspace.playback.time_s,
        },
        "result": workspace.result.to_dict(),
        "schema_version": workspace.schema_version,
        "view": {
            "pitch_deg": workspace.view.pitch_deg,
            "yaw_deg": workspace.view.yaw_deg,
            "zoom": workspace.view.zoom,
        },
    }


def _result_from_payload(value: object, name: str) -> GroundSimulationResult:
    payload = workspace_object(value, name)
    return cast(
        GroundSimulationResult,
        record_from_dict(GroundSimulationResult, payload),
    )


def _parse_comparison(value: object) -> GroundPlaybackComparisonState | None:
    if value is None:
        return None
    payload = workspace_object(value, "comparison")
    exact_workspace_fields(payload, {"result", "visible"}, "comparison", "v2")
    if type(payload["visible"]) is not bool:
        raise TypeError("comparison visible must be a boolean")
    return GroundPlaybackComparisonState(
        _result_from_payload(payload["result"], "comparison result"),
        payload["visible"],
    )


def migrate_ground_workspace_v1(
    workspace: GroundPlaybackWorkspace,
) -> GroundPlaybackWorkspaceV2:
    """Normalize one validated v1 workspace into the v2 in-memory contract."""
    if type(workspace) is not GroundPlaybackWorkspace:
        raise TypeError("workspace must use the exact GroundPlaybackWorkspace type")
    return GroundPlaybackWorkspaceV2(
        result=workspace.result,
        comparison=None,
        playback=workspace.playback,
        view=workspace.view,
    )


def _parse_v2_payload(payload: dict[str, Any]) -> GroundPlaybackWorkspaceV2:
    exact_workspace_fields(
        payload,
        {"schema_version", "result", "comparison", "playback", "view"},
        "workspace",
        "v2",
    )
    playback = workspace_object(payload["playback"], "playback")
    view = workspace_object(payload["view"], "view")
    exact_workspace_fields(playback, {"time_s", "speed", "loop"}, "playback", "v2")
    exact_workspace_fields(view, {"yaw_deg", "pitch_deg", "zoom"}, "view", "v2")
    return GroundPlaybackWorkspaceV2(
        result=_result_from_payload(payload["result"], "result"),
        comparison=_parse_comparison(payload["comparison"]),
        playback=GroundPlaybackState(
            playback["time_s"], playback["speed"], playback["loop"]
        ),
        view=GroundPlaybackViewState(view["yaw_deg"], view["pitch_deg"], view["zoom"]),
        schema_version=payload["schema_version"],
    )


def _decode_workspace_document(
    text: str,
    *,
    max_bytes: int = GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
    max_points_per_result: int = GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
    max_combined_points: int = GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
) -> tuple[dict[str, Any], int]:
    """Decode one bounded workspace document for strict or dispatch loaders."""
    if type(text) is not str:
        raise TypeError("ground playback workspace JSON must be text")
    byte_limit = _bounded_limit(
        max_bytes, "max_bytes", GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2
    )
    _bounded_limit(
        max_points_per_result,
        "max_points_per_result",
        GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
    )
    _bounded_limit(
        max_combined_points,
        "max_combined_points",
        GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
    )
    if len(text.encode("utf-8")) > byte_limit:
        raise ValueError("ground playback workspace JSON exceeds the import size limit")
    return strict_json_object(text), byte_limit


def _validate_candidate_limits(
    candidate: GroundPlaybackWorkspaceV2,
    *,
    max_points_per_result: int,
    max_combined_points: int,
) -> GroundPlaybackWorkspaceV2:
    _validate_point_limits(
        candidate.result,
        candidate.comparison.result if candidate.comparison is not None else None,
        max_points_per_result=max_points_per_result,
        max_combined_points=max_combined_points,
    )
    return candidate


def ground_workspace_v2_from_json(
    text: str,
    *,
    max_bytes: int = GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
    max_points_per_result: int = GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
    max_combined_points: int = GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
) -> GroundPlaybackWorkspaceV2:
    """Parse only one strict v2 workspace under comparison-aware bounds."""
    payload, _byte_limit = _decode_workspace_document(
        text,
        max_bytes=max_bytes,
        max_points_per_result=max_points_per_result,
        max_combined_points=max_combined_points,
    )
    if payload.get("schema_version") != GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2:
        raise ValueError("expected ground playback workspace schema v2")
    return _validate_candidate_limits(
        _parse_v2_payload(payload),
        max_points_per_result=max_points_per_result,
        max_combined_points=max_combined_points,
    )


def load_ground_workspace_versioned_json(
    text: str,
    *,
    max_bytes: int = GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
    max_points_per_result: int = GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
    max_combined_points: int = GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
) -> GroundPlaybackWorkspaceLoad:
    """Dispatch v1/v2 and retain source-version provenance with normalized v2."""
    payload, byte_limit = _decode_workspace_document(
        text,
        max_bytes=max_bytes,
        max_points_per_result=max_points_per_result,
        max_combined_points=max_combined_points,
    )
    schema_version = payload.get("schema_version")
    if schema_version == GROUND_PLAYBACK_WORKSPACE_SCHEMA:
        candidate = migrate_ground_workspace_v1(
            ground_workspace_from_json(
                text,
                max_bytes=byte_limit,
                max_points=max_points_per_result,
            )
        )
    elif schema_version == GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2:
        candidate = _parse_v2_payload(payload)
    else:
        raise ValueError("unsupported ground playback workspace schema_version")
    return GroundPlaybackWorkspaceLoad(
        workspace=_validate_candidate_limits(
            candidate,
            max_points_per_result=max_points_per_result,
            max_combined_points=max_combined_points,
        ),
        source_schema_version=cast(str, schema_version),
    )


def ground_workspace_from_versioned_json(
    text: str,
    *,
    max_bytes: int = GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
    max_points_per_result: int = GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT,
    max_combined_points: int = GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED,
) -> GroundPlaybackWorkspaceV2:
    """Dispatch strict v1/v2 and return the normalized workspace v2 object."""
    return load_ground_workspace_versioned_json(
        text,
        max_bytes=max_bytes,
        max_points_per_result=max_points_per_result,
        max_combined_points=max_combined_points,
    ).workspace


def ground_workspace_v2_to_json(
    workspace: GroundPlaybackWorkspaceV2,
    *,
    max_bytes: int = GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2,
) -> str:
    """Serialize one exact v2 workspace and enforce its UTF-8 output bound."""
    if type(workspace) is not GroundPlaybackWorkspaceV2:
        raise TypeError("workspace must use the exact GroundPlaybackWorkspaceV2 type")
    byte_limit = _bounded_limit(
        max_bytes, "max_bytes", GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2
    )
    document = f"{canonical_numeric_json(_workspace_payload(workspace))}\n"
    if len(document.encode("utf-8")) > byte_limit:
        raise ValueError("ground playback workspace JSON exceeds the output size limit")
    return document


__all__ = [
    "GROUND_PLAYBACK_WORKSPACE_MAX_BYTES_V2",
    "GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_COMBINED",
    "GROUND_PLAYBACK_WORKSPACE_MAX_POINTS_PER_RESULT",
    "GROUND_PLAYBACK_WORKSPACE_SCHEMA_V2",
    "GroundPlaybackComparisonState",
    "GroundPlaybackWorkspaceLoad",
    "GroundPlaybackWorkspaceV2",
    "ground_workspace_from_versioned_json",
    "ground_workspace_v2_from_json",
    "ground_workspace_v2_to_json",
    "load_ground_workspace_versioned_json",
    "migrate_ground_workspace_v1",
]
