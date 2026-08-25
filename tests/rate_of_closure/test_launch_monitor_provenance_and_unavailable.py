"""Comprehensive provenance, persistence/load, and unavailable-state contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_canonical_v2 import (
    load_canonical_dataset_reference,
    validate_dataset_job_page,
)
from rate_of_closure.launch_monitor_strokes_gained import (
    SourceBackedStrokesGainedRequest,
    StrokesGainedBaseline,
    calculate_source_backed_strokes_gained,
)
from rate_of_closure.launch_monitor_strokes_gained_baseline import (
    BaselineState,
)
from rate_of_closure.launch_monitor_workspace_v3 import (
    WorkspaceExportAuthorization,
    create_workspace_bundle,
    parse_workspace_project,
    serialize_workspace_project,
)
from rate_of_closure.player_covariation import (
    CovariationRequest,
    analyze_player_covariation,
)

FIXTURES_DIR = (
    Path(__file__).parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
)
GOLDEN_V3 = json.loads(
    (FIXTURES_DIR / "launch_monitor_workspace_v3_golden.json").read_text(
        encoding="utf-8"
    )
)


def _baseline() -> StrokesGainedBaseline:
    states = (
        BaselineState("fairway", "standard", "hole-1", 100.0, 2.8, 0.1),
        BaselineState("fairway", "standard", "hole-1", 200.0, 3.8, 0.14),
        BaselineState("green", "standard", "hole-1", 20.0, 1.5, 0.08),
    )
    return StrokesGainedBaseline(
        baseline_id="synthetic-baseline-v1",
        version="1.0.0",
        source_url="https://example.org/baseline",
        license="CC0-1.0",
        table_sha256="b" * 64,
        states=states,
    )


def test_provenance_metadata_enforces_exact_sha_and_commit_length() -> None:
    """Validate canonical dataset references strictly enforce SHA256 and commits."""
    reference_data = {
        "root_id": "launch-monitor-authority",
        "repository": "D-sorganization/Launch-Monitor-Flight-Model-Campaign",
        "commit": "d469b8a427418fa00e99b0ad488e4310b067697d",
        "manifest_sha256": (
            "b45fd9100e6786d32dce229224ed901f02c20ef5c44962769faf6cc94700c299"
        ),
        "content_sha256": (
            "7bedf88ba473c947db2d4d078a73ee0ccd3512ffa182b751ea0a23298d1ab10c"
        ),
        "expected_row_count": 261666,
    }
    ref = load_canonical_dataset_reference(reference_data)
    assert ref.expected_row_count == 261666
    assert ref.commit == "d469b8a427418fa00e99b0ad488e4310b067697d"

    # Short commit SHA fails closed
    with pytest.raises(ValueError, match="commit"):
        load_canonical_dataset_reference({**reference_data, "commit": "d469b8a"})

    # Short manifest SHA fails closed
    with pytest.raises(ValueError, match="manifest_sha256"):
        load_canonical_dataset_reference(
            {**reference_data, "manifest_sha256": "abc123"}
        )


def test_persistence_load_round_trip_is_deterministic_and_row_free() -> None:
    """Verify serialization/deserialization maintain exact structure without rows."""
    project = parse_workspace_project(GOLDEN_V3)
    serialized = serialize_workspace_project(project)
    reloaded = parse_workspace_project(serialized)

    assert json.loads(serialized) == GOLDEN_V3
    assert reloaded == project
    assert '"rows"' not in serialized
    assert '"records"' not in serialized
    assert '"source_rows"' not in serialized


def test_bundle_export_handles_authorized_and_unauthorized_paths(
    tmp_path: Path,
) -> None:
    """Verify workspace bundle exports fail closed without approval."""
    project = parse_workspace_project(GOLDEN_V3)
    rows = pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p2", "p2"],
            "face_angle": [1.0, 1.5, 2.0, 2.5],
            "club_path": [0.5, 1.0, 1.5, 2.0],
        }
    )

    # Unauthorized export: backing rows omitted
    unauth_dir = create_workspace_bundle(
        tmp_path / "unauth_bundle",
        project,
        rows,
        WorkspaceExportAuthorization(
            include_backing_rows=True, restricted_data_approved=False
        ),
    )
    manifest_unauth = json.loads(
        (unauth_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_unauth["backing_data"]["status"] == "unavailable"
    assert "restricted approval" in manifest_unauth["backing_data"]["reason"]
    assert "backing_rows.csv" not in manifest_unauth["files"]

    # Authorized export: backing rows and deterministic joins included
    auth_dir = create_workspace_bundle(
        tmp_path / "auth_bundle",
        project,
        rows,
        WorkspaceExportAuthorization(
            include_backing_rows=True, restricted_data_approved=True
        ),
    )
    manifest_auth = json.loads((auth_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_auth["backing_data"]["status"] == "available"
    assert "backing_rows.csv" in manifest_auth["files"]
    assert "backing_join.csv" in manifest_auth["files"]

    join_content = (auth_dir / "backing_join.csv").read_text(encoding="utf-8")
    assert join_content.startswith("result_row_index,row_sha256\n")
    assert len(join_content.strip().split("\n")) == 5


def test_strokes_gained_fails_closed_when_course_state_unmet() -> None:
    """Verify source-backed strokes gained fails closed when course-state is missing."""
    frame = pd.DataFrame(
        {
            "shot_id": ["s1", "s2"],
            "start_lie": ["rough", "rough"],  # not in baseline table
            "start_context": ["standard", "standard"],
            "finish_lie": ["green", "green"],
            "finish_context": ["standard", "standard"],
            "target": ["hole-1", "hole-1"],
            "start_distance": [150.0, 160.0],
            "finish_distance": [10.0, 15.0],
        }
    )
    request = SourceBackedStrokesGainedRequest(
        before_lie_column="start_lie",
        before_context_column="start_context",
        before_target_column="target",
        before_distance_column="start_distance",
        after_lie_column="finish_lie",
        after_context_column="finish_context",
        after_target_column="target",
        after_distance_column="finish_distance",
        before_distance_unit="yd",
        after_distance_unit="yd",
    )
    with pytest.raises(ValueError, match="outside the baseline"):
        calculate_source_backed_strokes_gained(frame, _baseline(), request)


def test_player_covariation_returns_insufficient_samples_unavailable() -> None:
    """Verify player covariation returns insufficient_samples state."""
    frame = pd.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "face_angle": [1.0, 2.0],
            "club_path": [0.5, 1.5],
        }
    )
    request = CovariationRequest(
        player_column="player_id",
        x_column="face_angle",
        y_column="club_path",
        min_samples=4,
        confidence_level=0.95,
    )
    result = analyze_player_covariation(frame, request)
    assert result.pooled.status == "insufficient_samples"
    assert result.pooled.pearson_r is None
    assert result.within_player.status == "insufficient_samples"
    assert result.between_player.status == "insufficient_samples"


def test_dataset_job_page_strictly_rejects_private_rows() -> None:
    """Dataset job page validator rejects pages containing shot_id or row_index."""
    valid_page: dict[str, Any] = {
        "contract_version": "launch-monitor-dataset-job/1.0.0",
        "job_id": "c" * 32,
        "offset": 0,
        "limit": 50,
        "total_items": 1,
        "next_offset": None,
        "items": [
            {
                "group_by": "club",
                "group": "Driver",
                "metric": "ball_speed",
                "n": 100,
                "mean": 165.2,
                "standard_deviation": 3.4,
                "minimum": 155.0,
                "maximum": 172.0,
            }
        ],
    }
    validated = validate_dataset_job_page(valid_page)
    assert validated["total_items"] == 1

    leaky_page = {
        **valid_page,
        "items": [{**valid_page["items"][0], "shot_id": "secret-123"}],
    }
    with pytest.raises(ValueError, match="private rows"):
        validate_dataset_job_page(leaky_page)
