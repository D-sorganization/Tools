"""Contract tests enforcing forbidden-identity policy for launch-monitor analytics.

Player identity must never be inferred from session, club, source, filename,
or row order. Analysis requires explicitly selected, user-attested identity
evidence and fails closed otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_canonical_v2 import (
    build_player_covariation_payload,
    validate_player_covariation_response,
)
from rate_of_closure.launch_monitor_longitudinal import (
    LongitudinalRequest,
    analyze_longitudinal_performance,
)
from rate_of_closure.launch_monitor_workspace import (
    AnalysisSelection,
    DatasetReference,
    LaunchMonitorProject,
    PlayerIdentityBinding,
    build_player_covariation_request,
)
from rate_of_closure.launch_monitor_workspace_v3 import (
    parse_workspace_project,
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


def _dataset_reference() -> DatasetReference:
    return DatasetReference(
        source_name="authorized-corpus",
        repository="D-sorganization/Launch-Monitor-Flight-Model-Campaign",
        revision="d469b8a427418fa00e99b0ad488e4310b067697d",
        relative_path="data/authority/database/shot_corpus_parquet",
        sha256="a" * 64,
        row_count=261_666,
    )


def _synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"s-{i}" for i in range(12)],
            "session_id": ["sess-1"] * 4 + ["sess-2"] * 4 + ["sess-3"] * 4,
            "session_order": [1] * 4 + [2] * 4 + [3] * 4,
            "club": ["Driver"] * 12,
            "source_id": ["campaign-source"] * 12,
            "filename": ["data.csv"] * 12,
            "row_index": list(range(12)),
            "monitor_vendor": ["TrackMan"] * 12,
            "player_id": ["p1"] * 12,
            "face_angle": [1.0, 1.5, 2.0, 2.5] * 3,
            "club_path": [0.5, 1.0, 1.5, 2.0] * 3,
            "ball_speed": [
                150.0,
                151.0,
                152.0,
                153.0,
                155.0,
                156.0,
                157.0,
                158.0,
                160.0,
                161.0,
                162.0,
                163.0,
            ],
            "start_lie": ["fairway"] * 12,
            "start_context": ["standard"] * 12,
            "finish_lie": ["green"] * 12,
            "finish_context": ["standard"] * 12,
            "target": ["hole-1"] * 12,
            "start_distance": [150.0] * 12,
            "finish_distance": [15.0] * 12,
        }
    )


def test_player_identity_binding_rejects_unattested_and_blank_columns() -> None:
    """PlayerIdentityBinding strictly rejects unattested or blank column names."""
    with pytest.raises(ValueError, match="user-attested"):
        PlayerIdentityBinding("player_id", user_attested=False)

    with pytest.raises(ValueError, match="column"):
        PlayerIdentityBinding("", user_attested=True)

    with pytest.raises(ValueError, match="column"):
        PlayerIdentityBinding("   ", user_attested=True)


def test_covariation_request_builder_enforces_attestation() -> None:
    """Workspace request builder requires explicit user attestation."""
    project = LaunchMonitorProject(
        name="Test",
        dataset=_dataset_reference(),
        identity=PlayerIdentityBinding("player_id", user_attested=True),
        selection=AnalysisSelection("face_angle", "club_path", 4, 0.95),
    )
    request = build_player_covariation_request(project)
    assert request["player_identity"]["user_attested"] is True
    assert request["player_identity"]["column"] == "player_id"


def test_workspace_v3_fails_closed_on_unattested_player_or_session() -> None:
    """Workspace v3 parser rejects unattested player, session, or order evidence."""
    unattested_player = {
        **GOLDEN_V3,
        "identity_evidence": {
            **GOLDEN_V3["identity_evidence"],
            "player": {
                **GOLDEN_V3["identity_evidence"]["player"],
                "user_attested": False,
            },
        },
    }
    with pytest.raises(ValueError, match="player identity must be user-attested"):
        parse_workspace_project(unattested_player)

    missing_evidence = {
        **GOLDEN_V3,
        "identity_evidence": {
            **GOLDEN_V3["identity_evidence"],
            "player": {
                "column": "player_id",
                "user_attested": True,
                "evidence": "",
            },
        },
    }
    with pytest.raises(ValueError, match="player evidence is required"):
        parse_workspace_project(missing_evidence)


def test_player_covariation_payload_builder_enforces_distinct_columns() -> None:
    """Payload builder forbids using player_id as x or y variable."""
    records: list[dict[str, Any]] = [{"player_id": "p1", "face_angle": 1.0}]
    with pytest.raises(ValueError, match="distinct and non-empty"):
        build_player_covariation_payload(
            records,
            player_column="face_angle",
            x_column="face_angle",
            y_column="club_path",
            min_samples=4,
            confidence_level=0.95,
        )

    with pytest.raises(ValueError, match="distinct and non-empty"):
        build_player_covariation_payload(
            records,
            player_column="",
            x_column="face_angle",
            y_column="club_path",
            min_samples=4,
            confidence_level=0.95,
        )


def test_player_covariation_response_validator_rejects_untrusted_identity() -> None:
    """Canonical validator rejects untrusted or inferred player identity."""
    response = {
        "contract_version": "launch-monitor-player-covariation/1.0.0",
        "analysis_kind": "selected_pair",
        "status": "available",
        "request": {},
        "pooled": {},
        "within_player": {},
        "between_player": {},
        "per_player": [],
        "meta_analysis": {},
        "missingness": {},
        "units": {},
        "lineage": {"backing_records": []},
        "availability": [],
        "uncertainty": {},
        "player_identity": {
            "trust_level": "inferred_from_session",
            "identifier_column": "session_id",
        },
        "vendor_provenance": [],
        "claims": {
            "device_emulation": False,
            "device_certification": False,
            "causal_inference": False,
        },
        "definitions": {},
        "warnings": [],
    }
    with pytest.raises(ValueError, match="trusted identity"):
        validate_player_covariation_response(response)


def test_longitudinal_fails_closed_when_order_is_unattested() -> None:
    """Longitudinal analysis requires explicit order and fails closed if untrusted."""
    frame = _synthetic_frame()

    # Unattested player identity fails closed
    unattested_player_req = LongitudinalRequest(
        metric_column="ball_speed",
        session_column="session_id",
        session_order_column="session_order",
        player_column="player_id",
        player_identity_attested=False,
        session_identity_attested=True,
        higher_is_better=True,
        min_sessions=3,
    )
    with pytest.raises(ValueError, match="attested"):
        analyze_longitudinal_performance(frame, unattested_player_req)

    # Unattested session identity fails closed
    unattested_session_req = LongitudinalRequest(
        metric_column="ball_speed",
        session_column="session_id",
        session_order_column="session_order",
        player_column="player_id",
        player_identity_attested=True,
        session_identity_attested=False,
        higher_is_better=True,
        min_sessions=3,
    )
    with pytest.raises(ValueError, match="attested"):
        analyze_longitudinal_performance(frame, unattested_session_req)

    # Valid attestation runs successfully
    valid_req = LongitudinalRequest(
        metric_column="ball_speed",
        session_column="session_id",
        session_order_column="session_order",
        player_column="player_id",
        player_identity_attested=True,
        session_identity_attested=True,
        higher_is_better=True,
        min_sessions=3,
    )
    result = analyze_longitudinal_performance(frame, valid_req)
    assert len(result.players) == 1
    assert result.players[0].status == "ok"


def test_identity_cannot_smuggle_private_rows_in_workspace() -> None:
    """Identity evidence metadata cannot contain embedded row arrays."""
    smuggled = {
        **GOLDEN_V3,
        "identity_evidence": {
            **GOLDEN_V3["identity_evidence"],
            "player": {
                **GOLDEN_V3["identity_evidence"]["player"],
                "rows": [{"player_id": "p1"}],
            },
        },
    }
    with pytest.raises(ValueError, match="rows"):
        parse_workspace_project(smuggled)
