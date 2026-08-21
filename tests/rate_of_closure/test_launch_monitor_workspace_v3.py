"""Cross-client project/export v3 contracts for launch-monitor workspaces."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_workspace_v3 import (
    WorkspaceExportAuthorization,
    create_workspace_bundle,
    parse_workspace_project,
    serialize_workspace_project,
)


def _golden() -> dict[str, object]:
    path = (
        Path(__file__).parents[2]
        / "src/rate_of_closure/web/src/model/__fixtures__"
        / "launch_monitor_workspace_v3_golden.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_shared_v3_golden_round_trips_without_rows() -> None:
    project = parse_workspace_project(_golden())
    serialized = serialize_workspace_project(project)

    assert json.loads(serialized) == _golden()
    assert '"rows"' not in serialized
    assert project.identity_evidence.session is not None
    assert project.identity_evidence.order is not None


def test_v3_rejects_embedded_rows_and_unattested_identity() -> None:
    embedded = {**_golden(), "rows": [{"player_id": "p1"}]}
    with pytest.raises(ValueError, match="unknown|rows"):
        parse_workspace_project(embedded)

    unattested = _golden()
    evidence = unattested["identity_evidence"]
    assert isinstance(evidence, dict)
    player = evidence["player"]
    assert isinstance(player, dict)
    player["user_attested"] = False
    with pytest.raises(ValueError, match="attested"):
        parse_workspace_project(unattested)


def test_restricted_bundle_fails_closed_without_explicit_approval(
    tmp_path: Path,
) -> None:
    rows = pd.DataFrame({"player_id": ["p1"], "face_angle": [1.0]})
    output = create_workspace_bundle(
        tmp_path / "bundle",
        parse_workspace_project(_golden()),
        rows,
        WorkspaceExportAuthorization(include_backing_rows=True),
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert "backing_rows.csv" not in manifest["files"]
    assert "backing_join.csv" not in manifest["files"]
    assert manifest["backing_data"]["status"] == "unavailable"
    assert "restricted approval" in manifest["backing_data"]["reason"]


def test_authorized_restricted_bundle_has_deterministic_row_join(
    tmp_path: Path,
) -> None:
    rows = pd.DataFrame({"player_id": ["p1", "p1"], "face_angle": [1.0, 2.0]})
    authorization = WorkspaceExportAuthorization(
        include_backing_rows=True,
        restricted_data_approved=True,
    )
    first = create_workspace_bundle(
        tmp_path / "first", parse_workspace_project(_golden()), rows, authorization
    )
    second = create_workspace_bundle(
        tmp_path / "second", parse_workspace_project(_golden()), rows, authorization
    )

    first_join = (first / "backing_join.csv").read_text(encoding="utf-8")
    assert first_join == (second / "backing_join.csv").read_text(encoding="utf-8")
    assert first_join.splitlines()[0] == "result_row_index,row_sha256"
    expected_row_hash = (
        "44bcce6d01c15405"  # pragma: allowlist secret
        "60681daf1db90869"  # pragma: allowlist secret
        "e9d4b15cd59d43bc"  # pragma: allowlist secret
        "f99ab4ea694ccf1d"  # pragma: allowlist secret
    )
    assert first_join.splitlines()[1].endswith(expected_row_hash)
    assert (first / "backing_rows.csv").is_file()


def test_result_payload_cannot_smuggle_backing_rows() -> None:
    payload = _golden()
    analyses = payload["analyses"]
    assert isinstance(analyses, list)
    analysis = analyses[0]
    assert isinstance(analysis, dict)
    result = analysis["result"]
    assert isinstance(result, dict)
    result["status"] = "available"
    result["payload"] = {"records": [{"player_id": "p1"}]}
    result["response_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="row-bearing|records"):
        parse_workspace_project(payload)
