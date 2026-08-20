"""Contract tests for the launch-monitor player workspace seam."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_workspace import (
    AnalysisSelection,
    DatasetReference,
    LaunchMonitorProject,
    PlayerIdentityBinding,
    build_player_covariation_request,
    export_analysis_bundle,
    load_project,
    save_project,
)


def _reference() -> DatasetReference:
    return DatasetReference(
        source_name="private-corpus",
        repository="D-sorganization/Launch-Monitor-Flight-Model-Campaign",
        revision="97f3ecf",
        relative_path="data/authority/database/shot_corpus_parquet",
        sha256="a" * 64,
        row_count=261_666,
    )


def _project() -> LaunchMonitorProject:
    return LaunchMonitorProject(
        name="Face and path",
        dataset=_reference(),
        identity=PlayerIdentityBinding("player_id", user_attested=True),
        selection=AnalysisSelection("face_angle", "club_path", 10, 0.95),
    )


def test_player_identity_must_be_explicit_and_user_attested() -> None:
    with pytest.raises(ValueError, match="attested"):
        PlayerIdentityBinding("player_id", user_attested=False)
    with pytest.raises(ValueError, match="column"):
        PlayerIdentityBinding("", user_attested=True)


def test_covariation_request_delegates_to_backend_contract_v2() -> None:
    request = build_player_covariation_request(_project())

    assert request["contract_version"] == "2.0.0"
    assert request["operation"] == "player_covariation"
    assert request["player_identity"] == {
        "column": "player_id",
        "user_attested": True,
    }
    assert request["variables"] == {"x": "face_angle", "y": "club_path"}
    assert "records" not in request


def test_project_round_trip_keeps_reference_but_never_embeds_rows(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "analysis.lmproject.json"
    save_project(destination, _project())
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["contract_version"] == "2.0.0"
    assert payload["dataset"]["revision"] == "97f3ecf"
    assert "rows" not in payload
    assert load_project(destination) == _project()


def test_full_export_contains_project_result_backing_rows_and_hashes(
    tmp_path: Path,
) -> None:
    rows = pd.DataFrame(
        {
            "shot_id": ["s1", "s2"],
            "player_id": ["p1", "p1"],
            "face_angle": [1.0, 2.0],
            "club_path": [0.5, 1.5],
        }
    )
    output = export_analysis_bundle(
        tmp_path / "bundle", _project(), {"contract_version": "2.0.0"}, rows
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        "project.json",
        "result.json",
        "backing_rows.csv",
    }
    assert all(len(item["sha256"]) == 64 for item in manifest["files"].values())
    assert (output / "backing_rows.csv").read_text(encoding="utf-8").splitlines()[
        0
    ] == ("shot_id,player_id,face_angle,club_path")


def test_project_rejects_malformed_or_unpinned_dataset() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        DatasetReference("source", "org/repo", "rev", "data", "nope", 1)
    with pytest.raises(ValueError, match="different"):
        LaunchMonitorProject(
            name="bad",
            dataset=_reference(),
            identity=PlayerIdentityBinding("face_angle", user_attested=True),
            selection=AnalysisSelection("face_angle", "club_path", 3, 0.95),
        )


def test_pyqt_player_workspace_runs_grouped_analysis_only_after_attestation(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    from rate_of_closure.ui.pyqt6.launch_monitor_player_workspace import (
        LaunchMonitorPlayerWorkspace,
    )

    panel = LaunchMonitorPlayerWorkspace()
    qtbot.addWidget(panel)
    frame = pd.DataFrame(
        {
            "player_id": ["p1"] * 6 + ["p2"] * 6,
            "face_angle": list(range(12)),
            "club_path": [value * 0.8 + value % 2 for value in range(12)],
        }
    )
    panel.set_dataset(frame, "test.csv")
    panel.identity_combo.setCurrentText("player_id")
    panel.min_samples_spin.setValue(4)
    assert not panel.run_button.isEnabled()
    panel.attestation.setChecked(True)
    assert panel.run_button.isEnabled()
    result = panel.run_player_analysis()

    assert result.request.group_by == "player_id"
    assert len(result.groups) == 2
    assert "2 player groups analyzed" in panel.status.text()
    assert panel.covariation_result is not None
    assert panel.covariation_result.meta_analysis.contributor_count == 2
    assert panel.covariation_view.table.rowCount() == 2
    assert "unknown" in panel.covariation_view.axes[0].get_xlabel()
    assert panel._export_payload["backing_data"]
    panel.run_pair_scan()
    assert panel._export_payload["mode"] == "exploratory_pair_scan"
    assert panel.covariation_view.table.rowCount() > 0
