"""Atomic workspace file-adapter tests for Tools #4220."""

from __future__ import annotations

from pathlib import Path

import pytest

from rate_of_closure.application import workspace_files
from rate_of_closure.application.workspace_document import (
    VersionedPayload,
    WorkspaceDocument,
    WorkspaceLayout,
    WorkspaceMetadata,
)


def _document(title: str) -> WorkspaceDocument:
    return WorkspaceDocument(
        metadata=WorkspaceMetadata(
            document_id="workspace.file.test",
            title=title,
            created_at_utc="2026-08-07T12:00:00Z",
            modified_at_utc="2026-08-07T12:00:00Z",
            app_version="1.0.0",
            provenance={},
        ),
        model_session=VersionedPayload("session", 1, {}),
        prescribed_torque_profiles=(),
        club_configuration=VersionedPayload("club", 1, {}),
        variation_plan=None,
        layout=WorkspaceLayout(("simulation",), ("simulation",), "simulation"),
    )


def test_atomic_write_and_validated_read_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "showcase.roc-workspace.json"

    assert workspace_files.write_workspace_atomic(_document("First"), target)

    assert workspace_files.read_workspace(target) == _document("First")
    assert not list(tmp_path.glob(".*.tmp"))


def test_cancelled_destination_is_a_no_op(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())

    assert workspace_files.write_workspace_atomic(_document("Ignored"), None) is False

    assert set(tmp_path.iterdir()) == before


def test_serialization_failure_does_not_touch_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "showcase.json"
    target.write_text("last-known-good", encoding="utf-8")

    def fail_serialization(_document: WorkspaceDocument) -> str:
        raise ValueError("invalid workspace")

    monkeypatch.setattr(workspace_files, "workspace_to_json", fail_serialization)

    with pytest.raises(ValueError, match="invalid workspace"):
        workspace_files.write_workspace_atomic(_document("New"), target)
    assert target.read_text(encoding="utf-8") == "last-known-good"
    assert not list(tmp_path.glob(".*.tmp"))


def test_replace_failure_preserves_existing_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rate_of_closure.application import atomic_text_files

    target = tmp_path / "showcase.json"
    target.write_text("last-known-good", encoding="utf-8")

    def fail_replace(_source: str | Path, _target: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(atomic_text_files.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        workspace_files.write_workspace_atomic(_document("New"), target)
    assert target.read_text(encoding="utf-8") == "last-known-good"
    assert not list(tmp_path.glob(".*.tmp"))


def test_read_rejects_invalid_document(tmp_path: Path) -> None:
    target = tmp_path / "invalid.json"
    target.write_text('{"schema":"wrong"}', encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        workspace_files.read_workspace(target)


def test_atomic_text_export_preserves_existing_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "layout.json"
    target.write_text("last-known-good", encoding="utf-8")

    def fail_replace(_source: str | Path, _target: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(workspace_files.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        workspace_files.write_text_atomic('{"new":true}\n', target)
    assert target.read_text(encoding="utf-8") == "last-known-good"
    assert not list(tmp_path.glob(".*.tmp"))
