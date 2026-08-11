"""Canonical regional surface-plan import/export boundary tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.application.regional_surface_plan import (
    EDITOR_PROVIDER_ID,
    editor_draft_from_regional_surface_plan_request,
    illustrative_regional_surface_plan_draft,
    regional_surface_plan_request_for_draft,
    validate_regional_surface_plan_draft,
)
from rate_of_closure.application.regional_surface_plan_files import (
    read_regional_surface_plan_request,
    write_regional_surface_plan_request_atomic,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    MAX_REGIONAL_PLAN_WIRE_BYTES,
)
from shared.python.swing_sim.ground.regional_plan_wire import (
    regional_material_plan_request_from_dict,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_editor_request_file_round_trip_is_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    request = validate_regional_surface_plan_draft(
        illustrative_regional_surface_plan_draft()
    )
    target = tmp_path / "regional-plan.json"

    assert write_regional_surface_plan_request_atomic(request, target)

    assert target.read_bytes() == request.to_json().encode("utf-8")
    loaded = read_regional_surface_plan_request(target)
    assert loaded == request
    imported = editor_draft_from_regional_surface_plan_request(loaded)
    assert regional_surface_plan_request_for_draft(imported, loaded) == request


def test_cancelled_request_write_is_a_no_op(tmp_path: Path) -> None:
    request = validate_regional_surface_plan_draft(
        illustrative_regional_surface_plan_draft()
    )

    assert write_regional_surface_plan_request_atomic(request, None) is False
    assert list(tmp_path.iterdir()) == []


def test_request_replace_failure_preserves_last_known_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rate_of_closure.application import atomic_text_files

    request = validate_regional_surface_plan_draft(
        illustrative_regional_surface_plan_draft()
    )
    target = tmp_path / "regional-plan.json"
    target.write_text("last-known-good", encoding="utf-8")
    monkeypatch.setattr(
        atomic_text_files.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        write_regional_surface_plan_request_atomic(request, target)

    assert target.read_text(encoding="utf-8") == "last-known-good"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "text, message",
    [
        ("{not-json", "JSON"),
        ('{"request_id":"one","request_id":"two"}', "[Dd]uplicate"),
        (" " * (MAX_REGIONAL_PLAN_WIRE_BYTES + 1), "maximum wire size"),
    ],
    ids=("corrupt", "duplicate", "oversize"),
)
def test_request_read_rejects_corruption_duplicates_and_oversize(
    tmp_path: Path, text: str, message: str
) -> None:
    target = tmp_path / "invalid.json"
    target.write_text(text, encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        read_regional_surface_plan_request(target)


def test_import_rejects_non_editor_provenance_without_coercion() -> None:
    request = validate_regional_surface_plan_draft(
        illustrative_regional_surface_plan_draft()
    )
    payload = request.to_dict()
    payload["provenance"]["producer"] = "external.course.authority"
    external = regional_material_plan_request_from_dict(payload)

    with pytest.raises(ValueError, match="editor producer"):
        editor_draft_from_regional_surface_plan_request(external)

    payload = request.to_dict()
    payload["provenance"]["input_sha256"] = "0" * 64
    mismatched = regional_material_plan_request_from_dict(payload)
    with pytest.raises(ValueError, match="digest does not match"):
        editor_draft_from_regional_surface_plan_request(mismatched)


def test_import_rejects_editor_provider_with_unsupported_axis() -> None:
    request = validate_regional_surface_plan_draft(
        illustrative_regional_surface_plan_draft()
    )
    payload = request.to_dict()
    payload["axis_unit"] = [0.0, 0.0, 1.0]
    rotated = regional_material_plan_request_from_dict(payload)

    with pytest.raises(ValueError, match="axis qualification"):
        editor_draft_from_regional_surface_plan_request(rotated)


def test_edited_import_rebinds_provenance_instead_of_reusing_stale_digest() -> None:
    request = validate_regional_surface_plan_draft(
        illustrative_regional_surface_plan_draft()
    )
    imported = editor_draft_from_regional_surface_plan_request(request)
    edited = replace(imported, request_id="edited-plan")

    rebound = regional_surface_plan_request_for_draft(edited, request)

    assert rebound.request_id == "edited-plan"
    assert rebound.provenance.producer == EDITOR_PROVIDER_ID
    assert rebound.provenance.input_sha256 != request.provenance.input_sha256
