"""Contract and golden tests for the selected-head engineering sidecar."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.club import get_club, serialize_clubhead_stl
from rate_of_closure.club.engineering_sidecar import (
    default_clubhead_engineering_filename,
    serialize_clubhead_engineering_sidecar,
    write_clubhead_engineering_sidecar_atomic,
)

pytestmark = pytest.mark.unit

_DRIVER = "Driver 10.5\u00b0"
_FIXTURE = (
    Path(__file__).parent / "fixtures" / "clubhead_engineering_sidecar_driver_10_5.json"
)


def test_driver_sidecar_matches_cross_surface_golden_contract() -> None:
    """The sidecar is deterministic and names unavailable physics honestly."""
    payload = serialize_clubhead_engineering_sidecar(get_club(_DRIVER))

    assert payload == _FIXTURE.read_bytes()
    assert payload.endswith(b"\n")
    assert json.loads(payload)["mesh"]["companion_filename"] == "driver-10-5.stl"


def test_unavailable_tensor_and_cg_never_contain_substitute_values() -> None:
    """Partial offsets and a scalar moment cannot masquerade as full values."""
    document = json.loads(serialize_clubhead_engineering_sidecar(get_club(_DRIVER)))
    head = document["mass_properties"]["head"]

    assert head["center_of_mass_m"]["status"] == "unavailable"
    assert "value" not in head["center_of_mass_m"]
    assert head["inertia_tensor_at_com_kg_m2"]["status"] == "unavailable"
    assert "value" not in head["inertia_tensor_at_com_kg_m2"]
    assert document["mass_properties"]["assembly"]["status"] == "unavailable"
    assert "value" not in document["mass_properties"]["assembly"]
    assert document["frames"]["world_from_head"]["status"] == "unavailable"
    assert "rotation" not in document["frames"]["world_from_head"]


def test_sidecar_digest_identifies_the_unchanged_companion_stl() -> None:
    """Non-mesh evidence changes do not alter the deterministic STL identity."""
    base = get_club(_DRIVER)
    edited = replace(
        base,
        name="Same mesh evidence",
        length_m=0.9,
        lie_deg=70.0,
        moi_about_shaft_kg_m2=1.0e-3,
        cg_depth_m=0.04,
        cg_height_m=0.04,
    )
    base_document = json.loads(serialize_clubhead_engineering_sidecar(base))
    edited_document = json.loads(serialize_clubhead_engineering_sidecar(edited))

    assert serialize_clubhead_stl(base) == serialize_clubhead_stl(edited)
    assert base_document["mesh"]["sha256"] == edited_document["mesh"]["sha256"]
    assert (
        base_document["mass_properties"]["head"]
        != edited_document["mass_properties"]["head"]
    )


def test_sidecar_filename_is_portable_and_atomic_write_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The JSON artifact follows STL filename hardening and atomic replacement."""
    spec = get_club(_DRIVER)
    assert default_clubhead_engineering_filename(spec) == "driver-10-5.engineering.json"
    output = tmp_path / "driver.engineering.json"
    output.write_bytes(b"existing artifact")

    def fail_replace(_source: Path, _target: Path) -> Path:
        raise OSError("replace denied")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace denied"):
        write_clubhead_engineering_sidecar_atomic(spec, output)

    assert output.read_bytes() == b"existing artifact"
    assert list(tmp_path.glob(".driver.engineering.json.*.tmp")) == []
