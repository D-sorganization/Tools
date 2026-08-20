"""Strict neutral Upstream terrain snapshot and one-way adapter tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from shared.python.swing_sim.ground.upstream_terrain_adapter import (
    UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION,
    FrameTransform,
    TerrainAdapterInterpretation,
    TerrainDispositionKind,
    TerrainFieldDisposition,
    UpstreamTerrainSnapshot,
    adapt_upstream_terrain_snapshot,
    upstream_snapshot_from_json,
)

_SHA = "c" * 64


def _snapshot() -> UpstreamTerrainSnapshot:
    return UpstreamTerrainSnapshot(
        terrain_id="fixture-region-7",
        terrain_revision="terrain-r3",
        source_frame_id="upstream:x_horizontal,y_horizontal,z_up",
        point_m=(10.0, 20.0, 2.0),
        normal_unit=(0.0, 0.0, 1.0),
        surface_velocity_m_s=(0.0, 0.0, 0.0),
        material_id="material-custom-fairway",
        material_revision="material-r2",
        material_name="Custom fairway",
        friction_coefficient=0.45,
        rolling_resistance=0.08,
        restitution=0.65,
        hardness_fraction=0.75,
        grass_height_m=0.015,
        compressibility_fraction=0.15,
        compression_damping_fraction=0.25,
        turf_density_kg_m3=120.0,
        moisture_fraction=0.3,
        source_sha256=_SHA,
    )


def _transform() -> FrameTransform:
    return FrameTransform(
        "upstream:x_horizontal,y_horizontal,z_up",
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        ),
        (0.0, 0.0, 0.0),
    )


def _interpretation() -> TerrainAdapterInterpretation:
    return TerrainAdapterInterpretation(
        source_friction_coefficient=0.45,
        static_friction=0.50,
        kinetic_friction=0.40,
        firmness_pa=1_000_000.0,
        friction_method="bounded engineering split supplied by the caller",
        firmness_method="caller-supplied instrument estimate",
    )


def test_snapshot_wire_is_strict_canonical_and_versioned() -> None:
    snapshot = _snapshot()
    text = snapshot.to_json()

    assert snapshot.schema_version == UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION
    assert upstream_snapshot_from_json(text) == snapshot
    assert (
        upstream_snapshot_from_json(text).canonical_sha256()
        == snapshot.canonical_sha256()
    )

    with pytest.raises(ValueError, match="canonical"):
        upstream_snapshot_from_json(text.replace(":", ": ", 1))
    with pytest.raises(ValueError, match="duplicate"):
        upstream_snapshot_from_json(
            text.replace('"terrain_id":', '"terrain_id":"duplicate","terrain_id":', 1)
        )
    payload = snapshot.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        UpstreamTerrainSnapshot.from_dict(payload)


def test_adapter_requires_proper_explicit_frame_and_interpretations() -> None:
    with pytest.raises(ValueError, match="proper rotation"):
        FrameTransform(
            _snapshot().source_frame_id,
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
            (0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="source friction"):
        adapt_upstream_terrain_snapshot(
            _snapshot(),
            _transform(),
            replace(_interpretation(), source_friction_coefficient=0.46),
        )
    with pytest.raises(ValueError, match="kinetic_friction"):
        replace(_interpretation(), kinetic_friction=0.6)


def test_adapter_maps_local_tangent_plane_and_emits_complete_loss_report() -> None:
    adapted = adapt_upstream_terrain_snapshot(
        _snapshot(), _transform(), _interpretation()
    )
    surface = adapted.surface

    assert surface.surface_id.startswith("upstream-terrain:")
    assert len(surface.surface_id) == len("upstream-terrain:") + 64
    assert surface.provider_version == (
        f"upstream-terrain-adapter/v1:{surface.surface_id.split(':', 1)[1]}"
    )
    assert surface.normal_unit == (0.0, 1.0, 0.0)
    assert surface.height_m == 2.0
    assert surface.static_friction == 0.5
    assert surface.kinetic_friction == 0.4
    assert surface.firmness_pa == 1_000_000.0
    assert surface.rolling_resistance == 0.08
    assert surface.turf_density_kg_m3 == 120.0
    assert adapted.source_sha256 == _SHA
    assert adapted.adapter_version == "upstream-terrain-adapter/v1"
    assert len(adapted.transform_sha256) == 64
    assert len(adapted.interpretation_sha256) == 64
    assert len(adapted.adapter_input_sha256) == 64
    assert adapted.is_lossy
    assert {item.source_field for item in adapted.dispositions} == {
        "geometry",
        "terrain_identity",
        "snapshot_provenance",
        "source_frame_id",
        "surface_velocity_m_s",
        "frame_transform",
        "friction_coefficient",
        "firmness_pa",
        "material_scalars",
        "regional_topology",
    }
    assert {item.kind for item in adapted.dispositions} == {
        TerrainDispositionKind.EXACT,
        TerrainDispositionKind.EXPLICIT_INTERPRETATION,
        TerrainDispositionKind.LOCAL_LINEARIZATION,
        TerrainDispositionKind.NOT_REPRESENTED,
    }


@dataclass(frozen=True)
class _ExtendedSnapshot(UpstreamTerrainSnapshot):
    injected: str = "must-not-adapt"


def test_adapter_identity_binds_transform_interpretation_and_exact_snapshot_type() -> (
    None
):
    snapshot = _snapshot()
    translated = replace(_transform(), translation_m=(0.0, 1.0, 0.0))
    first = adapt_upstream_terrain_snapshot(snapshot, _transform(), _interpretation())
    second = adapt_upstream_terrain_snapshot(snapshot, translated, _interpretation())

    assert first.snapshot_sha256 == second.snapshot_sha256
    assert first.transform_sha256 != second.transform_sha256
    assert first.adapter_input_sha256 != second.adapter_input_sha256

    extended = _ExtendedSnapshot(**snapshot.to_dict())
    with pytest.raises(TypeError, match="exact"):
        extended.to_dict()
    with pytest.raises(TypeError, match="exact"):
        adapt_upstream_terrain_snapshot(extended, _transform(), _interpretation())


def test_adapter_output_rejects_forged_hashes_and_dispositions() -> None:
    adapted = adapt_upstream_terrain_snapshot(
        _snapshot(), _transform(), _interpretation()
    )

    with pytest.raises(ValueError, match="source_sha256"):
        replace(adapted, source_sha256="d" * 64)
    with pytest.raises(ValueError, match="adapter_input_sha256"):
        replace(adapted, adapter_input_sha256="e" * 64)
    with pytest.raises(ValueError, match="dispositions"):
        replace(adapted, dispositions=adapted.dispositions[:-1])
    with pytest.raises(TypeError, match="target_fields"):
        TerrainFieldDisposition(
            "field",
            TerrainDispositionKind.EXACT,
            "abc",  # type: ignore[arg-type]
            "invalid sequence input",
        )


def test_solver_surface_identity_is_injective_across_delimiter_text() -> None:
    first = replace(
        _snapshot(),
        terrain_id="a:b",
        material_id="c",
        terrain_revision="r|s",
        material_revision="t",
    )
    second = replace(
        _snapshot(),
        terrain_id="a",
        material_id="b:c",
        terrain_revision="r",
        material_revision="s|t",
    )

    first_surface = adapt_upstream_terrain_snapshot(
        first, _transform(), _interpretation()
    ).surface
    second_surface = adapt_upstream_terrain_snapshot(
        second, _transform(), _interpretation()
    ).surface

    assert first_surface.surface_id != second_surface.surface_id
    assert first_surface.provider_version != second_surface.provider_version
