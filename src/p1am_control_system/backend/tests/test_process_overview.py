"""F06 contracts for a reusable, synthetic process navigation model."""

from __future__ import annotations

import pytest
from process_overview import (
    AssetFaceplate,
    ProcessOverview,
    synthetic_process_overview,
)


def test_synthetic_overview_supports_progressive_multi_area_navigation() -> None:
    overview = synthetic_process_overview()

    assert overview.data_classification == "synthetic"
    assert overview.not_for_live_control is True
    assert [area.area_id for area in overview.areas] == [
        "SYNTHETIC.FEED",
        "SYNTHETIC.REACTOR",
        "SYNTHETIC.SEPARATION",
    ]
    assert all(area.assets for area in overview.areas)
    assert all(
        asset.detail_route.startswith("/operator/assets/")
        for area in overview.areas
        for asset in area.assets
    )
    assert all(asset.trend_tags for area in overview.areas for asset in area.assets)


def test_faceplate_exposes_consistent_operator_context() -> None:
    asset = synthetic_process_overview().areas[1].assets[0]

    assert asset.quality in {"good", "uncertain", "bad", "stale", "simulated"}
    assert asset.mode in {"off", "manual", "automatic", "unavailable"}
    assert asset.alarm_state in {"normal", "active", "shelved", "suppressed"}
    assert asset.interlock_state in {"clear", "permissive_missing", "tripped"}
    assert asset.primary_value.unit
    assert asset.primary_value.source_timestamp is not None


def test_overview_rejects_non_synthetic_identifiers() -> None:
    asset = synthetic_process_overview().areas[0].assets[0]

    with pytest.raises(ValueError, match="SYNTHETIC"):
        ProcessOverview(
            overview_id="PLANT.CONFIDENTIAL",
            title="Invalid",
            areas=synthetic_process_overview().areas,
            data_classification="synthetic",
            not_for_live_control=True,
        )

    with pytest.raises(ValueError, match="SYNTHETIC"):
        AssetFaceplate(**{**asset.model_dump(), "asset_id": "REAL.TAG"})
