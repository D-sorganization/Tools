"""Adversarial tests for executable regional-ground job qualification."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.application.regional_ground_execution_qualification import (
    qualify_regional_plan_for_launch,
)
from shared.python.swing_sim.ball_setup import BallSetup, BallSupportMode
from shared.python.swing_sim.flight.tests._regional_ground_pipeline_support import (
    _launch,
    _plan,
    _settings,
)
from shared.python.swing_sim.ground import (
    REGIONAL_GROUND_EXECUTOR_SOURCE,
    GroundRegionalMaterialPlanRequest,
    GroundRegionalMaterialRegion,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    regional_plan_request_sha256,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _source_plan() -> GroundRegionalMaterialPlanRequest:
    plan = _plan()
    source_surface = _settings().surface
    overlays = tuple(
        GroundRegionalMaterialRegion(
            region.region_id,
            region.precedence,
            region.lower_coordinate_m,
            region.upper_coordinate_m,
            replace(region.surface, height_m=source_surface.height_m),
        )
        for region in plan.regions
    )
    return replace(
        plan,
        base_surface=source_surface,
        axis_origin_m=(0.0, source_surface.height_m, 0.0),
        regions=overlays,
    )


def test_teed_driver_translates_base_overlays_axis_and_rebinds_provenance() -> None:
    source = _source_plan()
    launch = replace(
        _launch(),
        ball_setup=BallSetup(BallSupportMode.TEE, 0.0381),
    )

    qualified = qualify_regional_plan_for_launch(
        source,
        launch,
        _settings().surface,
        source_revision=REGIONAL_GROUND_EXECUTOR_SOURCE,
    )

    expected_height = -launch.ball_radius - launch.ball_setup.tee_height_m
    assert qualified.base_surface.height_m == pytest.approx(expected_height)
    assert qualified.axis_origin_m == pytest.approx((0.0, expected_height, 0.0))
    assert all(
        region.surface.height_m == pytest.approx(expected_height)
        for region in qualified.regions
    )
    assert qualified.provenance.input_sha256 != source.provenance.input_sha256
    assert qualified.provenance.source_revision == REGIONAL_GROUND_EXECUTOR_SOURCE
    assert regional_plan_request_sha256(qualified) != regional_plan_request_sha256(
        source
    )


def test_qualification_rejects_non_authoritative_source_surface() -> None:
    source = _source_plan()
    changed = replace(source.base_surface, rolling_resistance=0.3)

    with pytest.raises(ValueError, match="source plan base surface"):
        qualify_regional_plan_for_launch(
            replace(source, base_surface=changed),
            _launch(),
            _settings().surface,
            source_revision="revision",
        )


def test_qualification_digest_covers_every_overlay() -> None:
    source = _source_plan()
    first = qualify_regional_plan_for_launch(
        source,
        _launch(),
        _settings().surface,
        source_revision="revision",
    )
    changed_region = replace(
        source.regions[0],
        surface=replace(source.regions[0].surface, rolling_resistance=0.3),
    )
    changed_source = replace(
        source,
        regions=(changed_region, *source.regions[1:]),
    )
    second = qualify_regional_plan_for_launch(
        changed_source,
        _launch(),
        _settings().surface,
        source_revision="revision",
    )

    assert second.provenance.input_sha256 != first.provenance.input_sha256
    assert regional_plan_request_sha256(second) != regional_plan_request_sha256(first)
