"""Tests for the unit system and the common literature closure metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rate_of_closure.model import ImpactScenario, closure_metrics, solve
from rate_of_closure.units import (
    DISTANCE_UNITS,
    FIELD_GUIDANCE,
    QUANTITY_UNITS,
    convert_from_canonical,
    convert_to_canonical,
    display_distance_unit,
    format_distance_m,
    set_display_distance_unit,
)

pytestmark = pytest.mark.unit

_SCENARIO = ImpactScenario(clubhead_speed_mph=120.0)


class TestUnitConversions:
    def test_round_trip_is_identity_for_every_unit(self) -> None:
        for quantity, table in QUANTITY_UNITS.items():
            for unit in table:
                out = convert_from_canonical(
                    quantity, unit, convert_to_canonical(quantity, unit, 123.456)
                )
                assert out == pytest.approx(123.456, rel=1e-12), (quantity, unit)

    def test_known_conversions(self) -> None:
        assert convert_to_canonical("speed", "m/s", 53.645) == pytest.approx(
            120.0, abs=0.01
        )
        assert convert_to_canonical("rotation", "rad/s", 1.0) == pytest.approx(
            math.degrees(1.0)
        )
        assert convert_to_canonical("rotation", "rpm", 350.0) == pytest.approx(2100.0)
        assert convert_to_canonical("length", "in", 1.0) == pytest.approx(25.4)

    def test_unknown_unit_is_rejected(self) -> None:
        with pytest.raises(Exception, match="unknown"):
            convert_to_canonical("speed", "furlongs", 1.0)

    def test_every_scenario_field_has_hover_guidance_with_a_source(self) -> None:
        from dataclasses import fields

        for field in fields(ImpactScenario):
            guidance = FIELD_GUIDANCE[field.name]
            assert "Suggested range" in guidance, field.name
            assert "Source:" in guidance, field.name

    def test_directional_inputs_document_their_reference_frame(self) -> None:
        for key in (
            "clubhead_speed_mph",
            "omega_plane_dps",
            "omega_shaft_dps",
            "plane_yaw_deg",
            "plane_side_tilt_deg",
            "plane_forward_tilt_deg",
        ):
            assert "Reference frame:" in FIELD_GUIDANCE[key], key


class TestClosureMetrics:
    def test_metrics_restate_the_solved_delivery(self) -> None:
        result = solve(_SCENARIO)
        metrics = closure_metrics(_SCENARIO)
        assert metrics.ccv_dps == pytest.approx(result.closure_rate_dps)
        assert metrics.closure_deg_per_ft == pytest.approx(
            result.normalized_closure_deg_per_ft
        )
        assert metrics.closure_deg_per_inch == pytest.approx(
            metrics.closure_deg_per_ft / 12.0
        )
        assert metrics.closure_deg_per_ms == pytest.approx(metrics.ccv_dps / 1000.0)

    def test_r_isa_reconstructs_the_path_gap(self) -> None:
        """d / R_ISA must reproduce the small-angle path deviation."""
        result = solve(_SCENARIO)
        metrics = closure_metrics(_SCENARIO)
        gap_deg = math.degrees(0.040 / metrics.r_isa_m)
        assert gap_deg == pytest.approx(abs(result.path_deviation_deg), rel=0.02)

    def test_r_isa_units_agree(self) -> None:
        metrics = closure_metrics(_SCENARIO)
        assert metrics.r_isa_ft == pytest.approx(metrics.r_isa_m / 0.3048)

    def test_time_to_square_is_inverse_ccv(self) -> None:
        metrics = closure_metrics(_SCENARIO)
        assert metrics.time_to_square_from_1deg_open_ms == pytest.approx(
            1000.0 / metrics.ccv_dps
        )
        # Tour-rate sanity: about half a millisecond per degree.
        assert 0.3 < metrics.time_to_square_from_1deg_open_ms < 0.8

    def test_non_closing_face_reports_infinite_ratios(self) -> None:
        metrics = closure_metrics(
            ImpactScenario(
                clubhead_speed_mph=120.0, omega_plane_dps=0.0, omega_shaft_dps=0.0
            )
        )
        assert math.isinf(metrics.r_isa_m)
        assert math.isinf(metrics.time_to_square_from_1deg_open_ms)
        assert metrics.toe_heel_speed_delta_mph == pytest.approx(0.0)

    def test_toe_heel_delta_matches_hand_calculation(self) -> None:
        """|omega x (L z_hat)| with L = 117 mm."""
        result = solve(_SCENARIO)
        metrics = closure_metrics(_SCENARIO)
        omega = np.radians(np.array(result.omega_dps))
        expected = float(np.linalg.norm(np.cross(omega, np.array([0.0, 0.0, 0.117]))))
        assert metrics.toe_heel_speed_delta_mph * 0.44704 == pytest.approx(
            expected, rel=1e-9
        )


class TestDistanceQuantity:
    """H6 (#4125): the ball-flight Distance quantity — yards default."""

    def test_default_display_unit_is_yards(self) -> None:
        assert display_distance_unit() == "yd"
        assert next(iter(DISTANCE_UNITS)) == "yd"
        assert QUANTITY_UNITS["distance"] is DISTANCE_UNITS

    def test_canonical_stays_si_metres(self) -> None:
        # Factor converts displayed -> canonical metres exactly.
        assert DISTANCE_UNITS["m"] == 1.0
        assert DISTANCE_UNITS["yd"] == pytest.approx(0.9144)
        assert convert_to_canonical("distance", "yd", 100.0) == pytest.approx(91.44)
        assert convert_from_canonical("distance", "yd", 91.44) == pytest.approx(100.0)

    def test_round_trip_is_exact_to_float(self) -> None:
        for unit in DISTANCE_UNITS:
            back = convert_from_canonical(
                "distance", unit, convert_to_canonical("distance", unit, 123.4)
            )
            assert back == pytest.approx(123.4, rel=1e-12)

    def test_format_follows_the_selected_unit(self) -> None:
        assert format_distance_m(91.44) == "100.0 yd"
        set_display_distance_unit("m")
        assert format_distance_m(91.44) == "91.4 m"

    def test_unknown_unit_rejected(self) -> None:
        with pytest.raises(ValueError):
            set_display_distance_unit("furlong")
