"""Putting bridge, catalog, and reference-putt pins (#4125 H3).

The pinned reference putt is mirrored value-for-value by the vitest
parity suite in ``web/src/model/putting.test.ts`` (identical RK4 step
and constants on both sides).
"""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.glossary import FIELD_TO_TERM, GLOSSARY
from rate_of_closure.plotting import (
    PUTTING_CATALOG,
    extract_putting,
    putting_catalog_keys,
)
from rate_of_closure.putting import PUTT_EXPLANATIONS, putter_specs
from shared.python.swing_sim.putting import (
    GreenConditions,
    PutterSpec,
    simulate_putt,
    strike,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]


class TestPutterBridge:
    def test_library_putter_is_offered(self) -> None:
        specs = putter_specs()
        assert specs, "putter list must never be empty"
        assert "Putter" in specs  # the H1 club-library putter
        spec = specs["Putter"]
        assert isinstance(spec, PutterSpec)
        assert spec.head_mass_kg == pytest.approx(0.350)
        assert spec.loft_deg == pytest.approx(3.0)

    def test_every_row_field_has_explanation_and_glossary_term(self) -> None:
        for field in PUTT_EXPLANATIONS:
            assert len(PUTT_EXPLANATIONS[field]) > 80, field
            assert field in FIELD_TO_TERM, field
            assert FIELD_TO_TERM[field] in GLOSSARY, field


class TestPuttingCatalog:
    def test_keys_are_namespaced_and_stable(self) -> None:
        keys = putting_catalog_keys()
        assert all(key.startswith("putting.") for key in keys)
        assert set(keys) == {
            "putting.path_x",
            "putting.path_y",
            "putting.speed",
            "putting.time",
            "putting.rollout",
            "putting.skid_distance",
            "putting.skid_fraction",
            "putting.time_total",
            "putting.break",
            "putting.holed",
        }

    def test_extractors_return_declared_shapes(self) -> None:
        result = simulate_putt(
            strike(putter_specs()["Putter"], 1.8),
            GreenConditions(stimp_ft=10.0),
            3.0,
        )
        for key, spec in PUTTING_CATALOG.items():
            value = extract_putting(result, key)
            if spec.is_series:
                assert isinstance(value, np.ndarray), key
                assert np.all(np.isfinite(value)), key
            else:
                assert isinstance(value, float), key
                assert np.isfinite(value), key

    def test_unknown_key_is_rejected(self) -> None:
        result = simulate_putt(
            strike(putter_specs()["Putter"], 1.5),
            GreenConditions(stimp_ft=10.0),
            3.0,
        )
        with pytest.raises(ValueError):
            extract_putting(result, "putting.nope")


class TestReferencePuttPins:
    """Numeric pins mirrored by web/src/model/putting.test.ts."""

    def test_reference_launch(self) -> None:
        launch = strike(putter_specs()["Putter"], 1.8)
        assert launch.ball_speed_mps == pytest.approx(2.828565312464848, rel=1e-12)
        assert launch.launch_angle_deg == pytest.approx(3.5452147542505257, rel=1e-9)
        assert launch.horizontal_speed_mps == pytest.approx(
            2.8231523192738344, rel=1e-12
        )
        assert launch.spin_rad_s == pytest.approx(-3.153929533539754, rel=1e-12)

    def test_reference_breaking_putt(self) -> None:
        launch = strike(putter_specs()["Putter"], 1.8)
        green = GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=90.0)
        result = simulate_putt(launch, green, 3.0)
        assert not result.holed
        assert result.total_distance_m == pytest.approx(4.417405938785078, rel=1e-9)
        assert result.skid_distance_m == pytest.approx(0.5103817275162047, rel=1e-9)
        assert result.time_s == pytest.approx(4.388, abs=2e-3)
        assert result.break_m == pytest.approx(0.8176068791755766, rel=1e-9)
        assert result.miss_distance_m == pytest.approx(1.4994647284222105, rel=1e-9)

    def test_reference_holed_putt(self) -> None:
        launch = strike(putter_specs()["Putter"], 1.6)
        result = simulate_putt(launch, GreenConditions(stimp_ft=10.0), 3.0)
        assert result.holed
        assert result.speed_at_hole_mps == pytest.approx(0.5903262895096224, rel=1e-9)
        assert result.margin_mps == pytest.approx(0.2283133618862715, rel=1e-9)
