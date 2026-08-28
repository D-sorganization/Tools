"""Putting bridge, catalog, and reference-putt pins (#4125 H3).

The pinned reference putt is mirrored value-for-value by the vitest
parity suite in ``web/src/model/putting.test.ts`` (identical RK4 step
and constants on both sides).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rate_of_closure.glossary import FIELD_TO_TERM, GLOSSARY
from rate_of_closure.plotting import (
    PUTTING_CATALOG,
    extract_putting,
    putting_catalog_keys,
)
from rate_of_closure.putting import (
    PUTT_EXPLANATIONS,
    green_surface_from_document,
    putter_head_documents,
    putter_specs,
)
from shared.python.swing_sim.putting import (
    GREEN_SURFACE_FORMAT,
    GreenConditions,
    GridGreenSurface,
    PlanarGreenSurface,
    PutterSpec,
    green_surface_to_json,
    simulate_putt,
    strike,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]


class TestPutterBridge:
    def test_library_putter_is_offered(self) -> None:
        specs = putter_specs()
        assert specs, "putter list must never be empty"
        assert "Blade Putter" in specs  # the H1 club-library putters
        assert "Mallet Putter" in specs
        spec = specs["Blade Putter"]
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
            strike(putter_specs()["Blade Putter"], 1.8),
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
            strike(putter_specs()["Blade Putter"], 1.5),
            GreenConditions(stimp_ft=10.0),
            3.0,
        )
        with pytest.raises(ValueError):
            extract_putting(result, "putting.nope")


class TestReferencePuttPins:
    """Numeric pins mirrored by web/src/model/putting.test.ts."""

    def test_reference_launch(self) -> None:
        launch = strike(putter_specs()["Blade Putter"], 1.8)
        assert launch.ball_speed_mps == pytest.approx(2.828565312464848, rel=1e-12)
        assert launch.launch_angle_deg == pytest.approx(3.5452147542505257, rel=1e-9)
        assert launch.horizontal_speed_mps == pytest.approx(
            2.8231523192738344, rel=1e-12
        )
        assert launch.spin_rad_s == pytest.approx(-3.153929533539754, rel=1e-12)

    def test_reference_breaking_putt(self) -> None:
        launch = strike(putter_specs()["Blade Putter"], 1.8)
        green = GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=90.0)
        result = simulate_putt(launch, green, 3.0)
        assert not result.holed
        assert result.total_distance_m == pytest.approx(4.417405938785078, rel=1e-9)
        assert result.skid_distance_m == pytest.approx(0.5103817275162047, rel=1e-9)
        assert result.time_s == pytest.approx(4.388, abs=2e-3)
        assert result.break_m == pytest.approx(0.8176068791755766, rel=1e-9)
        assert result.miss_distance_m == pytest.approx(1.4994647284222105, rel=1e-9)

    def test_reference_holed_putt(self) -> None:
        launch = strike(putter_specs()["Blade Putter"], 1.6)
        result = simulate_putt(launch, GreenConditions(stimp_ft=10.0), 3.0)
        assert result.holed
        assert result.speed_at_hole_mps == pytest.approx(0.5903262895096224, rel=1e-9)
        assert result.margin_mps == pytest.approx(0.2283133618862715, rel=1e-9)


class TestPutterHeadBridge:
    """The P3 head documents the Qt tab actually solves with (#4800 P6)."""

    def test_library_heads_are_the_documented_no_mesh_fallback(self) -> None:
        heads = putter_head_documents()
        assert set(heads) == set(putter_specs())
        blade = heads["Blade Putter"]
        assert blade.provenance.source_kind == "library"
        assert blade.provenance.library_name == "Blade Putter"
        # The fallback deliberately carries no tensor, so P1 applies its
        # catalogue default and the results stay bit-identical.
        assert blade.cg_m is None
        assert blade.inertia_at_cg_kg_m2 is None

    def test_head_document_reproduces_the_v1_spec_exactly(self) -> None:
        from shared.python.golf_club.putter_head import putter_spec

        for name, spec in putter_specs().items():
            assert putter_spec(putter_head_documents()[name]) == spec


class TestGreenDocumentBridge:
    """Import dispatch is on the declared format, never on shape."""

    def test_tools_wire_is_read_by_the_tools_reader(self) -> None:
        surface = PlanarGreenSurface(grade_percent=2.0, aspect_deg=90.0)
        parsed, wire = green_surface_from_document(green_surface_to_json(surface))
        assert parsed == surface
        assert wire == GREEN_SURFACE_FORMAT

    def test_upstreamdrift_topography_is_read_by_the_p9_adapter(self) -> None:
        text = json.dumps(
            {
                "contours": [
                    {"x": x * 0.5, "y": y * 0.5, "elevation": -0.01 * x}
                    for y in range(4)
                    for x in range(4)
                ]
            }
        )
        parsed, wire = green_surface_from_document(text)
        assert isinstance(parsed, GridGreenSurface)
        assert parsed.spacing_m == pytest.approx(0.5)
        assert "upstreamdrift" in wire

    def test_a_declared_but_unknown_format_is_refused_not_guessed(self) -> None:
        """A wrong ``format`` never falls through to the UD reader."""
        with pytest.raises(ValueError):
            green_surface_from_document('{"format": "swing_sim.green_surface/9"}')

    def test_non_object_and_non_text_documents_are_refused(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_document("[]")
        with pytest.raises(TypeError):
            green_surface_from_document(b"{}")  # type: ignore[arg-type]
