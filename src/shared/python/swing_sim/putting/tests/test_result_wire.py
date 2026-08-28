"""``swing_sim.putting_result/2`` wire gates (#4800 P5).

Analytic/structural gates first — the capture margin is checked
against the closed-form Holmes/Penner radius, the break summary
against the retained samples — then the wire posture (byte-determinism,
fail-closed parsing) and the **supersede** contract: v2 refuses v1, the
v1 archive reader refuses v2, and neither migrates.

Mirrored test-for-test by ``web/src/model/puttingResultWire.test.ts``.
"""

from __future__ import annotations

import json
import math

import pytest

from shared.python.swing_sim.putting import (
    MINIMAL_PUTTERS,
    PUTTING_RESULT_FORMAT,
    PUTTING_RESULT_FORMAT_V1,
    PlanarGreenSurface,
    PuttingResultDocument,
    PuttingResultProvenance,
    PuttingResultV1Archive,
    effective_hole_radius_m,
    putting_result_document,
    putting_result_from_json,
    putting_result_to_json,
    putting_result_v1_archive_from_json,
    simulate_putt_on_surface,
    strike,
)

#: Every refusal in this package is a TypeError or a ValueError
#: (``shared.python.contracts.ContractViolationError`` subclasses
#: ``ValueError``), so the gates never assert a blind ``Exception``.
REFUSED = (TypeError, ValueError)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

BLADE = MINIMAL_PUTTERS["Blade Putter"]
FLAT = PlanarGreenSurface(grade_percent=0.0, aspect_deg=0.0)
BREAKING = PlanarGreenSurface(grade_percent=2.0, aspect_deg=90.0)

MINIMAL_PROVENANCE = PuttingResultProvenance(
    putter_source="minimal",
    putter_name="Blade Putter",
    stroke_source="declared",
    capture_model="effective_radius",
)


def _document(
    *,
    surface: PlanarGreenSurface = FLAT,
    speed_mps: float = 1.6,
    hole_distance_m: float = 3.0,
    face_angle_deg: float = 0.0,
    path_angle_deg: float = 0.0,
    aim_deg: float = 0.0,
) -> PuttingResultDocument:
    launch = strike(
        BLADE,
        speed_mps,
        aim_deg=aim_deg,
        face_angle_deg=face_angle_deg,
        path_angle_deg=path_angle_deg,
    )
    result = simulate_putt_on_surface(
        launch, surface, stimp_ft=10.0, hole_distance_m=hole_distance_m
    )
    return putting_result_document(
        launch, result, MINIMAL_PROVENANCE, hole_distance_m=hole_distance_m
    )


class TestDerivedFields:
    def test_capture_margin_is_the_closed_form_effective_radius(self) -> None:
        document = _document(surface=BREAKING)
        expected = (
            effective_hole_radius_m(document.speed_at_closest_mps)
            - document.closest_approach_m
        )
        assert document.capture_margin_m == pytest.approx(expected, rel=1e-15)
        assert document.effective_hole_radius_m == pytest.approx(
            effective_hole_radius_m(document.speed_at_closest_mps), rel=1e-15
        )

    def test_a_holed_putt_passed_inside_the_effective_hole(self) -> None:
        document = _document(speed_mps=1.6, hole_distance_m=3.0)
        assert document.holed
        assert document.capture_margin_m >= 0.0
        assert document.miss_distance_m is None
        assert document.margin_mps is not None

    def test_a_missed_putt_reports_how_much_hole_it_needed(self) -> None:
        document = _document(surface=BREAKING)
        assert not document.holed
        assert document.capture_margin_m < 0.0
        assert document.miss_distance_m is not None

    def test_a_straight_putt_on_a_flat_green_never_leaves_the_line(self) -> None:
        document = _document()
        assert document.apex_break_m == 0.0
        assert document.final_break_m == 0.0
        assert document.entry_azimuth_deg == pytest.approx(0.0, abs=1e-12)
        assert document.start_azimuth_deg == 0.0

    def test_the_apex_break_bounds_the_final_break(self) -> None:
        document = _document(surface=BREAKING)
        assert abs(document.apex_break_m) >= abs(document.final_break_m)
        assert document.apex_break_at_m >= 0.0

    def test_the_start_azimuth_is_carried_from_the_launch(self) -> None:
        launch = strike(BLADE, 1.6, aim_deg=1.5, face_angle_deg=0.5)
        result = simulate_putt_on_surface(
            launch, FLAT, stimp_ft=10.0, hole_distance_m=3.0
        )
        document = putting_result_document(
            launch, result, MINIMAL_PROVENANCE, hole_distance_m=3.0
        )
        assert document.start_azimuth_deg == launch.start_azimuth_deg
        assert document.sidespin_rad_s == launch.sidespin_rad_s
        # A start line right of the target line drifts right (y = left).
        assert document.final_break_m < 0.0

    def test_building_from_a_wrong_type_is_refused(self) -> None:
        launch = strike(BLADE, 1.6)
        with pytest.raises(TypeError):
            putting_result_document(
                launch, "not a result", MINIMAL_PROVENANCE, hole_distance_m=3.0
            )  # type: ignore[arg-type]


class TestProvenanceFailsClosed:
    def test_a_mesh_putter_must_carry_its_digest(self) -> None:
        with pytest.raises(REFUSED):
            PuttingResultProvenance(
                putter_source="mesh",
                putter_name="Milled",
                stroke_source="declared",
                capture_model="effective_radius",
            )

    def test_a_mesh_putter_must_not_carry_a_library_name(self) -> None:
        with pytest.raises(REFUSED):
            PuttingResultProvenance(
                putter_source="mesh",
                putter_name="Milled",
                stroke_source="declared",
                capture_model="effective_radius",
                putter_mesh_sha256="a" * 64,
                putter_library_name="Blade Putter",
            )

    def test_a_library_putter_must_name_its_library_entry(self) -> None:
        with pytest.raises(REFUSED):
            PuttingResultProvenance(
                putter_source="library",
                putter_name="Blade Putter",
                stroke_source="declared",
                capture_model="effective_radius",
            )

    def test_a_minimal_putter_carries_neither(self) -> None:
        with pytest.raises(REFUSED):
            PuttingResultProvenance(
                putter_source="minimal",
                putter_name="Blade Putter",
                stroke_source="declared",
                capture_model="effective_radius",
                putter_library_name="Blade Putter",
            )

    def test_an_imported_stroke_must_name_its_source(self) -> None:
        with pytest.raises(REFUSED):
            PuttingResultProvenance(
                putter_source="minimal",
                putter_name="Blade Putter",
                stroke_source="interchange",
                capture_model="effective_radius",
            )

    def test_a_declared_stroke_must_not_claim_an_import(self) -> None:
        with pytest.raises(REFUSED):
            PuttingResultProvenance(
                putter_source="minimal",
                putter_name="Blade Putter",
                stroke_source="declared",
                capture_model="effective_radius",
                stroke_source_id="mjcf-fixture",
            )

    def test_an_unknown_source_kind_is_refused(self) -> None:
        with pytest.raises(REFUSED):
            PuttingResultProvenance(
                putter_source="guessed",
                putter_name="Blade Putter",
                stroke_source="declared",
                capture_model="effective_radius",
            )


class TestWirePosture:
    def test_round_trip_is_byte_identical(self) -> None:
        text = putting_result_to_json(_document(surface=BREAKING))
        assert putting_result_to_json(putting_result_from_json(text)) == text

    def test_identical_putts_serialize_byte_identically(self) -> None:
        assert putting_result_to_json(_document()) == putting_result_to_json(
            _document()
        )

    def test_the_declared_format_is_v2(self) -> None:
        payload = json.loads(putting_result_to_json(_document()))
        assert payload["format"] == PUTTING_RESULT_FORMAT
        assert payload["provenance"]["kernel"] == "RK4-2ms-v1"

    def test_keys_are_sorted(self) -> None:
        text = putting_result_to_json(_document())
        keys = list(json.loads(text).keys())
        assert keys == sorted(keys)

    def test_an_unknown_field_is_refused(self) -> None:
        payload = json.loads(putting_result_to_json(_document()))
        payload["extra"] = 1
        with pytest.raises(REFUSED):
            putting_result_from_json(json.dumps(payload))

    def test_a_missing_field_is_refused(self) -> None:
        payload = json.loads(putting_result_to_json(_document()))
        del payload["launch"]["sidespin_rad_s"]
        with pytest.raises(REFUSED):
            putting_result_from_json(json.dumps(payload))

    def test_a_non_finite_value_is_refused(self) -> None:
        payload = json.loads(putting_result_to_json(_document()))
        payload["roll"]["final_break_m"] = math.nan
        with pytest.raises(REFUSED):
            putting_result_from_json(json.dumps(payload, allow_nan=True))

    def test_a_wrong_format_is_refused(self) -> None:
        payload = json.loads(putting_result_to_json(_document()))
        payload["format"] = "swing_sim.putting_result/3"
        with pytest.raises(REFUSED):
            putting_result_from_json(json.dumps(payload))

    def test_a_non_string_payload_is_refused(self) -> None:
        with pytest.raises(REFUSED):
            putting_result_from_json(b"{}")  # type: ignore[arg-type]


def _v1_payload() -> dict[str, object]:
    return {
        "format": PUTTING_RESULT_FORMAT_V1,
        "summary": {
            "skid_distance_m": 0.31,
            "total_distance_m": 3.02,
            "time_s": 4.1,
            "break_m": 0.0,
            "holed": False,
            "speed_at_hole_mps": None,
            "margin_mps": None,
            "miss_distance_m": 0.02,
        },
    }


class TestV2SupersedesV1:
    def test_v2_refuses_a_v1_payload_and_says_why(self) -> None:
        with pytest.raises(REFUSED, match="superseded"):
            putting_result_from_json(json.dumps(_v1_payload()))

    def test_the_v1_archive_reader_refuses_a_v2_payload(self) -> None:
        text = putting_result_to_json(_document())
        with pytest.raises(REFUSED, match="putting_result/2"):
            putting_result_v1_archive_from_json(text)

    def test_v1_reads_as_archive_evidence(self) -> None:
        archive = putting_result_v1_archive_from_json(json.dumps(_v1_payload()))
        assert isinstance(archive, PuttingResultV1Archive)
        assert not isinstance(archive, PuttingResultDocument)
        assert archive.miss_distance_m == pytest.approx(0.02)
        assert archive.margin_mps is None

    def test_the_archive_record_carries_no_2d_evidence(self) -> None:
        archive = putting_result_v1_archive_from_json(json.dumps(_v1_payload()))
        for name in ("start_azimuth_deg", "sidespin_rad_s", "capture_margin_m"):
            assert not hasattr(archive, name)

    def test_an_unknown_v1_field_is_refused(self) -> None:
        payload = _v1_payload()
        summary = payload["summary"]
        assert isinstance(summary, dict)
        summary["start_azimuth_deg"] = 0.0
        with pytest.raises(REFUSED):
            putting_result_v1_archive_from_json(json.dumps(payload))
