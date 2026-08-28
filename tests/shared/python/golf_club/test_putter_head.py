"""Putter-head import gates (epic #4800, P3).

Analytic gates first: every numeric assertion is a closed-form
consequence of the documented model (box inertia, the ``J r tau / 2I``
twist form, the P1 default-MOI fallback), never a pinned output of the
code under test.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from shared.python.contracts import PreconditionError
from shared.python.golf_club.putter_head import (
    PUTTER_CONTACT_TIME_S,
    PUTTER_HEAD_FORMAT,
    PutterHeadDocument,
    PutterHeadProvenance,
    head_moi_for_strike,
    putter_head_from_json,
    putter_head_from_library,
    putter_head_from_mesh,
    putter_head_from_stl,
    putter_head_to_json,
    putter_spec,
    strike_with_head,
    twist_response,
)
from shared.python.swing_sim.impact import GOLF_BALL_MASS_KG
from shared.python.swing_sim.putting import (
    DEFAULT_PUTTER_MOI_KG_M2,
    MINIMAL_PUTTERS,
    PutterSpec,
    strike,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SHA = "a" * 64

#: Float32-exact blade-like box: 31.25 mm deep, 31.25 mm tall,
#: 125 mm heel-to-toe (all coordinates are powers of two so the STL
#: float32 round-trip is bit-exact).
_BLADE_EXTENTS = (0.03125, 0.03125, 0.125)
_BLADE_MASS_KG = 0.35


def _box_mesh(
    extents: tuple[float, float, float], center: tuple[float, float, float]
) -> np.ndarray:
    """A watertight outward-wound rectangular box (C1 test idiom)."""
    hx, hy, hz = (e / 2.0 for e in extents)
    corners = np.asarray(center) + np.array(
        [[sx, sy, sz] for sx in (-hx, hx) for sy in (-hy, hy) for sz in (-hz, hz)]
    )
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    triangles = []
    for a, b, c, d in faces:
        triangles.append(corners[[a, b, c]])
        triangles.append(corners[[a, c, d]])
    return np.asarray(triangles, dtype=np.float64)


def _box_inertia_diag(
    extents: tuple[float, float, float], mass: float
) -> tuple[float, float, float]:
    ax, ay, az = extents
    return (
        mass / 12.0 * (ay**2 + az**2),
        mass / 12.0 * (ax**2 + az**2),
        mass / 12.0 * (ax**2 + ay**2),
    )


def _blade_document() -> PutterHeadDocument:
    return putter_head_from_mesh(
        "Milled Blade",
        _box_mesh(_BLADE_EXTENTS, (0.0, 0.0, 0.0)),
        mesh_sha256=_SHA,
        loft_deg=3.0,
        target_mass_kg=_BLADE_MASS_KG,
    )


def _library_document() -> PutterHeadDocument:
    spec = MINIMAL_PUTTERS["Blade Putter"]
    return putter_head_from_library(
        spec.name, head_mass_kg=spec.head_mass_kg, loft_deg=spec.loft_deg
    )


class TestProvenance:
    def test_mesh_requires_sha_and_exactly_one_selector(self) -> None:
        good = PutterHeadProvenance(
            source_kind="mesh", mesh_sha256=_SHA, density_kg_m3=8000.0
        )
        assert good.density_kg_m3 == 8000.0
        with pytest.raises(ValueError):
            PutterHeadProvenance(source_kind="mesh", density_kg_m3=8000.0)
        with pytest.raises(ValueError):
            PutterHeadProvenance(
                source_kind="mesh", mesh_sha256="ZZ" * 32, density_kg_m3=8000.0
            )
        with pytest.raises(ValueError):
            PutterHeadProvenance(source_kind="mesh", mesh_sha256=_SHA)
        with pytest.raises(ValueError):
            PutterHeadProvenance(
                source_kind="mesh",
                mesh_sha256=_SHA,
                density_kg_m3=8000.0,
                target_mass_kg=0.35,
            )
        with pytest.raises(ValueError):
            PutterHeadProvenance(
                source_kind="mesh",
                mesh_sha256=_SHA,
                density_kg_m3=8000.0,
                library_name="Blade Putter",
            )

    def test_library_requires_name_and_nothing_else(self) -> None:
        good = PutterHeadProvenance(source_kind="library", library_name="Blade Putter")
        assert good.library_name == "Blade Putter"
        with pytest.raises((TypeError, ValueError)):
            PutterHeadProvenance(source_kind="library")
        with pytest.raises(ValueError):
            PutterHeadProvenance(
                source_kind="library", library_name="Blade Putter", mesh_sha256=_SHA
            )

    def test_unknown_source_kind_refused(self) -> None:
        with pytest.raises(ValueError):
            PutterHeadProvenance(source_kind="guessed", library_name="x")


class TestMeshConstruction:
    def test_box_head_matches_closed_form_inertia(self) -> None:
        document = _blade_document()
        assert document.head_mass_kg == pytest.approx(_BLADE_MASS_KG, rel=1e-12)
        assert document.cg_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
        expected = _box_inertia_diag(_BLADE_EXTENTS, _BLADE_MASS_KG)
        tensor = np.asarray(document.inertia_at_cg_kg_m2)
        assert tensor == pytest.approx(np.diag(expected), rel=1e-12, abs=1e-18)

    def test_density_selector_sets_mass_from_volume(self) -> None:
        mesh = _box_mesh(_BLADE_EXTENTS, (0.0, 0.0, 0.0))
        density = 2800.0
        document = putter_head_from_mesh(
            "Cast Blade",
            mesh,
            mesh_sha256=_SHA,
            loft_deg=3.0,
            density_kg_m3=density,
        )
        volume = float(np.prod(_BLADE_EXTENTS))
        assert document.head_mass_kg == pytest.approx(density * volume, rel=1e-12)
        assert document.provenance.density_kg_m3 == density
        assert document.provenance.target_mass_kg is None

    def test_selector_is_exactly_one(self) -> None:
        mesh = _box_mesh(_BLADE_EXTENTS, (0.0, 0.0, 0.0))
        with pytest.raises(PreconditionError):
            putter_head_from_mesh("x", mesh, mesh_sha256=_SHA, loft_deg=3.0)
        with pytest.raises(PreconditionError):
            putter_head_from_mesh(
                "x",
                mesh,
                mesh_sha256=_SHA,
                loft_deg=3.0,
                density_kg_m3=8000.0,
                target_mass_kg=0.35,
            )

    def test_open_mesh_refused_by_the_c1_authority(self) -> None:
        mesh = _box_mesh(_BLADE_EXTENTS, (0.0, 0.0, 0.0))[:-1]
        with pytest.raises(PreconditionError):
            putter_head_from_mesh(
                "x", mesh, mesh_sha256=_SHA, loft_deg=3.0, target_mass_kg=0.35
            )

    def test_mesh_document_requires_mass_properties(self) -> None:
        with pytest.raises(ValueError):
            PutterHeadDocument(
                name="x",
                head_mass_kg=0.35,
                loft_deg=3.0,
                cor=0.78,
                provenance=PutterHeadProvenance(
                    source_kind="mesh", mesh_sha256=_SHA, target_mass_kg=0.35
                ),
            )

    def test_v1_spec_contracts_enforced_through_v2(self) -> None:
        mesh = _box_mesh(_BLADE_EXTENTS, (0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            putter_head_from_mesh(
                "x", mesh, mesh_sha256=_SHA, loft_deg=45.0, target_mass_kg=0.35
            )
        with pytest.raises(ValueError):
            putter_head_from_mesh(
                "x", mesh, mesh_sha256=_SHA, loft_deg=3.0, target_mass_kg=5.0
            )


class TestLibraryConstruction:
    def test_library_document_carries_no_tensor(self) -> None:
        document = _library_document()
        assert document.cg_m is None
        assert document.inertia_at_cg_kg_m2 is None
        assert document.provenance.source_kind == "library"
        assert document.provenance.library_name == "Blade Putter"

    def test_library_document_refuses_mass_properties(self) -> None:
        with pytest.raises(ValueError):
            PutterHeadDocument(
                name="Blade Putter",
                head_mass_kg=0.35,
                loft_deg=3.0,
                cor=0.78,
                provenance=PutterHeadProvenance(
                    source_kind="library", library_name="Blade Putter"
                ),
                inertia_at_cg_kg_m2=((1e-4, 0, 0), (0, 1e-4, 0), (0, 0, 1e-4)),
            )

    def test_putter_spec_recovers_the_v1_record(self) -> None:
        document = _library_document()
        assert putter_spec(document) == MINIMAL_PUTTERS["Blade Putter"]


class TestStlPath:
    @staticmethod
    def _binary_stl_bytes(triangles: np.ndarray) -> bytes:
        header = b"putter head gate".ljust(80, b"\0")
        blob = [header, struct.pack("<I", len(triangles))]
        for a, b, c in triangles:
            normal = np.cross(b - a, c - a)
            normal = normal / np.linalg.norm(normal)
            record = struct.pack("<12fH", *normal, *a, *b, *c, 0)
            blob.append(record)
        return b"".join(blob)

    def test_stl_round_trip_matches_from_mesh(self, tmp_path: Path) -> None:
        mesh = _box_mesh(_BLADE_EXTENTS, (0.0, 0.0, 0.0))
        payload = self._binary_stl_bytes(mesh)
        stl_path = tmp_path / "blade.stl"
        stl_path.write_bytes(payload)
        document = putter_head_from_stl(
            "Milled Blade", stl_path, loft_deg=3.0, target_mass_kg=_BLADE_MASS_KG
        )
        assert document.provenance.mesh_sha256 == hashlib.sha256(payload).hexdigest()
        # Coordinates are float32-exact, so the mass properties match
        # the float64 mesh bit-for-bit.
        reference = _blade_document()
        assert document.cg_m == reference.cg_m
        assert document.inertia_at_cg_kg_m2 == reference.inertia_at_cg_kg_m2
        assert document.head_mass_kg == reference.head_mass_kg

    def test_truncated_stl_refused(self, tmp_path: Path) -> None:
        stl_path = tmp_path / "bad.stl"
        stl_path.write_bytes(b"\0" * 40)
        with pytest.raises(ValueError):
            putter_head_from_stl(
                "x", stl_path, loft_deg=3.0, target_mass_kg=_BLADE_MASS_KG
            )


class TestWire:
    def test_mesh_round_trip_is_byte_deterministic(self) -> None:
        document = _blade_document()
        text = putter_head_to_json(document)
        parsed = putter_head_from_json(text)
        assert parsed == document
        assert putter_head_to_json(parsed) == text
        payload = json.loads(text)
        assert payload["format"] == PUTTER_HEAD_FORMAT
        assert list(payload) == sorted(payload)

    def test_library_round_trip_omits_mesh_fields(self) -> None:
        document = _library_document()
        text = putter_head_to_json(document)
        payload = json.loads(text)
        assert "cg_m" not in payload
        assert "inertia_at_cg_kg_m2" not in payload
        assert payload["provenance"] == {
            "source_kind": "library",
            "library_name": "Blade Putter",
        }
        assert putter_head_from_json(text) == document

    def test_unknown_fields_refused(self) -> None:
        text = putter_head_to_json(_library_document())
        payload = json.loads(text)
        payload["smoothing"] = True
        with pytest.raises(ValueError, match="unknown fields"):
            putter_head_from_json(json.dumps(payload))
        payload = json.loads(text)
        payload["provenance"]["vendor"] = "acme"
        with pytest.raises(ValueError, match="unknown fields"):
            putter_head_from_json(json.dumps(payload))

    def test_wrong_format_and_nonfinite_refused(self) -> None:
        payload = json.loads(putter_head_to_json(_library_document()))
        payload["format"] = "golf_club.putter_head/2"
        with pytest.raises(ValueError, match="format"):
            putter_head_from_json(json.dumps(payload))
        text = putter_head_to_json(_library_document()).replace(
            str(MINIMAL_PUTTERS["Blade Putter"].head_mass_kg), "Infinity"
        )
        with pytest.raises(ValueError):
            putter_head_from_json(text)
        with pytest.raises(TypeError):
            putter_head_from_json(b"{}")  # type: ignore[arg-type]


class TestHeadMoiForStrike:
    def test_library_fallback_returns_none(self) -> None:
        assert head_moi_for_strike(_library_document(), 10.0, 5.0) is None

    def test_single_axis_offsets_pick_the_matching_moment(self) -> None:
        document = _blade_document()
        _, moi_yy, moi_zz = _box_inertia_diag(_BLADE_EXTENTS, _BLADE_MASS_KG)
        assert head_moi_for_strike(document) == pytest.approx(moi_yy, rel=1e-12)
        assert head_moi_for_strike(document, 10.0, 0.0) == pytest.approx(
            moi_yy, rel=1e-12
        )
        assert head_moi_for_strike(document, 0.0, 8.0) == pytest.approx(
            moi_zz, rel=1e-12
        )

    def test_combined_offset_matches_directional_closed_form(self) -> None:
        document = _blade_document()
        _, moi_yy, moi_zz = _box_inertia_diag(_BLADE_EXTENTS, _BLADE_MASS_KG)
        r_t, r_h = 10.0e-3, 6.0e-3
        expected = (r_t**2 + r_h**2) / (r_t**2 / moi_yy + r_h**2 / moi_zz)
        assert head_moi_for_strike(document, 10.0, 6.0) == pytest.approx(
            expected, rel=1e-12
        )

    def test_scalar_feeds_p1_hook_exactly(self) -> None:
        document = _blade_document()
        result = strike_with_head(document, 2.0, strike_offset_toe_mm=10.0)
        expected = strike(
            putter_spec(document),
            2.0,
            strike_offset_toe_mm=10.0,
            head_moi_kg_m2=head_moi_for_strike(document, 10.0, 0.0),
        )
        assert result.launch == expected


class TestTwistGates:
    """Analytic gates for the quasi-static twist (written first, TDD)."""

    def test_symmetric_head_center_strike_has_zero_twist(self) -> None:
        twist = twist_response(_blade_document(), 2.0)
        assert twist.face_twist_open_deg == 0.0
        assert twist.loft_twist_add_deg == 0.0

    def test_twist_sign_flips_toe_vs_heel_and_high_vs_low(self) -> None:
        document = _blade_document()
        toe = twist_response(document, 2.0, strike_offset_toe_mm=10.0)
        heel = twist_response(document, 2.0, strike_offset_toe_mm=-10.0)
        assert toe.face_twist_open_deg > 0.0  # toe strike opens the face
        assert heel.face_twist_open_deg == -toe.face_twist_open_deg
        high = twist_response(document, 2.0, strike_offset_high_mm=6.0)
        low = twist_response(document, 2.0, strike_offset_high_mm=-6.0)
        assert high.loft_twist_add_deg > 0.0  # high strike adds loft
        assert low.loft_twist_add_deg == -high.loft_twist_add_deg

    def test_twist_matches_offset_impulse_over_moi_closed_form(self) -> None:
        document = _blade_document()
        _, moi_yy, _ = _box_inertia_diag(_BLADE_EXTENTS, _BLADE_MASS_KG)
        speed, toe_mm = 2.0, 10.0
        twist = twist_response(document, speed, strike_offset_toe_mm=toe_mm)
        r_t = toe_mm * 1e-3
        mass_eff = 1.0 / (1.0 / _BLADE_MASS_KG + r_t**2 / moi_yy)
        reduced = mass_eff * GOLF_BALL_MASS_KG / (mass_eff + GOLF_BALL_MASS_KG)
        impulse = (1.0 + 0.78) * reduced * speed * math.cos(math.radians(3.0))
        expected = math.degrees(impulse * r_t / moi_yy * PUTTER_CONTACT_TIME_S / 2.0)
        assert twist.normal_impulse_n_s == pytest.approx(impulse, rel=1e-12)
        assert twist.face_twist_open_deg == pytest.approx(expected, rel=1e-12)

    def test_higher_moi_head_twists_less(self) -> None:
        blade = _blade_document()
        mallet = putter_head_from_mesh(
            "Deep Mallet",
            _box_mesh((0.125, 0.03125, 0.125), (0.0, 0.0, 0.0)),
            mesh_sha256=_SHA,
            loft_deg=3.0,
            target_mass_kg=0.36,
        )
        blade_twist = twist_response(blade, 2.0, strike_offset_toe_mm=10.0)
        mallet_twist = twist_response(mallet, 2.0, strike_offset_toe_mm=10.0)
        assert 0.0 < mallet_twist.face_twist_open_deg < blade_twist.face_twist_open_deg

    def test_library_fallback_uses_the_catalogue_default(self) -> None:
        document = _library_document()
        speed, toe_mm = 2.0, 10.0
        twist = twist_response(document, speed, strike_offset_toe_mm=toe_mm)
        r_t = toe_mm * 1e-3
        mass = document.head_mass_kg
        mass_eff = 1.0 / (1.0 / mass + r_t**2 / DEFAULT_PUTTER_MOI_KG_M2)
        reduced = mass_eff * GOLF_BALL_MASS_KG / (mass_eff + GOLF_BALL_MASS_KG)
        impulse = (1.0 + 0.78) * reduced * speed * math.cos(math.radians(3.0))
        expected = math.degrees(
            impulse * r_t / DEFAULT_PUTTER_MOI_KG_M2 * PUTTER_CONTACT_TIME_S / 2.0
        )
        assert twist.head_moi_kg_m2 is None
        assert twist.face_twist_open_deg == pytest.approx(expected, rel=1e-12)

    def test_twist_is_small_at_putt_speeds(self) -> None:
        """Plausibility: fractions of a degree, not tens (see module docs)."""
        twist = twist_response(
            _blade_document(),
            3.0,
            strike_offset_toe_mm=20.0,
            strike_offset_high_mm=10.0,
        )
        assert abs(twist.face_twist_open_deg) < 1.0
        assert abs(twist.loft_twist_add_deg) < 1.0

    def test_rejects_out_of_range_inputs(self) -> None:
        document = _blade_document()
        with pytest.raises(ValueError):
            twist_response(document, 0.0)
        with pytest.raises(ValueError):
            twist_response(document, 2.0, shaft_lean_deg=20.0)
        with pytest.raises(ValueError):
            twist_response(document, 2.0, strike_offset_toe_mm=50.0)
        with pytest.raises(ValueError):
            twist_response(document, 2.0, strike_offset_high_mm=-30.0)


class TestLibraryFallbackReconciliation:
    def test_library_head_reproduces_p1_default_moi_behavior(self) -> None:
        """The resolved PutterSpec TODO: v2-from-library == v1 defaults."""
        spec = MINIMAL_PUTTERS["Blade Putter"]
        document = _library_document()
        for speed, toe, high in (
            (0.5, 0.0, 0.0),
            (1.8, 10.0, 0.0),
            (2.5, -8.0, 6.0),
            (3.2, 15.0, -5.0),
        ):
            via_head = strike_with_head(
                document,
                speed,
                strike_offset_toe_mm=toe,
                strike_offset_high_mm=high,
            ).launch
            via_v1_default = strike(
                spec,
                speed,
                strike_offset_toe_mm=toe,
                strike_offset_high_mm=high,
                head_moi_kg_m2=None,
            )
            assert via_head == via_v1_default

    def test_mesh_head_departs_from_the_default_off_center_only(self) -> None:
        document = _blade_document()
        spec_v1 = PutterSpec(
            name=document.name,
            head_mass_kg=document.head_mass_kg,
            loft_deg=document.loft_deg,
        )
        centered = strike_with_head(document, 2.0).launch
        assert centered == strike(spec_v1, 2.0)
        off_center = strike_with_head(document, 2.0, strike_offset_toe_mm=10.0).launch
        assert off_center != strike(spec_v1, 2.0, strike_offset_toe_mm=10.0)


class TestStrikeWithHead:
    def test_full_stroke_parameters_pass_through(self) -> None:
        document = _blade_document()
        result = strike_with_head(
            document,
            2.0,
            -1.0,
            aim_deg=2.0,
            face_angle_deg=1.0,
            path_angle_deg=-1.0,
            attack_angle_deg=1.5,
            strike_offset_toe_mm=8.0,
        )
        expected = strike(
            putter_spec(document),
            2.0,
            -1.0,
            aim_deg=2.0,
            face_angle_deg=1.0,
            path_angle_deg=-1.0,
            attack_angle_deg=1.5,
            strike_offset_toe_mm=8.0,
            head_moi_kg_m2=head_moi_for_strike(document, 8.0, 0.0),
        )
        assert result.launch == expected
        assert result.twist.face_twist_open_deg > 0.0
