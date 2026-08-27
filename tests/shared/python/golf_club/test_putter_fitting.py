"""Putter-fitting counterfactual gates (epic #4800, P5).

The load-bearing gate is closed form, not a pinned output: two putters
differing **only** in MOI, at a fixed strike-offset variance and a
fixed face-to-path mismatch, must separate by exactly the MOI ratio,
because P1's effective-mass law makes the offset-driven part of the
start line scale as ``1/I``::

    T = (1 + e) / (1 + m/M + m r^2 / I)
    start = aim + face + atan2((2/7) sin(fp), T cos(fp))

The rest gate the comparator's reuse (labels, held-fixed inputs,
refusals) and the report wire's determinism.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from shared.python.golf_club.fitting_engine import CounterfactualSpec
from shared.python.golf_club.putter_fitting import (
    PUTTER_FITTING_REPORT_FORMAT,
    PutterCounterfactual,
    compare_putter_counterfactuals,
    putter_fitting_report_to_json,
    scenario_for_head,
)
from shared.python.golf_club.putter_head import (
    PutterHeadDocument,
    head_moi_for_strike,
    putter_head_from_library,
    putter_head_from_mesh,
    putter_spec,
)
from shared.python.swing_sim.putting import (
    DEFAULT_PUTTER_MOI_KG_M2,
    MINIMAL_PUTTERS,
    PlanarGreenSurface,
    PuttingResultProvenance,
    putting_result_to_json,
)
from shared.python.swing_sim.putting.variation import (
    PUTT_STRIKE_TOE_KEY,
    PuttScenario,
    PuttStroke,
    PuttVariationPlan,
    evaluate_putt,
    run_putt_dispersion,
)
from shared.python.swing_sim.variation import NoiseSpec

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SHA = "b" * 64
_BLADE_EXTENTS = (0.03125, 0.03125, 0.125)
_BLADE_MASS_KG = 0.35

PROVENANCE = PuttingResultProvenance(
    putter_source="minimal",
    putter_name="placeholder",
    stroke_source="declared",
    capture_model="effective_radius",
)

STROKE = PuttStroke(clubhead_speed_mps=1.6, face_angle_deg=0.0, path_angle_deg=1.5)

PLAN = PuttVariationPlan(
    noise=(NoiseSpec(PUTT_STRIKE_TOE_KEY, "normal", 6.0),), n_runs=64, seed=11
)


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


def _mesh_head() -> PutterHeadDocument:
    return putter_head_from_mesh(
        "Milled Blade",
        _box_mesh(_BLADE_EXTENTS, (0.0, 0.0, 0.0)),
        mesh_sha256=_SHA,
        loft_deg=3.0,
        target_mass_kg=_BLADE_MASS_KG,
    )


def _library_head() -> PutterHeadDocument:
    spec = MINIMAL_PUTTERS["Blade Putter"]
    return putter_head_from_library(
        spec.name, head_mass_kg=spec.head_mass_kg, loft_deg=spec.loft_deg
    )


def _scenario() -> PuttScenario:
    return PuttScenario(
        scenario_id="p5-fitting",
        putter=MINIMAL_PUTTERS["Blade Putter"],
        stroke=STROKE,
        surface=PlanarGreenSurface(grade_percent=0.0, aspect_deg=0.0),
        stimp_ft=10.0,
        hole_distance_m=3.0,
        provenance=PROVENANCE,
    )


class TestScenarioRebinding:
    def test_the_head_supplies_the_putter_and_its_provenance(self) -> None:
        head = _mesh_head()
        bound = scenario_for_head(head, _scenario())
        assert bound.putter == putter_spec(head)
        assert bound.head_moi_kg_m2 == head_moi_for_strike(head)
        assert bound.provenance.putter_source == "mesh"
        assert bound.provenance.putter_name == "Milled Blade"
        assert bound.provenance.putter_mesh_sha256 == _SHA
        # The green, the hole, and the stroke are the scenario's.
        assert bound.stroke == STROKE
        assert bound.hole_distance_m == 3.0

    def test_a_library_head_keeps_p1s_catalogue_default_bit_for_bit(self) -> None:
        bound = scenario_for_head(_library_head(), _scenario())
        assert bound.head_moi_kg_m2 == DEFAULT_PUTTER_MOI_KG_M2
        assert bound.provenance.putter_source == "library"
        explicit = putting_result_to_json(evaluate_putt(bound))
        fallback = putting_result_to_json(
            evaluate_putt(
                PuttScenario(
                    scenario_id=bound.scenario_id,
                    putter=bound.putter,
                    stroke=bound.stroke,
                    surface=bound.surface,
                    stimp_ft=bound.stimp_ft,
                    hole_distance_m=bound.hole_distance_m,
                    provenance=bound.provenance,
                    head_moi_kg_m2=None,
                )
            )
        )
        assert explicit == fallback

    def test_wrong_types_are_refused(self) -> None:
        with pytest.raises(TypeError):
            scenario_for_head("nope", _scenario())  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            scenario_for_head(_mesh_head(), "nope")  # type: ignore[arg-type]


class TestMoiGate:
    def test_higher_moi_tightens_the_start_line_by_the_moi_ratio(self) -> None:
        report = compare_putter_counterfactuals(
            _mesh_head(),
            _scenario(),
            PLAN,
            (
                PutterCounterfactual(
                    spec=CounterfactualSpec(label="high-moi"), moi_scale=2.0
                ),
            ),
        )
        blade = report.baseline.summary.start_line_sigma_deg
        mallet = report.counterfactuals[0].summary.start_line_sigma_deg
        assert mallet < blade
        assert mallet / blade == pytest.approx(0.5, rel=1e-4)
        assert report.counterfactuals[0].head_moi_kg_m2 == pytest.approx(
            2.0 * report.baseline.head_moi_kg_m2, rel=1e-15
        )

    def test_lower_moi_loosens_the_start_line_by_the_moi_ratio(self) -> None:
        report = compare_putter_counterfactuals(
            _mesh_head(),
            _scenario(),
            PLAN,
            (
                PutterCounterfactual(
                    spec=CounterfactualSpec(label="low-moi"), moi_scale=0.5
                ),
            ),
        )
        ratio = (
            report.counterfactuals[0].summary.start_line_sigma_deg
            / report.baseline.summary.start_line_sigma_deg
        )
        assert ratio == pytest.approx(2.0, rel=1e-4)

    def test_the_baseline_is_the_plain_dispersion_study(self) -> None:
        head = _mesh_head()
        report = compare_putter_counterfactuals(
            head,
            _scenario(),
            PLAN,
            (
                PutterCounterfactual(
                    spec=CounterfactualSpec(label="high-moi"), moi_scale=2.0
                ),
            ),
        )
        direct, _ = run_putt_dispersion(scenario_for_head(head, _scenario()), PLAN)
        assert report.baseline.summary == direct.summary

    def test_the_mesh_moi_is_the_documents_twist_moment(self) -> None:
        head = _mesh_head()
        report = compare_putter_counterfactuals(head, _scenario(), PLAN, ())
        assert report.baseline.moi_source == "mesh"
        assert report.baseline.head_moi_kg_m2 == head_moi_for_strike(head)

    def test_a_library_head_reports_the_catalogue_default_source(self) -> None:
        report = compare_putter_counterfactuals(_library_head(), _scenario(), PLAN, ())
        assert report.baseline.moi_source == "catalogue_default"
        assert report.baseline.head_moi_kg_m2 == DEFAULT_PUTTER_MOI_KG_M2


class TestComparatorReuse:
    def test_counterfactuals_keep_their_requested_order_and_labels(self) -> None:
        report = compare_putter_counterfactuals(
            _mesh_head(),
            _scenario(),
            PLAN,
            (
                PutterCounterfactual(
                    spec=CounterfactualSpec(label="high-moi"), moi_scale=2.0
                ),
                PutterCounterfactual(
                    spec=CounterfactualSpec(label="heavier", head_mass_scale=1.1)
                ),
            ),
        )
        assert report.baseline.label == "baseline"
        assert [item.label for item in report.counterfactuals] == [
            "high-moi",
            "heavier",
        ]
        assert report.counterfactuals[1].head_mass_kg == pytest.approx(
            1.1 * report.baseline.head_mass_kg, rel=1e-15
        )

    def test_duplicate_labels_are_refused(self) -> None:
        with pytest.raises(ValueError):
            compare_putter_counterfactuals(
                _mesh_head(),
                _scenario(),
                PLAN,
                (
                    PutterCounterfactual(spec=CounterfactualSpec(label="same")),
                    PutterCounterfactual(
                        spec=CounterfactualSpec(label="same"), moi_scale=2.0
                    ),
                ),
            )

    def test_a_counterfactual_may_not_call_itself_the_baseline(self) -> None:
        with pytest.raises(ValueError):
            compare_putter_counterfactuals(
                _mesh_head(),
                _scenario(),
                PLAN,
                (PutterCounterfactual(spec=CounterfactualSpec(label="baseline")),),
            )

    def test_shaft_and_cg_counterfactuals_are_refused_not_ignored(self) -> None:
        for spec in (
            CounterfactualSpec(label="soft", ei_scale=0.8),
            CounterfactualSpec(label="torque", gj_scale=1.2),
            CounterfactualSpec(label="cg-back", cg_back_delta_m=0.005),
            CounterfactualSpec(label="cg-toe", cg_toe_delta_m=0.005),
        ):
            with pytest.raises(ValueError, match="refused rather than ignored"):
                PutterCounterfactual(spec=spec)

    def test_the_moi_scale_is_bounded(self) -> None:
        for scale in (0.1, 10.0):
            with pytest.raises(ValueError):
                PutterCounterfactual(
                    spec=CounterfactualSpec(label="wild"), moi_scale=scale
                )

    def test_wrong_argument_types_are_refused(self) -> None:
        with pytest.raises(TypeError):
            PutterCounterfactual(spec="nope")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            compare_putter_counterfactuals(
                _mesh_head(),
                _scenario(),
                PLAN,
                [PutterCounterfactual(spec=CounterfactualSpec(label="listed"))],
            )  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            compare_putter_counterfactuals(_mesh_head(), _scenario(), "nope", ())  # type: ignore[arg-type]


class TestReportWire:
    def _report_json(self) -> str:
        report = compare_putter_counterfactuals(
            _mesh_head(),
            _scenario(),
            PLAN,
            (
                PutterCounterfactual(
                    spec=CounterfactualSpec(label="high-moi"), moi_scale=2.0
                ),
            ),
        )
        return putter_fitting_report_to_json(report)

    def test_identical_runs_are_byte_identical(self) -> None:
        assert self._report_json() == self._report_json()

    def test_the_payload_declares_its_format_and_inputs(self) -> None:
        payload = json.loads(self._report_json())
        assert payload["format"] == PUTTER_FITTING_REPORT_FORMAT
        assert payload["scenario_id"] == "p5-fitting"
        assert payload["putter_name"] == "Milled Blade"
        assert payload["seed"] == PLAN.seed
        assert payload["n_runs"] == PLAN.n_runs
        assert payload["variables"] == [
            {
                "variable_key": PUTT_STRIKE_TOE_KEY,
                "distribution": "normal",
                "scale": 6.0,
            }
        ]

    def test_only_variants_carry_deltas_against_the_baseline(self) -> None:
        payload = json.loads(self._report_json())
        assert "deltas_vs_baseline" not in payload["baseline"]
        deltas = payload["counterfactuals"][0]["deltas_vs_baseline"]
        assert set(deltas) == {
            "make_percent",
            "leave_p50_m",
            "leave_p95_m",
            "start_line_sigma_deg",
        }
        assert deltas["start_line_sigma_deg"] < 0.0

    def test_keys_are_sorted(self) -> None:
        keys = list(json.loads(self._report_json()).keys())
        assert keys == sorted(keys)

    def test_a_wrong_report_type_is_refused(self) -> None:
        with pytest.raises(TypeError):
            putter_fitting_report_to_json("nope")  # type: ignore[arg-type]
