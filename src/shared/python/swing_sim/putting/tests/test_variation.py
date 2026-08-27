"""Monte-Carlo putting dispersion gates (#4800 P5).

Analytic gates first. The load-bearing ones:

* **Zero variance is the deterministic limit.** A plan with no declared
  distributions performs no sampling and every sample is
  ``putting_result/2``-byte-identical to the single deterministic putt.
* **Aim-only dispersion is the aim distribution.** With a square face
  and path the start line *is* the aim, so the reported start-line
  spread equals the sampled aim's spread exactly — which also proves
  the study draws from the canonical shared sampler and nothing else.
* **The start line follows P1's effective-mass law exactly.** Each run's
  start azimuth is checked against the closed form
  ``aim + face + atan2((2/7) sin(fp), T cos(fp))`` with
  ``T = (1+e)/(1 + m/M + m r^2/I)``, and the MOI gate checks the
  consequence: the offset-driven spread scales as ``1/I``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from shared.python.swing_sim.impact import GOLF_BALL_MASS_KG
from shared.python.swing_sim.putting import (
    MINIMAL_PUTTERS,
    PlanarGreenSurface,
    PuttingResultProvenance,
    putting_result_to_json,
)
from shared.python.swing_sim.putting.dispersion import (
    PUTT_DISPERSION_FORMAT,
    PuttDispersionSummary,
    PuttOutcome,
    putt_dispersion_from_json,
    putt_dispersion_to_json,
    summarize_putt_outcomes,
)
from shared.python.swing_sim.putting.variation import (
    CATEGORY_PUTTING,
    PUTT_AIM_KEY,
    PUTT_FACE_KEY,
    PUTT_PATH_KEY,
    PUTT_SPEED_KEY,
    PUTT_STRIKE_TOE_KEY,
    PuttScenario,
    PuttStroke,
    PuttVariationPlan,
    evaluate_putt,
    putt_outcome,
    run_putt_dispersion,
)
from shared.python.swing_sim.variation import NoiseSpec, variable_registry
from shared.python.swing_sim.variation.analysis import (
    finite_sample_standard_deviation,
)
from shared.python.swing_sim.variation.sampling import sample_inputs

#: Every refusal in this package is a TypeError or a ValueError
#: (``shared.python.contracts.ContractViolationError`` subclasses
#: ``ValueError``), so the gates never assert a blind ``Exception``.
REFUSED = (TypeError, ValueError)

pytestmark = [pytest.mark.unit, pytest.mark.physics]

BLADE = MINIMAL_PUTTERS["Blade Putter"]
FLAT = PlanarGreenSurface(grade_percent=0.0, aspect_deg=0.0)
ROLLING_CAP = 2.0 / 7.0

PROVENANCE = PuttingResultProvenance(
    putter_source="minimal",
    putter_name="Blade Putter",
    stroke_source="declared",
    capture_model="effective_radius",
)


def _scenario(
    *,
    stroke: PuttStroke | None = None,
    head_moi_kg_m2: float | None = None,
    surface: PlanarGreenSurface = FLAT,
) -> PuttScenario:
    return PuttScenario(
        scenario_id="p5-gate",
        putter=BLADE,
        stroke=stroke or PuttStroke(clubhead_speed_mps=1.6),
        surface=surface,
        stimp_ft=10.0,
        hole_distance_m=3.0,
        provenance=PROVENANCE,
        head_moi_kg_m2=head_moi_kg_m2,
    )


def _closed_form_start_deg(
    aim_deg: float, face_deg: float, path_deg: float, toe_mm: float, moi: float
) -> float:
    """P1's effective-mass start line (module docstring)."""
    radius_m = abs(toe_mm) * 1e-3
    head_mass_eff = (
        BLADE.head_mass_kg
        if radius_m == 0.0
        else 1.0 / (1.0 / BLADE.head_mass_kg + radius_m**2 / moi)
    )
    transfer = (1.0 + BLADE.cor) * head_mass_eff / (head_mass_eff + GOLF_BALL_MASS_KG)
    face_to_path = math.radians(path_deg - face_deg)
    deflection = math.atan2(
        ROLLING_CAP * math.sin(face_to_path), transfer * math.cos(face_to_path)
    )
    return aim_deg + face_deg + math.degrees(deflection)


class TestRegisteredVocabulary:
    def test_the_five_putting_variables_are_registered(self) -> None:
        registry = variable_registry()
        keys = (
            PUTT_SPEED_KEY,
            PUTT_FACE_KEY,
            PUTT_PATH_KEY,
            PUTT_AIM_KEY,
            PUTT_STRIKE_TOE_KEY,
        )
        for key in keys:
            assert key in registry
            assert registry[key].category == CATEGORY_PUTTING
            assert registry[key].typical_scale > 0.0

    def test_the_declared_scales_are_labelled_illustrative(self) -> None:
        registry = variable_registry()
        assert "Illustrative" in registry[PUTT_AIM_KEY].guidance


class TestDeterministicLimit:
    def test_a_plan_with_no_variance_reproduces_the_putt_bit_identically(
        self,
    ) -> None:
        scenario = _scenario()
        expected = putting_result_to_json(evaluate_putt(scenario))
        _report, documents = run_putt_dispersion(
            scenario, PuttVariationPlan(n_runs=5, seed=0)
        )
        assert len(documents) == 5
        assert [putting_result_to_json(item) for item in documents] == [expected] * 5

    def test_a_zero_variance_study_has_zero_dispersion(self) -> None:
        report, _documents = run_putt_dispersion(
            _scenario(), PuttVariationPlan(n_runs=4, seed=3)
        )
        assert report.summary.start_line_sigma_deg == 0.0
        assert report.summary.total_distance_sigma_m == 0.0
        assert report.variables == ()

    def test_the_same_seed_gives_the_same_study(self) -> None:
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 0.6),), n_runs=32, seed=5
        )
        first, _ = run_putt_dispersion(_scenario(), plan)
        second, _ = run_putt_dispersion(_scenario(), plan)
        assert putt_dispersion_to_json(first) == putt_dispersion_to_json(second)

    def test_a_different_seed_gives_a_different_study(self) -> None:
        base = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 0.6),), n_runs=32, seed=5
        )
        other = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 0.6),), n_runs=32, seed=6
        )
        first, _ = run_putt_dispersion(_scenario(), base)
        second, _ = run_putt_dispersion(_scenario(), other)
        assert first.summary.start_line_mean_deg != second.summary.start_line_mean_deg


class TestSharedSamplerIsTheOnlySource:
    def test_aim_only_dispersion_equals_the_sampled_aim_dispersion(self) -> None:
        scenario = _scenario()
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 0.75),), n_runs=64, seed=17
        )
        report, documents = run_putt_dispersion(scenario, plan)
        based = PuttVariationPlan(
            noise=plan.noise,
            base_variables=scenario.stroke.base_values(),
            n_runs=plan.n_runs,
            seed=plan.seed,
        )
        aims = np.asarray(sample_inputs(based), dtype=float)[:, 0]
        # A square face and path leave the start line equal to the aim.
        assert [item.start_azimuth_deg for item in documents] == list(aims)
        assert report.summary.start_line_sigma_deg == (
            finite_sample_standard_deviation(aims)
        )

    def test_a_wider_declared_scale_widens_the_start_line(self) -> None:
        narrow = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 0.25),), n_runs=64, seed=2
        )
        wide = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 1.0),), n_runs=64, seed=2
        )
        tight, _ = run_putt_dispersion(_scenario(), narrow)
        loose, _ = run_putt_dispersion(_scenario(), wide)
        assert loose.summary.start_line_sigma_deg > (tight.summary.start_line_sigma_deg)

    def test_declared_truncation_bounds_are_respected(self) -> None:
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 3.0, lower=-1.0, upper=1.0),),
            n_runs=48,
            seed=9,
        )
        _report, documents = run_putt_dispersion(_scenario(), plan)
        assert all(abs(item.start_azimuth_deg) <= 1.0 for item in documents)


class TestEffectiveMassLaw:
    def test_each_run_matches_the_closed_form_start_line(self) -> None:
        moi = 4.5e-4
        stroke = PuttStroke(
            clubhead_speed_mps=1.6, face_angle_deg=0.0, path_angle_deg=1.5
        )
        scenario = _scenario(stroke=stroke, head_moi_kg_m2=moi)
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_STRIKE_TOE_KEY, "normal", 6.0),), n_runs=48, seed=11
        )
        _report, documents = run_putt_dispersion(scenario, plan)
        based = PuttVariationPlan(
            noise=plan.noise,
            base_variables=stroke.base_values(),
            n_runs=plan.n_runs,
            seed=plan.seed,
        )
        offsets = np.asarray(sample_inputs(based), dtype=float)[:, 0]
        for document, toe_mm in zip(documents, offsets, strict=True):
            assert document.start_azimuth_deg == pytest.approx(
                _closed_form_start_deg(0.0, 0.0, 1.5, float(toe_mm), moi), rel=1e-12
            )

    def test_higher_moi_tightens_the_start_line_by_the_moi_ratio(self) -> None:
        stroke = PuttStroke(
            clubhead_speed_mps=1.6, face_angle_deg=0.0, path_angle_deg=1.5
        )
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_STRIKE_TOE_KEY, "normal", 6.0),), n_runs=120, seed=11
        )
        low_moi, high_moi = 4.5e-4, 9.0e-4
        blade, _ = run_putt_dispersion(
            _scenario(stroke=stroke, head_moi_kg_m2=low_moi), plan
        )
        mallet, _ = run_putt_dispersion(
            _scenario(stroke=stroke, head_moi_kg_m2=high_moi), plan
        )
        assert mallet.summary.start_line_sigma_deg < (
            blade.summary.start_line_sigma_deg
        )
        ratio = mallet.summary.start_line_sigma_deg / blade.summary.start_line_sigma_deg
        assert ratio == pytest.approx(low_moi / high_moi, rel=1e-4)

    def test_a_square_stroke_makes_the_start_line_moi_free(self) -> None:
        # face == path: the tangential impulse has no lever, so the
        # effective-mass reduction cannot move the start line at all.
        stroke = PuttStroke(
            clubhead_speed_mps=1.6, face_angle_deg=1.0, path_angle_deg=1.0
        )
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_STRIKE_TOE_KEY, "normal", 6.0),), n_runs=32, seed=4
        )
        report, _ = run_putt_dispersion(
            _scenario(stroke=stroke, head_moi_kg_m2=4.5e-4), plan
        )
        assert report.summary.start_line_sigma_deg == 0.0


class TestFailsClosed:
    def test_a_non_putting_variable_is_refused(self) -> None:
        with pytest.raises(REFUSED):
            PuttVariationPlan(
                noise=(NoiseSpec("swing_sim.club.mass_kg", "normal", 1.0),)
            )

    def test_duplicate_variables_are_refused(self) -> None:
        with pytest.raises(REFUSED):
            PuttVariationPlan(
                noise=(
                    NoiseSpec(PUTT_AIM_KEY, "normal", 0.5),
                    NoiseSpec(PUTT_AIM_KEY, "normal", 0.6, spec_id="second"),
                )
            )

    def test_a_plan_must_run_at_least_once(self) -> None:
        with pytest.raises(REFUSED):
            PuttVariationPlan(n_runs=0)

    def test_a_negative_seed_is_refused(self) -> None:
        with pytest.raises(REFUSED):
            PuttVariationPlan(seed=-1)

    def test_the_plan_may_not_declare_a_base(self) -> None:
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 0.5),),
            base_variables={PUTT_AIM_KEY: 1.0},
            n_runs=4,
        )
        with pytest.raises(REFUSED, match="scenario is the plan's base"):
            run_putt_dispersion(_scenario(), plan)

    def test_a_draw_outside_the_model_envelope_raises(self) -> None:
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_SPEED_KEY, "normal", 5.0),), n_runs=64, seed=1
        )
        with pytest.raises(REFUSED):
            run_putt_dispersion(_scenario(), plan)

    def test_wrong_argument_types_are_refused(self) -> None:
        with pytest.raises(TypeError):
            run_putt_dispersion("nope", PuttVariationPlan())  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            run_putt_dispersion(_scenario(), "nope")  # type: ignore[arg-type]


class TestSummaryAndWire:
    def test_a_summary_needs_two_runs(self) -> None:
        outcome = PuttOutcome(
            holed=True,
            start_azimuth_deg=0.0,
            leave_distance_m=0.0,
            total_distance_m=3.0,
            break_m=0.0,
            capture_margin_m=0.01,
        )
        with pytest.raises(REFUSED):
            summarize_putt_outcomes((outcome,))

    def test_a_holed_putt_leaves_nothing(self) -> None:
        with pytest.raises(REFUSED):
            PuttOutcome(
                holed=True,
                start_azimuth_deg=0.0,
                leave_distance_m=0.4,
                total_distance_m=3.0,
                break_m=0.0,
                capture_margin_m=0.01,
            )

    def test_make_percent_is_the_holed_fraction(self) -> None:
        scenario = _scenario()
        plan = PuttVariationPlan(
            noise=(NoiseSpec(PUTT_AIM_KEY, "normal", 1.2),), n_runs=40, seed=21
        )
        report, documents = run_putt_dispersion(scenario, plan)
        holed = sum(1 for item in documents if item.holed)
        assert report.summary.holed_count == holed
        assert report.summary.make_percent == pytest.approx(100.0 * holed / 40)
        assert 0.0 <= report.summary.make_percent <= 100.0
        outcomes = tuple(putt_outcome(item) for item in documents)
        assert all(item.leave_distance_m == 0.0 for item in outcomes if item.holed)

    def test_the_wire_round_trips_byte_identically(self) -> None:
        report, _ = run_putt_dispersion(
            _scenario(),
            PuttVariationPlan(
                noise=(
                    NoiseSpec(PUTT_AIM_KEY, "normal", 0.6),
                    NoiseSpec(PUTT_FACE_KEY, "normal", 0.4),
                ),
                n_runs=16,
                seed=8,
            ),
        )
        text = putt_dispersion_to_json(report)
        assert putt_dispersion_to_json(putt_dispersion_from_json(text)) == text
        payload = json.loads(text)
        assert payload["format"] == PUTT_DISPERSION_FORMAT
        assert [item["variable_key"] for item in payload["variables"]] == [
            PUTT_AIM_KEY,
            PUTT_FACE_KEY,
        ]

    def test_an_unknown_wire_field_is_refused(self) -> None:
        report, _ = run_putt_dispersion(_scenario(), PuttVariationPlan(n_runs=2))
        payload = json.loads(putt_dispersion_to_json(report))
        payload["extra"] = 1
        with pytest.raises(REFUSED):
            putt_dispersion_from_json(json.dumps(payload))

    def test_a_non_finite_summary_value_is_refused(self) -> None:
        with pytest.raises(REFUSED):
            PuttDispersionSummary(
                n_runs=2,
                holed_count=0,
                make_percent=0.0,
                leave_mean_m=math.nan,
                leave_p50_m=0.0,
                leave_p95_m=0.0,
                leave_max_m=0.0,
                start_line_mean_deg=0.0,
                start_line_sigma_deg=0.0,
                start_line_p05_deg=0.0,
                start_line_p95_deg=0.0,
                total_distance_mean_m=0.0,
                total_distance_sigma_m=0.0,
            )

    def test_more_holed_than_run_is_refused(self) -> None:
        with pytest.raises(REFUSED):
            PuttDispersionSummary(
                n_runs=2,
                holed_count=3,
                make_percent=100.0,
                leave_mean_m=0.0,
                leave_p50_m=0.0,
                leave_p95_m=0.0,
                leave_max_m=0.0,
                start_line_mean_deg=0.0,
                start_line_sigma_deg=0.0,
                start_line_p05_deg=0.0,
                start_line_p95_deg=0.0,
                total_distance_mean_m=0.0,
                total_distance_sigma_m=0.0,
            )
