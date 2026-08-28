"""Putter-fitting counterfactuals (epic #4800, P5).

Through the fitting comparator, not beside it
---------------------------------------------
This module is the putting outcome function for
:func:`.fitting_engine.evaluate_counterfactual_set` — the comparator
:func:`.fitting_engine.compare_counterfactuals` itself runs on. The
bounded what-if is the shipped :class:`.fitting_engine.CounterfactualSpec`
verbatim, and the report keeps the comparator's shape: an identity, a
baseline, and per-variant outcomes carrying deltas against it.

The comparator's **held-fixed semantics** carry over unchanged: the
stroke, the green, the hole, and the plan's seed are declared inputs,
so a heavier or higher-MOI head is evaluated under exactly the swing
and read the golfer already has. Only the equipment moves.

What a putter counterfactual may change
---------------------------------------
:class:`PutterCounterfactual` wraps the shared spec and adds the one
knob putting needs and the full swing does not: ``moi_scale``, the
head's twist moment about its CG. It **refuses** the shared spec's
shaft and CG knobs (``cg_back_delta_m``, ``cg_toe_delta_m``,
``ei_scale``, ``gj_scale``) rather than accepting and ignoring them —
the putting chain has no shaft-delivery model, so an accepted-but-
ignored shaft counterfactual would report "no difference" for a
question that was never asked.

A counterfactual changes what the *strike* sees — head mass, loft, and
the scalar twist MOI feeding P1's
``strike(..., head_moi_kg_m2=...)`` hook. It deliberately does not
fabricate a modified mesh or a modified inertia tensor: the P3
``golf_club.putter_head/1`` document is the **baseline's** provenance
and is reported as such.

Outcome metrics
---------------
Each variant runs the P5 Monte-Carlo study
(:func:`~shared.python.swing_sim.putting.variation.run_putt_dispersion`)
under the same seeded plan, and reports make percentage, the leave
distribution, and start-line dispersion. Start-line dispersion is the
metric MOI acts on, through P1's effective-mass law::

    M_eff = 1 / (1/M + r^2 / I)      T = (1 + e) M_eff / (M_eff + m)
    start = aim + face + atan2((2/7) sin(path - face), T cos(path - face))

Expanding ``1/T`` gives the exact statement the gate checks::

    T = (1 + e) / (1 + m/M + m r^2 / I)

so the strike-offset-dependent part of the start line scales as
``1/I``: at a fixed strike-offset variance and a fixed face-to-path
mismatch, doubling the head's MOI halves the start-line spread the
offset contributes. Higher MOI is tighter, and by a computable
amount — not merely "in the right direction".

Baseline MOI comes from :func:`.putter_head.head_moi_for_strike`, which
for a toe-only strike offset is exactly ``I_yy``. A library-fallback
head carries no tensor, so the comparison starts from P1's documented
catalogue default and the report says so (``moi_source``).

The report serializes to deterministic sorted-keys JSON
(``golf_club.putter_fitting_report/1``), so two identical runs are
byte-identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from shared.python.swing_sim.putting.dispersion import (
    PuttDispersionSummary,
    PuttVariableDeclaration,
)
from shared.python.swing_sim.putting.impact import (
    DEFAULT_PUTTER_MOI_KG_M2,
    PutterSpec,
)
from shared.python.swing_sim.putting.variation import (
    PuttScenario,
    PuttVariationPlan,
    run_putt_dispersion,
)

from ._validation import require_finite_float, require_identifier
from .fitting_engine import CounterfactualSpec, evaluate_counterfactual_set
from .putter_head import PutterHeadDocument, head_moi_for_strike, putter_spec

PUTTER_FITTING_REPORT_FORMAT = "golf_club.putter_fitting_report/1"

#: MOI counterfactual bounds. Published putter MOIs span roughly
#: 3800-7000 g cm^2 (blade to high-MOI mallet), a factor under two;
#: [0.25, 4] leaves generous room around that without letting a sweep
#: wander into heads that do not exist.
_MIN_MOI_SCALE = 0.25
_MAX_MOI_SCALE = 4.0

_SUMMARY_METRICS = (
    "make_percent",
    "leave_mean_m",
    "leave_p50_m",
    "leave_p95_m",
    "leave_max_m",
    "start_line_mean_deg",
    "start_line_sigma_deg",
)

_DELTA_METRICS = (
    "make_percent",
    "leave_p50_m",
    "leave_p95_m",
    "start_line_sigma_deg",
)

__all__ = [
    "PUTTER_FITTING_REPORT_FORMAT",
    "PutterCounterfactual",
    "PutterFittingReport",
    "PuttingOutcome",
    "compare_putter_counterfactuals",
    "putter_fitting_report_to_json",
    "scenario_for_head",
]


@dataclass(frozen=True)
class PutterCounterfactual:
    """One bounded putter what-if (module docstring).

    Attributes:
        spec: The shared :class:`.fitting_engine.CounterfactualSpec` —
            label, ``head_mass_scale``, and ``loft_delta_deg`` are the
            knobs the putting chain models. Its shaft and CG knobs must
            stay neutral; a non-neutral one is refused, never ignored.
        moi_scale: Multiplier on the head's twist MOI about its CG.
    """

    spec: CounterfactualSpec
    moi_scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.spec, CounterfactualSpec):
            raise TypeError("spec must be a CounterfactualSpec")
        object.__setattr__(
            self, "moi_scale", require_finite_float(self.moi_scale, "moi_scale")
        )
        if not _MIN_MOI_SCALE <= self.moi_scale <= _MAX_MOI_SCALE:
            raise ValueError(
                f"moi_scale must lie in [{_MIN_MOI_SCALE}, {_MAX_MOI_SCALE}]"
            )
        neutral = (
            self.spec.cg_back_delta_m == 0.0
            and self.spec.cg_toe_delta_m == 0.0
            and self.spec.ei_scale == 1.0
            and self.spec.gj_scale == 1.0
        )
        if not neutral:
            raise ValueError(
                "the putting chain models no shaft delivery or head CG offset; "
                "cg/ei/gj counterfactuals are refused rather than ignored"
            )


@dataclass(frozen=True)
class PuttingOutcome:
    """What one putter delivered over the shared seeded plan."""

    label: str
    head_mass_kg: float
    loft_deg: float
    head_moi_kg_m2: float
    moi_source: str
    summary: PuttDispersionSummary


@dataclass(frozen=True)
class PutterFittingReport:
    """Baseline plus per-counterfactual outcomes, all identity-carrying."""

    scenario_id: str
    putter_name: str
    seed: int
    n_runs: int
    variables: tuple[PuttVariableDeclaration, ...]
    baseline: PuttingOutcome
    counterfactuals: tuple[PuttingOutcome, ...]


def _baseline_moi(head: PutterHeadDocument) -> tuple[float, str]:
    """The head's twist MOI and where it came from (module docstring)."""
    measured = head_moi_for_strike(head)
    if measured is None:
        return DEFAULT_PUTTER_MOI_KG_M2, "catalogue_default"
    return measured, "mesh"


def scenario_for_head(head: PutterHeadDocument, scenario: PuttScenario) -> PuttScenario:
    """Rebind a scenario's putter to a P3 head document.

    The scenario supplies the green, the hole, and the nominal stroke;
    the head supplies the putter spec and its twist MOI, so the two can
    never disagree about which putter was tested.

    Args:
        head: The P3 ``golf_club.putter_head/1`` document.
        scenario: The putt to rebind.

    Returns:
        The rebound :class:`~...putting.variation.PuttScenario`.

    Raises:
        TypeError: If either argument is the wrong type.
    """
    if not isinstance(head, PutterHeadDocument):
        raise TypeError("head must be a PutterHeadDocument")
    if not isinstance(scenario, PuttScenario):
        raise TypeError("scenario must be a PuttScenario")
    moi, _source = _baseline_moi(head)
    return replace(
        scenario,
        putter=putter_spec(head),
        head_moi_kg_m2=moi,
        provenance=replace(
            scenario.provenance,
            putter_source=head.provenance.source_kind,
            putter_name=head.name,
            putter_mesh_sha256=head.provenance.mesh_sha256,
            putter_library_name=head.provenance.library_name,
        ),
    )


def _variant_scenario(
    base: PuttScenario,
    baseline_spec: PutterSpec,
    baseline_moi_kg_m2: float,
    counterfactual: PutterCounterfactual | None,
) -> tuple[PuttScenario, PutterSpec, float]:
    """Apply one counterfactual to the rebound baseline scenario."""
    if counterfactual is None:
        return base, baseline_spec, baseline_moi_kg_m2
    spec = replace(
        baseline_spec,
        head_mass_kg=baseline_spec.head_mass_kg * counterfactual.spec.head_mass_scale,
        loft_deg=baseline_spec.loft_deg + counterfactual.spec.loft_delta_deg,
    )
    moi = baseline_moi_kg_m2 * counterfactual.moi_scale
    return replace(base, putter=spec, head_moi_kg_m2=moi), spec, moi


def compare_putter_counterfactuals(
    head: PutterHeadDocument,
    scenario: PuttScenario,
    plan: PuttVariationPlan,
    counterfactuals: tuple[PutterCounterfactual, ...],
) -> PutterFittingReport:
    """Compare putters over one seeded stroke/read study.

    Every variant runs the *same* plan against the *same* scenario, so
    the only thing that differs between outcomes is the putter.

    Args:
        head: The baseline putter as a P3 document.
        scenario: The putt (green, hole, nominal stroke); its putter is
            rebound to ``head`` by :func:`scenario_for_head`.
        plan: The declared distributions, run count, and seed.
        counterfactuals: The bounded putter what-ifs.

    Returns:
        The :class:`PutterFittingReport`.

    Raises:
        TypeError: If any argument is the wrong type.
        ValueError: If labels collide or claim to be the baseline.
    """
    if not isinstance(counterfactuals, tuple) or not all(
        isinstance(item, PutterCounterfactual) for item in counterfactuals
    ):
        raise TypeError("counterfactuals must be a tuple of PutterCounterfactual")
    if not isinstance(plan, PuttVariationPlan):
        raise TypeError("plan must be a PuttVariationPlan")
    base = scenario_for_head(head, scenario)
    baseline_moi, moi_source = _baseline_moi(head)
    baseline_spec = base.putter
    by_label = {item.spec.label: item for item in counterfactuals}

    def evaluate(spec: CounterfactualSpec | None) -> PuttingOutcome:
        putter_case = None if spec is None else by_label[spec.label]
        variant, putter, moi = _variant_scenario(
            base, baseline_spec, baseline_moi, putter_case
        )
        report, _documents = run_putt_dispersion(variant, plan)
        return PuttingOutcome(
            label="baseline" if spec is None else spec.label,
            head_mass_kg=putter.head_mass_kg,
            loft_deg=putter.loft_deg,
            head_moi_kg_m2=moi,
            moi_source=moi_source,
            summary=report.summary,
        )

    baseline, variants = evaluate_counterfactual_set(
        tuple(item.spec for item in counterfactuals), evaluate
    )
    return PutterFittingReport(
        scenario_id=base.scenario_id,
        putter_name=head.name,
        seed=plan.seed,
        n_runs=plan.n_runs,
        variables=plan.declarations(),
        baseline=baseline,
        counterfactuals=variants,
    )


def _outcome_payload(
    outcome: PuttingOutcome, baseline: PuttingOutcome | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": outcome.label,
        "head_mass_kg": outcome.head_mass_kg,
        "loft_deg": outcome.loft_deg,
        "head_moi_kg_m2": outcome.head_moi_kg_m2,
        "moi_source": outcome.moi_source,
        "n_runs": outcome.summary.n_runs,
        "holed_count": outcome.summary.holed_count,
    }
    payload.update({name: getattr(outcome.summary, name) for name in _SUMMARY_METRICS})
    if baseline is not None:
        payload["deltas_vs_baseline"] = {
            name: getattr(outcome.summary, name) - getattr(baseline.summary, name)
            for name in _DELTA_METRICS
        }
    return payload


def putter_fitting_report_to_json(report: PutterFittingReport) -> str:
    """Serialize deterministically; identical runs are byte-identical."""
    if not isinstance(report, PutterFittingReport):
        raise TypeError("report must be PutterFittingReport")
    require_identifier(report.scenario_id, "scenario_id")
    payload = {
        "format": PUTTER_FITTING_REPORT_FORMAT,
        "scenario_id": report.scenario_id,
        "putter_name": report.putter_name,
        "seed": report.seed,
        "n_runs": report.n_runs,
        "variables": [
            {
                "variable_key": item.variable_key,
                "distribution": item.distribution,
                "scale": item.scale,
            }
            for item in report.variables
        ],
        "baseline": _outcome_payload(report.baseline, None),
        "counterfactuals": [
            _outcome_payload(outcome, report.baseline)
            for outcome in report.counterfactuals
        ],
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
