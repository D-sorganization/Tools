"""Monte-Carlo putting dispersion over stroke + green reading (P5, #4800).

Not a second sampler
--------------------
Sampling goes through the shared ``swing_sim.variation`` authority and
nowhere else: variables are declared in the shared namespaced registry
(:func:`~shared.python.swing_sim.variation.spec.register_variable`),
distributions are :class:`~...variation.spec.NoiseSpec` values, and the
draws come from :func:`~...variation.sampling.sample_inputs` — the
canonical seeded PCG64 stream with per-variable substreams. Nothing in
this module calls a random-number generator.

:class:`PuttVariationPlan` mirrors ``VariationPlan``'s *sampling-only*
shape rather than being one, exactly like
``golf_club.turf_variation.TurfVariationPlan`` and for the same reason:
putting variables are not legal inputs to any generic pipeline mode
(``delivery`` / ``swing`` / ``launch``), so a plan carrying them must
not be able to enter the generic evaluator.

**Import policy.** This module is *not* re-exported from the
``swing_sim.putting`` façade — importing it pulls the variation engine
(and therefore SciPy), which the rest of the putting package does not
need. Import it directly, the same policy ``swing_sim.variation``
itself declares.

What varies
-----------
Five declared variables under ``swing_sim.putting``:

* **Stroke** — ``clubhead_speed_mps`` (pace), ``face_angle_deg``,
  ``path_angle_deg`` (the two that split the start line per P1), and
  ``strike_offset_toe_mm`` (the strike-location variance a putter's
  MOI acts on).
* **Green reading** — ``aim_deg``, the golfer's read of the line. Aim
  is measured off the target line, so an aim error moves the start
  line without touching the stroke: exactly the decomposition a
  fitting study needs to keep equipment and read separated.

Registered defaults and scales are **illustrative** and carry that
guidance verbatim; a study that claims a golfer's real dispersion must
supply a calibrated distribution.

Fail-closed evaluation
----------------------
Every sampled run is evaluated through the shipped physics —
:func:`~.impact.strike` then
:func:`~.green.simulate_putt_on_surface` — and lands as a
``swing_sim.putting_result/2`` document (:mod:`.result_wire`), so a
Monte-Carlo sample and a hand-run putt are literally the same record.
A draw that leaves a model's validity envelope **raises**; it is not
clamped and not silently dropped. Declare
``NoiseSpec(lower=..., upper=...)`` truncation bounds — the shared
sampler clips them deterministically — when a distribution's tail
would leave the envelope.

Determinism
-----------
The plan's ``seed`` fixes the draws, so a study is reproducible run to
run and machine to machine. The degenerate plan — **no declared
distributions** — performs no sampling at all and evaluates the
nominal scenario ``n_runs`` times, so its every sample is
bit-identical to the deterministic single-putt result (a test gate).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import cast

import numpy as np

from shared.python.contracts import require, require_finite
from shared.python.swing_sim.variation import (
    NoiseSpec,
    PerturbationGroup,
    VariableDef,
    VariationPlan,
    register_variable,
    sample_inputs,
    variable_registry,
)

from .dispersion import (
    PuttDispersionReport,
    PuttOutcome,
    PuttVariableDeclaration,
    summarize_putt_outcomes,
)
from .green import CaptureModel, simulate_putt_on_surface
from .impact import PutterSpec, strike
from .result_wire import (
    PuttingResultDocument,
    PuttingResultProvenance,
    putting_result_document,
)
from .roll import DEFAULT_SLIDING_MU
from .surface import GreenSurface, GridGreenSurface, PlanarGreenSurface

CATEGORY_PUTTING = "swing_sim.putting"

PUTT_SPEED_KEY = f"{CATEGORY_PUTTING}.clubhead_speed_mps"
PUTT_FACE_KEY = f"{CATEGORY_PUTTING}.face_angle_deg"
PUTT_PATH_KEY = f"{CATEGORY_PUTTING}.path_angle_deg"
PUTT_AIM_KEY = f"{CATEGORY_PUTTING}.aim_deg"
PUTT_STRIKE_TOE_KEY = f"{CATEGORY_PUTTING}.strike_offset_toe_mm"

#: Registry keys this package owns; nothing else may enter a plan.
PUTT_VARIABLE_KEYS = frozenset(
    {
        PUTT_SPEED_KEY,
        PUTT_FACE_KEY,
        PUTT_PATH_KEY,
        PUTT_AIM_KEY,
        PUTT_STRIKE_TOE_KEY,
    }
)

#: Stroke-field name each registry key perturbs.
_KEY_TO_FIELD: Mapping[str, str] = MappingProxyType(
    {
        PUTT_SPEED_KEY: "clubhead_speed_mps",
        PUTT_FACE_KEY: "face_angle_deg",
        PUTT_PATH_KEY: "path_angle_deg",
        PUTT_AIM_KEY: "aim_deg",
        PUTT_STRIKE_TOE_KEY: "strike_offset_toe_mm",
    }
)

__all__ = [
    "CATEGORY_PUTTING",
    "PUTT_AIM_KEY",
    "PUTT_FACE_KEY",
    "PUTT_PATH_KEY",
    "PUTT_SPEED_KEY",
    "PUTT_STRIKE_TOE_KEY",
    "PUTT_VARIABLE_KEYS",
    "PuttScenario",
    "PuttStroke",
    "PuttVariationPlan",
    "evaluate_putt",
    "putt_outcome",
    "run_putt_dispersion",
]


@dataclass(frozen=True)
class PuttStroke:
    """The delivered stroke — P1's :func:`~.impact.strike` arguments.

    These are the *declared* nominal values a variation plan perturbs;
    each field is validated by ``strike`` itself at evaluation time, so
    an out-of-envelope draw is refused there rather than clamped here.
    """

    clubhead_speed_mps: float
    shaft_lean_deg: float = 0.0
    aim_deg: float = 0.0
    face_angle_deg: float = 0.0
    path_angle_deg: float = 0.0
    attack_angle_deg: float = 0.0
    strike_offset_toe_mm: float = 0.0
    strike_offset_high_mm: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "clubhead_speed_mps",
            "shaft_lean_deg",
            "aim_deg",
            "face_angle_deg",
            "path_angle_deg",
            "attack_angle_deg",
            "strike_offset_toe_mm",
            "strike_offset_high_mm",
        ):
            require_finite(getattr(self, name), name)

    def base_values(self) -> dict[str, float]:
        """This stroke as the registry-keyed base a plan varies about."""
        return {key: getattr(self, name) for key, name in _KEY_TO_FIELD.items()}


@dataclass(frozen=True)
class PuttScenario:
    """One fully specified putt: putter, stroke, green, and hole.

    Attributes:
        scenario_id: Stable identity carried into every report.
        putter: The v1 putter spec (build it through
            ``golf_club.putter_head`` when a v2 document exists).
        stroke: Nominal delivered stroke.
        surface: Green geometry (P2).
        stimp_ft: Stimpmeter reading [feet].
        hole_distance_m: Distance to the hole centre [m].
        provenance: Putter/stroke origin recorded in every result.
        mu_slide: Sliding friction for the skid phase.
        capture_model: Hole-capture model (P2 default: the published
            effective-radius model).
        head_moi_kg_m2: Putter-head MOI for P1's off-centre
            effective-mass reduction; ``None`` selects P1's catalogue
            default. Fill it from
            ``golf_club.putter_head.head_moi_for_strike``.
    """

    scenario_id: str
    putter: PutterSpec
    stroke: PuttStroke
    surface: GreenSurface
    stimp_ft: float
    hole_distance_m: float
    provenance: PuttingResultProvenance
    mu_slide: float = DEFAULT_SLIDING_MU
    capture_model: CaptureModel = "effective_radius"
    head_moi_kg_m2: float | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.scenario_id, str) and bool(self.scenario_id.strip()),
            "scenario_id must be a name",
        )
        require(isinstance(self.putter, PutterSpec), "putter must be a PutterSpec")
        require(isinstance(self.stroke, PuttStroke), "stroke must be a PuttStroke")
        require(
            isinstance(self.surface, (PlanarGreenSurface, GridGreenSurface)),
            "surface must be a GreenSurface",
        )
        require(
            isinstance(self.provenance, PuttingResultProvenance),
            "provenance must be PuttingResultProvenance",
        )
        require_finite(self.stimp_ft, "stimp_ft")
        require_finite(self.hole_distance_m, "hole_distance_m")
        require_finite(self.mu_slide, "mu_slide")
        require(
            self.provenance.capture_model == self.capture_model,
            "provenance capture_model must match the scenario",
            (self.provenance.capture_model, self.capture_model),
        )
        if self.head_moi_kg_m2 is not None:
            require_finite(self.head_moi_kg_m2, "head_moi_kg_m2")


def evaluate_putt(scenario: PuttScenario) -> PuttingResultDocument:
    """Run one putt end to end and return its ``putting_result/2`` record.

    Args:
        scenario: The fully specified putt.

    Returns:
        The v2 :class:`~.result_wire.PuttingResultDocument`.

    Raises:
        TypeError: If ``scenario`` is not a :class:`PuttScenario`.
        ValueError / ContractViolationError: If any value leaves a
            model's validity envelope.
    """
    if not isinstance(scenario, PuttScenario):
        raise TypeError("scenario must be a PuttScenario")
    stroke = scenario.stroke
    launch = strike(
        scenario.putter,
        stroke.clubhead_speed_mps,
        stroke.shaft_lean_deg,
        aim_deg=stroke.aim_deg,
        face_angle_deg=stroke.face_angle_deg,
        path_angle_deg=stroke.path_angle_deg,
        attack_angle_deg=stroke.attack_angle_deg,
        strike_offset_toe_mm=stroke.strike_offset_toe_mm,
        strike_offset_high_mm=stroke.strike_offset_high_mm,
        head_moi_kg_m2=scenario.head_moi_kg_m2,
    )
    result = simulate_putt_on_surface(
        launch,
        scenario.surface,
        stimp_ft=scenario.stimp_ft,
        hole_distance_m=scenario.hole_distance_m,
        mu_slide=scenario.mu_slide,
        capture_model=scenario.capture_model,
    )
    return putting_result_document(
        launch,
        result,
        scenario.provenance,
        hole_distance_m=scenario.hole_distance_m,
    )


def putt_outcome(document: PuttingResultDocument) -> PuttOutcome:
    """The dispersion outcome carried by one v2 result record."""
    if not isinstance(document, PuttingResultDocument):
        raise TypeError("document must be PuttingResultDocument")
    miss = document.miss_distance_m
    if document.holed:
        leave = 0.0
    elif miss is None:
        raise ValueError("a missed putt must report a leave distance")
    else:
        leave = float(miss)
    return PuttOutcome(
        holed=document.holed,
        start_azimuth_deg=document.start_azimuth_deg,
        leave_distance_m=leave,
        total_distance_m=document.total_distance_m,
        break_m=document.final_break_m,
        capture_margin_m=document.capture_margin_m,
    )


@dataclass(frozen=True)
class PuttVariationPlan:
    """Reproducible putting-only plan on the canonical seeded sampler.

    Mirrors ``VariationPlan``'s sampling-only shape (see the module
    docstring). ``base_variables`` stays empty in a plan handed to
    :func:`run_putt_dispersion`: the *scenario* is the base, and two
    answers to one question is not a contract.

    An empty ``noise`` tuple is the legal degenerate plan — no declared
    variance, therefore no sampling, therefore ``n_runs`` copies of the
    nominal putt.
    """

    noise: tuple[NoiseSpec, ...] = ()
    base_variables: Mapping[str, float] = field(default_factory=dict)
    n_runs: int = 200
    seed: int = 0
    groups: tuple[PerturbationGroup, ...] = ()

    def __post_init__(self) -> None:
        require(self.n_runs >= 1, "n_runs must be >= 1", self.n_runs)
        require(self.seed >= 0, "seed must be >= 0", self.seed)
        require(not self.groups, "grouped putting variation is not yet supported")
        require(
            all(isinstance(spec, NoiseSpec) for spec in self.noise),
            "noise entries must be NoiseSpec",
        )
        keys = tuple(spec.variable_key for spec in self.noise)
        require(set(keys) <= PUTT_VARIABLE_KEYS, "noise contains non-putting variables")
        require(len(keys) == len(set(keys)), "noise variable keys must be unique")
        base = {str(key): float(value) for key, value in self.base_variables.items()}
        require(set(base) <= PUTT_VARIABLE_KEYS, "base contains non-putting variables")
        require(
            all(math.isfinite(value) for value in base.values()),
            "base must be finite",
        )
        object.__setattr__(self, "noise", tuple(self.noise))
        object.__setattr__(self, "base_variables", MappingProxyType(base))

    def resolved_base(self) -> dict[str, float]:
        """Registry defaults overlaid by explicit putting base values."""
        registry = variable_registry()
        resolved = {key: registry[key].default for key in PUTT_VARIABLE_KEYS}
        resolved.update(self.base_variables)
        return resolved

    def declarations(self) -> tuple[PuttVariableDeclaration, ...]:
        """What this plan varies, for the dispersion report."""
        return tuple(
            PuttVariableDeclaration(
                variable_key=spec.variable_key,
                distribution=spec.distribution,
                scale=spec.scale,
            )
            for spec in self.noise
        )


def _sampled_strokes(
    scenario: PuttScenario, plan: PuttVariationPlan
) -> tuple[PuttStroke, ...]:
    """Deterministic per-run strokes through the shared sampler."""
    if not plan.noise:
        return (scenario.stroke,) * plan.n_runs
    based = replace(plan, base_variables=scenario.stroke.base_values())
    # The plan mirrors VariationPlan's sampling-only shape but cannot BE
    # one: its putting variables are illegal in every pipeline mode.
    samples = np.asarray(sample_inputs(cast(VariationPlan, based)), dtype=float)
    fields = tuple(_KEY_TO_FIELD[spec.variable_key] for spec in plan.noise)
    return tuple(
        replace(
            scenario.stroke,
            **{name: float(value) for name, value in zip(fields, row, strict=True)},
        )
        for row in samples
    )


def run_putt_dispersion(
    scenario: PuttScenario, plan: PuttVariationPlan
) -> tuple[PuttDispersionReport, tuple[PuttingResultDocument, ...]]:
    """Evaluate a putting variation study (module docstring).

    Args:
        scenario: The nominal putt; its stroke is the plan's base.
        plan: Declared distributions, run count, and seed. It must not
            declare ``base_variables`` — the scenario is the base.

    Returns:
        The :class:`~.dispersion.PuttDispersionReport` and the per-run
        ``putting_result/2`` records behind it, in sample order.

    Raises:
        TypeError: If either argument is the wrong type.
        ContractViolationError: If the plan declares base values, or a
            draw leaves a model's validity envelope.
    """
    if not isinstance(scenario, PuttScenario):
        raise TypeError("scenario must be a PuttScenario")
    if not isinstance(plan, PuttVariationPlan):
        raise TypeError("plan must be a PuttVariationPlan")
    require(
        not plan.base_variables,
        "the scenario is the plan's base; declare no base_variables",
        sorted(plan.base_variables),
    )
    documents = tuple(
        evaluate_putt(replace(scenario, stroke=stroke))
        for stroke in _sampled_strokes(scenario, plan)
    )
    report = PuttDispersionReport(
        scenario_id=scenario.scenario_id,
        seed=plan.seed,
        variables=plan.declarations(),
        summary=summarize_putt_outcomes(
            tuple(putt_outcome(document) for document in documents)
        ),
    )
    return report, documents


def _register_putting_variables() -> None:
    """Declare the putting perturbation vocabulary (import-time, once)."""
    guidance = (
        "Illustrative sensitivity scale only; supply a calibrated "
        "stroke or green-reading distribution with provenance before "
        "quoting a make percentage or a dispersion as a golfer's."
    )
    definitions = (
        VariableDef(
            PUTT_SPEED_KEY,
            "Putter Head Speed",
            "m/s",
            1.5,
            0.05,
            guidance,
        ),
        VariableDef(
            PUTT_FACE_KEY,
            "Putter Face Angle",
            "deg",
            0.0,
            0.5,
            guidance,
        ),
        VariableDef(
            PUTT_PATH_KEY,
            "Putter Path Angle",
            "deg",
            0.0,
            1.0,
            guidance,
        ),
        VariableDef(
            PUTT_AIM_KEY,
            "Putting Aim (Green Read)",
            "deg",
            0.0,
            0.5,
            guidance,
        ),
        VariableDef(
            PUTT_STRIKE_TOE_KEY,
            "Putter Strike Offset (Toe)",
            "mm",
            0.0,
            3.0,
            guidance,
        ),
    )
    for definition in definitions:
        register_variable(definition)


_register_putting_variables()
