# Impact Dynamics and Acoustics Research Specification

Parent [#5068](https://github.com/D-sorganization/Tools/issues/5068).
Theory: [AffineDrift #4253](https://github.com/D-sorganization/AffineDrift/issues/4253).
Consumer: [UpstreamDrift #9700](https://github.com/D-sorganization/UpstreamDrift/issues/9700).
Status: reference foundation and prospective research design; no calibrated
full-club acoustics or player-effect claim. Inventory date: 2026-09-07.

## Existing Integration Points

The detailed cross-repository inventory is maintained in AffineDrift's
`docs/development/impact-acoustics/INVENTORY.md`. Within Tools, reuse:

- `golf_club`: physical inertia validation/mesh properties, profile/statics/modal
  shaft models, quasi-static delivery, fitting contracts/comparators, and the
  one-dimensional `impact_coupling` model.
- `swing_sim.impact`: existing impact types/solvers, unilateral contact and
  geometry/gear-effect behavior; preserve API contracts during upgrades.
- `swing_sim.model_interchange` and `delivery_interchange`: engine-neutral
  documents; neither currently identifies full human grip impedance.
- `swing_sim.variation` and flight impact-solution contracts: paired studies,
  sensitivity and reproducible reports.
- Signal Toolkit fitting, Signal Processing Studio and MATLAB Audio Processor:
  reusable signal operations/presentation. These are not yet calibrated golf
  acoustic pressure or structural-radiation implementations.
- Rate of Closure Python/React surfaces and existing golden fixtures: display
  qualified results without implementing another physics layer in the UI.

All `golf_club` and `swing_sim` paths above are under `src/shared/python`.
No public v1 function is removed or repurposed by this foundation. Consumers
must use the provider's reviewed package version; a sibling checkout is only an
explicit development test environment.

## Work Breakdown and Dependencies

| Slice | Issue                                                         | Scope and Exit Evidence                                                                                   | Depends On                                                     |
| ----- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| IA-T1 | [#5069](https://github.com/D-sorganization/Tools/issues/5069) | Full tensor detached contact reference; independent analytic/invariance/impulse-energy tests              | Theory definitions                                             |
| IA-T2 | [#5071](https://github.com/D-sorganization/Tools/issues/5071) | Legacy chain semantics, separation/event/stability and complete dissipation audit; preserved wire         | T1                                                             |
| IA-T3 | [#5072](https://github.com/D-sorganization/Tools/issues/5072) | Distributed prestress, bending/torsion/axial motion, rotating-base consistency and passive grip impedance | T1, T2                                                         |
| IA-T4 | [#5073](https://github.com/D-sorganization/Tools/issues/5073) | Coupled finite contact and flexible head/face, off-center friction/gear effect, launch and ringdown       | T1, T3                                                         |
| IA-T5 | [#5074](https://github.com/D-sorganization/Tools/issues/5074) | Calibrated signal ingestion, FRF/decay analysis and validated radiation                                   | T3, T4 for production predictions; ingestion can start earlier |
| IA-T6 | [#5075](https://github.com/D-sorganization/Tools/issues/5075) | Versioned study reports, compatibility and truthful application surfaces                                  | T2–T5 and consumer state/study contracts                       |

Existing #4562 and #4549 are foundations, not substitutes for these acceptance
criteria. Do not close the new epic merely because its planning or T1 is merged.

## IA-T1 Public Contract

`golf_club.impact_mobility` is an additive submodule. Import it directly; avoid
expanding eager package imports and inadvertently loading GUI/scientific stacks.

- `RigidContactBody(mass_kg, inertia_at_com_kg_m2, contact_offset_m)` copies input
  storage to immutable values using existing validators. Positive mass and
  physically realizable positive-definite COM inertia are required. All vectors
  and tensors use a common orthonormal frame; the offset is COM to contact.
- `contact_inverse_mass(body)` returns the symmetric point inverse-mass matrix
  in kg⁻¹. It maps an impulse on this body to contact-point velocity change.
- `normal_effective_mass(body, normal)` returns the reciprocal projected inverse
  mass. A finite unit normal is required; normalization is never guessed.
- `normal_impulse(closing_speed_mps, first_effective_mass_kg,
second_effective_mass_kg, restitution)` returns a nonnegative frictionless
  two-body impulse, zero for separating/touching states. COR is in [0,1].

This kernel contains no shaft, constraint, preload, face/contact compliance or
acoustics. It is a verification reference. Its scalar impulse must not be pasted
into a coupled friction solver as an independently fixed normal response.

Acceptance uses centered/principal-axis closed forms, direct spatial impulse
response with products of inertia, simultaneous frame rotation, momentum and
angular momentum balance, restitution and exact kinetic-energy loss. Invalid
geometry, units represented by wrong types, nonfinite values, singular inertias
and invalid COR are rejected at API boundaries. A caller still owns the truthful
frame/unit metadata; numeric validation cannot detect a mislabeled unit.

## Required State and Output Contract for Later Tiers

A study input must identify equipment and calibration, world/head/grip frames,
head/ball pose and twist, full inertia, actual contact point and local normal,
shaft station/basis identity, prestress and modal displacement/velocity,
per-hand wrench and impedance, boundary constraints and source time.
Do not substitute zero for unmeasured/unrepresented elastic states. Distinguish
prescribed grip motion (which can do work) from applied wrench control.

Outputs retain launch/spin, head response, vector impulse, force/moment history,
contact duration and event rule, structural state through ringdown, observer
pressure only when qualified, and separate numerical/measurement uncertainty.
Include model tier, assumptions, raw/calibration/code hashes, deterministic case
identity, failed/refused cases, convergence evidence and complete energy/work
accounting. Acoustic energy must not be double-counted as structural damping.

The legacy `decoupling_fraction` is a clipped relative ball-speed comparison,
not a fraction of inertial mass contributed by the golfer. Preserve its wire
meaning while exposing clearer metrics in a new schema. The old rigid-link
comparison is not a theorem bounding arbitrary preloaded or resonant systems.

## Verification and Validation Gates

Use tests first, DbC, LoD and DRY for every implementation. Shared laws stay here;
engine adapters and hypothesis orchestration stay in UpstreamDrift. Interfaces
must be small and frame-explicit; source arrays cannot be silently mutated.

For each dynamic tier register tolerances before generating results. Require:

1. Detached/static/zero-load limits and correct dimensional scaling; full tensor
   covariance and wrench/velocity power invariance.
2. Conservative energy closure, passive dissipation, gyroscopic zero work,
   tensile geometric stiffness sign, and explicit boundary work.
3. Independent time-step, spatial and modal convergence, with separation-event
   refinement. Candidate initial engineering targets: under 0.1% relative
   change in nonzero launch speed/impulse and under 1% in retained resonances;
   near-zero quantities use registered dimensional absolute tolerances.
   These are proposed numerical targets, not validated physical accuracy.
4. Contact-band FRF magnitude/phase and ringdown convergence; separately qualify
   acoustic discretization, observer position and pressure/decay convergence.
   Ball-speed convergence alone cannot qualify an audio waveform.
5. Independent calibration/validation splits, held-out strike positions and
   observer locations, sensor uncertainty and parameter identifiability.
6. Signal tests for normalization/Parseval, impulse response, damped oscillator,
   delay/phase recovery, calibration units, clipping and missing metadata.
   Psychoacoustic metrics require named validated algorithms and fixtures.

Physical tests progress from static/modal measurements to tensioned fixtures,
plate impact, assembled club, and then human studies. Human grip and perceptual
claims remain unqualified until the registered experiments actually run.
Synthetic tests and cross-engine agreement establish numerical behavior only.

## Turnover and Current Delivery

- Worktree: `C:/Users/diete/Repositories/Tools-impact-acoustics`.
- Branch: `feat/5068-impact-dynamics-foundation`; base `b4875be19`.
- Issue: #5069; implementation commit `SELF`; PR #5077; implementation `6d94f1d3d`.
- TDD: missing-module RED, then 30 passing independent reference tests.
- Validation: 383 package/impact/API tests passed, 2 skipped; the configured
  qt_api warning remains. Ruff, formatting, mypy and SPEC checks pass. Additive
  API baseline and source/test inventory manifests regenerated.
- All applicable commit/push hooks passed, including strict mypy and unit tests.
- Next: protected PR #5077 review, then #5071 and the structural milestones.
- Parent #5068 and downstream research slices remain open after T1 delivery.
