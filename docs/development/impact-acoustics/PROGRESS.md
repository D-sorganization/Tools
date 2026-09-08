# Impact and Acoustics Program Execution

The user authorized completion of the full planned program, not merely its
planning deliverables. All three parent epics remain open. Physical and
perceptual validation cannot be inferred from numerical fixtures.

## Requirement and Evidence Matrix

| Slice                          | Issue / Delivery               | Evidence Required Before Completion                                                                                           | Current State                                                                                                |
| ------------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Theory review                  | AffineDrift #4254 / PR #4258   | Corrected rendered theory, source ledger and inventory; protected delivery                                                    | Merged PR #4258 at `1ce02d7ae4d916d4c598b62be78cefa83452aaa7`                                                |
| Damping/transport correction   | AffineDrift #4277 / PR #4282   | Paired source, combined book, browser/PDF review and protected CI                                                             | Merged `1968897ec65044b8393705087fccdf755e3e89a2`; all CI passed                                             |
| Final theory synthesis         | AffineDrift #4255              | Qualified downstream results with uncertainty and limits                                                                      | Not started; depends on evidence                                                                             |
| Rigid reference                | Tools #5069 / PR #5077         | Analytic tensor/impulse gates and protected provider delivery                                                                 | Merged PR #5077 at `f7254461399ac18e5667a0215afd90a9ebff9d22`                                                |
| Lumped qualification           | Tools #5071 / PR #5082         | Events, work/loss ledger, timeout/step contracts, law-consistent restitution, scaling counterexamples, parity and convergence | PR #5082 open; latest observed remote head `476eaa98b`                                                       |
| Distributed shaft/grip         | Tools #5072                    | Prestressed rotating operators, passive impedance, beam limits, frame agreement, modal/mesh/time/FRF convergence              | Clamped roots, section kinetics and deformed-frame inertia verified; loaded assembly/stability/work/FRF open |
| Flexible contact               | Tools #5073                    | Off-center friction/contact and head/shaft modes; launch and ringdown; complete energy closure                                | Pending T3                                                                                                   |
| Acoustics                      | Tools #5074                    | Calibrated signals, identified transfer, qualified radiation and held-out validation                                          | Early ingestion PR #5084 merged; boundary probes identify qualification work, radiation/measurements pending |
| Reports/surfaces               | Tools #5075                    | Versioned provenance reports, consumer compatibility and truthful UI integration                                              | PR #5083 proposes study wire/surface; source review and qualified consumers pending                          |
| Integration plan               | UpstreamDrift #9701 / PR #9706 | Source/state inventory and protected delivery                                                                                 | Merged PR #9706 at `dbc6727aa4f0d422b7adaf6957e658e8997f7f29`                                                |
| Swing adapters                 | UpstreamDrift #9703            | Compatible rigid, elastic, prestress and wrench transfer on exact provider pin                                                | Pending provider contract                                                                                    |
| Counterfactual studies         | UpstreamDrift #9704            | Registered matched-state and matched-input studies, reproducible results, uncertainty                                         | Pending verified coupled model                                                                               |
| Physical/perceptual validation | UpstreamDrift #9705            | Synchronized calibrated measurements, held-out validation, blinded sweetness analysis                                         | Data/equipment availability requested; no experiment run                                                     |

## Current Implementation: IA-T3 (Partial)

`Tools-impact-shaft`, branch `feat/5072-prestressed-shaft`, base T2 `2f975d06e`.
Pushed checkpoints include tensile FEM `f1f8da112`, grip `8f025d570` and stationary
spatial shaft/full head `43c228da1`. Rotating transport is pushed at `59ca36c60c03a67ce8b9cfdfaac0d195f7863ad5`,
with all normal commit/push hooks passing and the remote head verified.
It retains the full head COM/inertia and explicit distributed section rotary
inertia, with separate gyroscopic, centrifugal, Euler and origin-acceleration
terms. Nominal inertial forcing is returned; equilibrium is never presumed.

The combined shaft/grip/head/API suite passes 110 tests (31.76 s). Independent
finite-rotation kinetic Hessians, a free inertial trajectory reconstructed from
an accelerating rotating frame, and published radial frequencies are checked.
See ROTATING_TRANSPORT.md for equations, tolerances and the exact/approximate
reference distinction. SHAFT_LINEAR_SYSTEM.md, GRIP_IMPEDANCE.md and
SHAFT_PRESTRESS.md retain earlier derivations and TDD evidence. These numerical
checks do not close #5072: loaded shape, consistent prestress/boundary work,
stability and mesh/time/modal/FRF/bandwidth qualification remain required.

The section-kinematics checkpoint is `52d6ec791`; elastic energy and full
internal tangent are published at `b625eb2cc`, and physical point-load work and
derivatives at `6323944ee658f1254da1a08bb899551f082c7b5c`. All normal hooks
passed for those pushes. Full-node assembly is published at
`87231b2f0c8d6fea45b37e9b92795eca740ba578`; clamped root finding is published at
`7d586c49467a1a5cf14a2ff860461022c6ad4128`, with all normal hooks passing.
Current `SELF` adds consistent section kinetic quadrature, using the same SE(3)
interpolation, existing physical mass properties and the existing spatial-inertia
kernel. Independent COM/spin, acceleration/angular-momentum and energy-rate
oracles pass. The shared relative-log map is reused by elasticity and inertia.
All 136 focused contracts and 543 golf/API tests pass, with two optional CAD skips
(75.67 s for the full run). Ruff 0.14.10 and actual three-module pre-push mypy pass.
Only a private no-export API entry is added. LOADED_INERTIA.md gives equations,
quadrature/input domains, TDD evidence and the distinction between geometric
energy exchange and damping. These are synthetic numerical checks.

Complete rotating loaded residual/derivatives, stability, moving work and
mesh/time/modal/FRF qualification remain open. T3 PR is not created. Inventory
and handoffs are refreshed and all nine manual gates pass. Run all normal
commit/push checks before publishing this kinetic checkpoint.

## Current Delivery and Data Status

UpstreamDrift prerequisite #9735 / PR #9745 merged on 2026-09-08 at 04:47:54 UTC
as `1b48707d54fb47655e43eaaffaad7b1739445e40`; a compare against main confirms
zero commits behind and three ahead. It preserves runtime imports and the exact
vendor pin. Local evidence includes 13 no-vendor contracts, 72 pinned-provider/
CLI/fallback tests, actual installed-wheel bootstrap, ownership mutation and
14 explicit-mode contracts. The earlier fixture-mode mismatch had seven RED
failures before correction. Obsolete branch parity/security failures are not
reported as current blockers after the merge.

Tools T1 #5077 merged at `f7254461399ac18e5667a0215afd90a9ebff9d22`; T2 #5082 remains open. The failed T2 UD consumer job
101931398231 (run 34180114995) was rerun through the normal REST endpoint after
confirming the upstream merge. The resulting UD job 101951943569 passed, including actual downstream
consumer contracts. The older T1 UD job 101870569015 (run 34163726663) has
now also passed as job 101955835753. Gasification_Model exists
as a private repository but checkout previously failed; a secret-metadata query
returned 403 and does not establish whether a credential is absent or expired.
Current Gasification job 102009926186 (run 34208168311, T2 head `476eaa98b`)
still fails private checkout with 404 before tests. The available browser is
signed out. Existing credential configuration was requested, without asking
for secret values; no absence/expiry diagnosis is inferred from metadata 403.
Renderer #5090 publishes `b8c6e6013ad9dab6eab0bbfe0096b449feb16abb`. Both Linux
attempts of run 34213771459 pass 73 browser and 23 PyQt tests; all ten PyQt PNGs
are byte-identical across attempts. All twenty initial desktop references were
individually reviewed and refreshed without changing thresholds. The fresh
production workflow 34217794994/job 102033568943 passes. The fresh Gasification consumer job 102039836106 fails checkout again. Rust
quality job 102033732755 fails a timing-dependent debounce-count assertion;
issue #5095 now owns its deterministic test repair, claim free and lease posted.
No Rust repair is implemented yet. Other protected checks remain pending;
overlapping PR #5087 is closed, unmerged. CI_FINDINGS.md links
the exact review ledger. No whole-product or human approval is invented.
AffineDrift #4282 merged as `1968897ec65044b8393705087fccdf755e3e89a2` after all
CI passed. Its paired damping correction, 562-page combined book, eight browser
screenshots and six changed PDF pages were reviewed; 128 targeted tests passed.
The merge preserved the incoming complete-swing chapter by rebuilding both.

DATA_CANDIDATES.md records RealImpact as a measured household-object acoustic
method candidate. A bounded ZIP-directory inspection found only deconvolved audio plus geometry
and location metadata for the iron-plate archive; raw paired force/audio is absent
from that directory. No recording values have been inspected or calibrated;
dataset licensing and preprocessing assumptions still need verification. It cannot
replace golf/player measurements. Hardware/data availability remains unanswered;
no experiment, acoustic effect or perceptual preference is inferred.

## Previous Implementation: IA-T2

Worktree: `C:/Users/diete/Repositories/Tools-impact-coupling`.
Branch: `fix/5071-qualified-impact-coupling`, stacked on T1 head `5932146f1`.
The initial T2 claim check was free. Lease label posting failed; the documented
fleet fail-open policy allows progress but is not evidence of a posted lease.

The legacy integrator returned unfinished collisions at max_time and accepted
under-resolved steps. Two tests demonstrated those failures. A single audited
solver now serves both the existing v1 result/report and the new audit API.
DOP853 uses dt_s as a maximum step; geometric clearance is a located terminal
root. First decreasing force-zero time is separate. Peak force remains sampled
at accepted steps plus the first-touch right-hand limit and requires refinement.

The audit initializes actual displacements/velocities (including preload),
tracks contact viscous loss, clipped-contact potential loss, shaft loss and
grip loss, and reports numerical closure error separately. Fixed-anchor work is
zero in the declared inertial translating frame. The v1 energy fraction remains
retained energy, and the v1 contact time remains geometric clearance time.
No scalar score is reinterpreted as coupled body mass.

## Verification Record

- Initial RED: 8 failing tests, including two demonstrated existing defects.
- Initial GREEN: 19 audit/legacy tests passed (92.09 s).
- Additional RED: invalid energy-ledger construction accepted negative loss.
- Broader GREEN: 396 golf_club, impact and API tests passed, 2 skipped (72.54 s).
  A previous run hit the existing 60-second per-test budget in the large report
  sweep. Its explicit step is now 0.5 microseconds, within the rate guard even
  at the 1e9 N/m synthetic stiffness. Production defaults are unchanged.
- The browser report parser passes all 6 clubFitting tests with the corrected v1
  fixture. Strict mypy on the three new modules passes.
- Additional RED: extreme frame velocities erased the closing speed by rounding
  or overflowed the derivative. The first-touch constructor now refuses a
  relative velocity that cannot be represented to relative 1e-9. All 15 audit
  tests pass after that guard (7.61 s).
- All nine required design-manual gates pass after the changes. Existing manual
  approval blockers remain; this work does not promote publication authority.
- Repository-wide Ruff 0.14.10 (the pinned version) reports all 3,697 files
  formatted and no lint errors. Unpinned local Ruff 0.16.4 gave unrelated
  baseline diagnostics; no unrelated files were changed to satisfy that version.
- API baseline regenerated additively; existing public symbols/signatures remain.

## Next Actions and Completion Boundary

The rotating checkpoint is saved with refreshed inventories and all nine
required gates. Construct a consistent loaded state and moving boundary work
audit; LOADED_STATE_REVIEW.md records the next formulation candidates and reuse
constraints. Preserve the existing `swing_sim.impact_interval` facade and rigid
full-inertia/friction/trace/wire capabilities as T4 integration points. It is
not a distributed shaft or radiation solver. Resolve remaining provider CI
through normal protections and inspect remote heads before changing branches.

The full goal remains active. T4-T6, exact-pin U2 adapters, U3 counterfactuals,
U4 physical/blinded validation and A2 synthesis remain required in the matrix.
Do not invent measurements or approvals to complete a gate.

## Stiff-Fixture Failure Discovered in the Broader Run

The broader run passed 393 tests and exposed a stale provenance assertion and
an existing wire fixture that returned a still-contacting state at 5 ms. Its
head/ball/grip masses are 0.2/0.04593/2.5 kg, speed 45 m/s, shaft stiffness
50,000 N/m, grip stiffness/damping 50,000 N/m and 50 N s/m, and contact
stiffness/damping 1,000,000 N/m and 1,000 N s/m. The fixture is synthetic.

With maximum dt 2 microseconds and a 10 ms horizon, the solver locates first
force release at 0.274235 ms and geometric clearance at 7.168249 ms. The old
truncated report gave 49.75424 m/s; completed clearance gives 52.09702 m/s.
The independently integrated work/energy residual is about 1.5e-10 J. These
are model outputs, not evidence that real golf contact lasts seven milliseconds.
The long/recontact behavior is a reason to qualify/calibrate the contact law.

A dedicated regression now rejects the original 5 ms timeout. The successful
wire fixture declares the longer horizon, and its test reads the committed
fixture instead of overwriting its expected result before comparing it.
Numeric wire comparisons use relative 1e-8 / absolute 1e-10 tolerances; identifiers
and structure remain exact. The generic provenance string is explicitly synthetic.
The baseline speed changes by only about 0.00018 m/s, while the previously
unfinished stiff case changes materially. Do not conceal that difference as
floating-point noise or certify the previous output as a completed collision.

## Execution Environment

Use the ignored worktree Python 3.12 `.venv`, with pytest-qt 4.5.0. Prepend its
Scripts directory to PATH and set PYTEST_ADDOPTS=-n0 and QT_QPA_PLATFORM=offscreen
for normal hooks. System Python 3.13 previously crashed a MuJoCo parallel worker.
The qualified serial unit run passed 1,614 tests with 29 skips, 9 expected failures
and one existing unexpected pass. No hook was bypassed. Tools pins Ruff 0.14.10.
Changed-module mypy with `--follow-imports=silent` passes; unrestricted recursive
mypy encounters existing fitting_document.py diagnostics and is not claimed green.

The previous stationary checkpoint passed 372 golf-club tests with two skips
(192.03 s), repository-wide Ruff, all nine manual gates, and commit/push hooks.
Publication approval remains separate. Refreshed rotating-checkpoint evidence
belongs in ROTATING_TRANSPORT.md; prior execution details remain in git history.

## Current Assembly Checkpoint

Section kinetics is published at `22cfc8df9`; current deformed-frame inertia passes
147 focused and 554 full golf/API tests, with two optional CAD skips.
See `DEFORMED_FRAME_INERTIA.md` for the residual/Jacobian/gyroscopic derivation.
Newly merged T5 ingestion is inventoried in `VIBROACOUSTICS_INGESTION_REVIEW.md`,
including exact-commit calibration/hash/alignment/spectral boundary probes.
It does not close complete rotating loaded dynamics, stability or measured gates.
The complete program requirement/evidence matrix above remains authoritative.
