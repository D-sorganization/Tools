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
| Distributed shaft/grip         | Tools #5072                    | Prestressed rotating operators, passive impedance, beam limits, frame agreement, modal/mesh/time/FRF convergence              | Loaded roots and M/G/K operators pass independent controls; stability/work/grip/FRF open                     |
| Flexible contact               | Tools #5073                    | Off-center friction/contact and head/shaft modes; launch and ringdown; complete energy closure                                | Pending T3                                                                                                   |
| Acoustics                      | Tools #5074                    | Calibrated signals, identified transfer, qualified radiation and held-out validation                                          | Early ingestion PR #5084 merged; boundary probes identify qualification work, radiation/measurements pending |
| Reports/surfaces               | Tools #5075                    | Versioned provenance reports, consumer compatibility and truthful UI integration                                              | PR #5083 merged at `cfca06449`; strict wire/evidence and consumer gaps remain on #5075                       |
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
Published checkpoint `22cfc8df9` adds consistent section kinetic quadrature, using the same SE(3)
interpolation, existing physical mass properties and the existing spatial-inertia
kernel. Independent COM/spin, acceleration/angular-momentum and energy-rate
oracles pass. The shared relative-log map is reused by elasticity and inertia.
All 136 focused contracts and 543 golf/API tests pass, with two optional CAD skips
(75.67 s for the full run). Ruff 0.14.10 and actual three-module pre-push mypy pass.
Only a private no-export API entry is added. LOADED_INERTIA.md gives equations,
quadrature/input domains, TDD evidence and the distinction between geometric
energy exchange and damping. These are synthetic numerical checks.

The later rotating loaded-root checkpoint below supersedes the residual/derivative
assembly gap. Stability, moving work and mesh/time/modal/FRF qualification remain
open. T3 PR is not created. Inventory
and handoffs are refreshed and all nine manual gates pass. Run all normal
commit/push checks before publishing each subsequent checkpoint.

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
Rust repair #5097 merged at `d9dec3602`; standard CI attempt 2 passes. Separate optional/private checks remain unqualified;
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

Rotating loaded roots are published at `f47f64acf`. Current private loaded
M/G/K/residual assembly reuses those roots and deformed inertial kernels.
All seven new Windows tests and nine API tests pass; the separate Linux run
passes 567 golf/API tests with two optional CAD skips (296.00 s).
`LOADED_DYNAMIC_OPERATORS.md` records the material-coordinate contract,
physical energy/Coriolis controls, guided axial spin-softening limit and
second-order frequency convergence. It also records the unchanged Windows
report-sweep timeout and separate Linux qualification. Stability, moving work,
passive grip/head integration and full FRF/convergence remain open.

T5 ingestion #5084 and T6 study records #5083 are merged foundations. Reuse
`swing_sim.vibroacoustics` and `swing_sim.impact_studies`. T6's 19 tests pass;
read-only probes confirm contradictory empty-acoustic availability, discarded
unknown fields, accepted dict-valued v1 report, and unverified measured/validated
labels. Issue #5075 records those gaps; metadata labels do not authenticate data.
UpstreamDrift renderer #9784 merged. Its authority prerequisite #9787 / PR #9804
publishes `d1563dffa` with actual native provenance, 116 strict authority tests,
three separate rolling tests and all 253 existing PDF pages computationally
validated. All normal push hooks pass; current-head protected CI is pending.
The complete program requirement/evidence matrix above remains authoritative.

## Current Frozen Spectral Checkpoint

Loaded operators are published at `78253741f`. Private frozen spectra now
retain nonsymmetric stiffness, growing roots and defective eigenbases; both
state and original quadratic residuals must pass explicit numerical tolerances.
All 22 final Windows tests pass. The final Linux golf/API run passes 589
tests with two optional CAD skips and three optional-plugin warnings (245.81 s).
`FROZEN_SPECTRA.md` records the derivation, independent counterexamples and
remaining stability, moving-work, loaded grip/head and FRF qualification.
This is a T3 foundation, not a full epic or physical/acoustic qualification.

## Current Rigid Attachment Checkpoint

Frozen spectra are published at `cbc43b659`. Private rigid nodal attachment
now reuses the loaded-chain inertia samples without duplicating mechanics.
Twelve final Windows tests pass. `LOADED_BODY_ATTACHMENT.md` derives the
offset-body mechanics and independent continuum tip-mass controls, including
the principal-axis assumption that the first reference fixture violated.
The final Linux golf/API run passes 601 tests with two optional CAD skips and three optional-plugin warnings (292.46 s). Loaded grip impedance, moving-boundary work,
stability, full FRF/bandwidth convergence and physical/acoustic gates stay open.

## Current Frozen Tip Response Checkpoint

Rigid attachment is published at `42f229951`; frozen tip response is published
at `f37e37bd2676ba0d6cbd590a2d6199fa22a4ef6d` through all normal hooks, with the
remote SHA verified. The new private point-force/torque
response reuses loaded operators and shared clamped balance/domain validation.
`FROZEN_TIP_RESPONSE.md` derives six-axis ports, support transfer, continuum
compliance, scaling and coefficient-cancellation refusal. All 41 harmonic and
spectral Windows tests and nine API tests pass. Final Linux regression passes
620 tests (216.99 s), with two optional CAD skips and three optional-plugin
warnings; all nine manual gates pass. Issue #5072 records the published evidence;
this is not a stability, swing-trajectory, impact or acoustic qualification.

Inventory #5101 / PR #5103 corrects scientific-import detection on a separate
main-based branch, with every one of 410 reclassified candidates reviewed and
publication-blocked. Main's contact-completion #5088 merged; preserve it during
future T3 integration. UD #9787 / PR #9804 is subject to repeated concurrent
rollbacks of compatible pins and claim-preservation guards. Validated local
`6235789dc` is committed but not published; further shared-branch pushes are
paused while the competing writer is active. Remote `3837792da` does not carry
this task's validated scientific authority. The full goal remains active.

## Current Acoustic Boundary Repair

`Tools-impact-signals`, branch `fix/5074-waveform-spectral-contracts`, owns the
renewed #5074 lease (`impact-acoustics-01a07d8a-t5`, expiry 2026-09-08T20:58Z).
The initial boundary run reproduces 16 failures and seven passes. All 41
focused ingestion/spectral tests now pass (24.04 s), with actual hook mypy.
Real immutable samples, signed linear lag and shared segment detrending refuse
undefined/nonfinite estimates. Odd/even PSD agrees with explicitly configured
SciPy controls. The final Linux ingestion/report/API run passes 69 tests (152.47 s), with three
optional-plugin warnings. All nine final manual gates pass. The local checkpoint
is `31d8fa738`; protected renderer integration is committed at `4ccaab389`,
with all 69 Windows tests passing (13.41 s) and all nine manual gates passing.
Normal publication is running.
Its `SIGNAL_BOUNDARY_QUALIFICATION.md` and scoped vibroacoustics handoff retain
calibration identity, complex FRF, noise/uncertainty, radiation and physical/
blinded completion requirements. No acoustic experiment or prediction is made.

Inventory PR #5103 publishes `1c2c9b19d` through normal hooks. Renderer PR #5090
has now merged as protected main `b64a70f39`; integration into the inventory
branch preserves all 410 reviewed source hashes and recomputes only the
conflicted root handoff hash/count. All 95 post-integration tests pass (111.13 s) after an isolated rerun following
a source-read timeout. The unchanged 60-second test limit and all nine manual
gates pass. Integration `ace9a007b` is published through all normal hooks;
protected CI is running.

## Current Moving-Grip Kinematics Checkpoint

The private moving-anchor implementation retains finite relative orientation,
actual point separation, body twist derivatives and both physical work ports.
`MOVING_GRIP_KINEMATICS.md` gives the chain-rule derivation and independent
finite-pose derivative, force/moment and power controls. Eight new tests and
23 existing grip controls pass (6.25 s); the API adds only one private empty
entry. Full Linux golf/API regression passes 628 tests (174.65 s), with two
optional CAD skips and three unavailable-plugin warnings. An earlier run
timed out inside the unchanged rotating-rod convergence test; that isolated
test passes (54.72 s) with one BLAS thread, as does the broad rerun. The
60-second per-test limit, equations and tolerances are unchanged. This is a
kinematic foundation, not the finite grip law or its loaded balance/tangent.
Anchor work in time evolution, stability and physical/acoustic validation
remain open. All nine final manual gates, repository Ruff 0.14.10 (3,742 files)
and actual hook mypy pass. Implementation is published at
`0d45c4b7f32e0d2917c93e9ef6e492c032580dd7` through every normal commit/push
hook; the remote SHA is verified and issue #5072 records the evidence. The
derivation explicitly distinguishes invariant summed internal-port power
from observer-dependent individual anchor power: an inertial ledger requires
inertial motion states, or explicit frame-work accounting for a moving observer.

Acoustic repair PR #5106 publishes `3a9362530922346814e8be65a9b3fbd8ae95481d`.
Both Python matrices, quality/docs checks and the UpstreamDrift consumer pass.
The Gasification_Model consumer fails at private repository lookup before
tests; credential configuration remains unresolved. Inventory PR #5103's
single retry repeats the Python 3.12 worker crash in the deterministic/freshness
test; a replacement worker again passes it (43.34 s). No assertion mismatch is
reported. Local generator profiling is in progress; no timeout or expectation
is relaxed and no further speculative CI retry is issued.

## Current Finite-Grip Constitutive Checkpoint

After published moving kinematics `0d45c4b7f` and turnover `74b366903`, a
separate finite-coordinate law now maps supplied Gram factors to root and
anchor physical reactions. `FINITE_GRIP_RESPONSE.md` derives storage, loss,
observer-specific power and the remaining geometric preload tangent. Both
finite and existing local paths share `_grip_energy.py`; the small-rotation
public API is unchanged. Initial collection is RED (module absent); all
37 finite/moving/local grip tests pass (10.11 s), as do nine API tests (7.82 s).
An independent physical-pose energy gradient verifies finite elastic reactions;
an accelerating-observer control distinguishes invariant internal-port power
from each observer-dependent port. These are synthetic numerical controls.

Full Linux golf/API regression passes 634 tests (232.32 s), with two optional
CAD skips and three unavailable-plugin warnings, one BLAS thread and the
unchanged 60-second limit. Repository Ruff 0.14.10 passes (3,745 files), as
does actual three-module hook mypy. The API adds only two empty private module
entries. All nine final manual gates pass; implementation is published at
`28c45eb15f73ab7cfc551efbe97996b21ed22ccb` through every normal commit/push
hook, with its remote SHA verified. Loaded root balance
and tangent, anchor/frame work in evolution, stability and impact/acoustic
qualification remain required; #5072 stays open.

Inventory PR #5103 publishes the test-granularity adjustment at
`81b28da05006bb0d8fd1e98ad072f7d97c21be40` through all normal hooks. All 96
local combined tests pass. Its previously failing Python 3.12 unit shard
now passes on CI, as does the UpstreamDrift consumer; the remaining public
embedded/native checks are running. Private Gasification_Model job
`102225339951` fails at repository lookup before tests. The final classifier
must be integrated and the inventory regenerated before combined delivery.

## Measured Boundary Data Discovery

`DATA_CANDIDATES.md` now records primary KITopen hand-arm impedance metadata,
including translational and rotational experiments, selected cross-axis
responses and separate validation records. It preserves interpolation flags,
an angular/linear excitation-unit ambiguity, differing payload licenses and
participant/trial grouping needs. No payload is downloaded or fitted. The
RADAR landing page is refused by the web reader; no alternate payload-fetch
path is used to bypass it. These are candidates for measured boundary-model
checks, not golf impacts, full six-axis identification or sweetness evidence.
The model-parameter comparison record is also catalogued for later review.

## Current Grip-Supported Chain Checkpoint

After published finite response `28c45eb15`, stationary grip attachments now
couple at arbitrary shaft nodes with separate support reactions, combined
elastic storage and the finite preload derivative. An all-node root solve
shares the original bounded Newton/backtracking iteration; the clamped entry
retains its exact root constraint. Frozen M/G/C/K arrays preserve existing
distributed/head inertia and distinguish damping from gyroscopic transport.
`GRIPPED_CHAIN.md` gives equations, frame/domain contracts and numerical oracles.

Initial chain collection is RED (5.39 s); the first 28 coupled/clamped tests
pass. Dynamic assembly is separately RED before implementation (6.80 s).
All 38 final focused coupled/dynamic/clamped/grip tests pass (19.75 s), with
independent additive axial/torsional compliance, interior/repeated support,
nonplanar world force/moment, rotating rod and coordinate-energy checks.
Nine API tests pass (5.40 s); only four empty private-module entries are added.
Five-module hook mypy and repository Ruff 0.14.10 (3,751 files) pass.

Complete Linux golf/API regression passes 646 tests (187.68 s), with two
optional CAD skips and three unavailable-plugin warnings; one BLAS thread
and the unchanged 60-second deadline are retained. All nine manual gates pass
before and after edits; published at eeea63b47 with all normal hooks passing. Stability,
full grip-supported FRF/bandwidth and mesh/modal convergence, then nonlinear
time evolution with anchor/frame work and time convergence remain required.
Impact/acoustic/physical/blinded qualifications and protected inventory
integration remain open. This is coupled model progress, not full epic closure.

## Current Authority Coordination Boundary

A fresh GitHub check reports UD #9804 merged as
`736ec2189a479fc1a9a7b45078b4d7d371b2590a`, from
`b8da0c0240c20926f8fb7ba9e55a7e5613c5d78d`. That revision restores compatible
pins but differs from the preserved locally tested authority `6235789dc`.
The previous review found the update call bypasses the restored preservation
helper and the native record has a different digest/schema/profile. A merged
PR does not make local validation transfer to different bytes. Audit current
protected main in a fresh claimed worktree; do not edit or overwrite the old
shared branch. The Tools #5103/#5106 private lookup block remains separate.

## Finite-Support Frequency Response In Progress

The private response now retains every node and individual grip reaction,
with balance/domain rechecks and shared clamped numerical/point-port code.
`GRIPPED_FREQUENCY_RESPONSE.md` supplies the independent rod derivation and
TDD evidence: RED collection, 36 initial passes, 50 focused final passes.
All 665 Linux golf/API tests and nine manual/type/lint gates pass. Published at 12bcf3d83 with all normal hooks passing and remote SHA verified.
Stability and the full program remain open.

## Finite-Grip Spectra In Progress

Separate G/C spectra and the all-node wrapper now pass 49 focused controls,
including two-node axial characteristic roots, two convergent continuum poles,
critical and unstable modes, immutable inputs and numerical/domain refusal.
`GRIPPED_SPECTRUM.md` supplies equations and limits. All 692 Linux golf/API tests and nine final manual/type/lint gates pass.
Publication remains pending; explicit stability and the
full physical/acoustic program remain open. UD #9825 records the confirmed
protected-main claim reconciliation bug; #8920/#8556 remain parameter gates.
