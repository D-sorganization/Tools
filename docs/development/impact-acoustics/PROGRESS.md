# Impact and Acoustics Program Execution

The user authorized completion of the full planned program, not merely its
planning deliverables. All three parent epics remain open. Physical and
perceptual validation cannot be inferred from numerical fixtures.

## Requirement and Evidence Matrix

| Slice                          | Issue / Delivery               | Evidence Required Before Completion                                                                                           | Current State                                                               |
| ------------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Theory review                  | AffineDrift #4254 / PR #4258   | Corrected rendered theory, source ledger and inventory; protected delivery                                                    | Merged PR #4258 at `1ce02d7ae4d916d4c598b62be78cefa83452aaa7`               |
| Final theory synthesis         | AffineDrift #4255              | Qualified downstream results with uncertainty and limits                                                                      | Not started; depends on evidence                                            |
| Rigid reference                | Tools #5069 / PR #5077         | Analytic tensor/impulse gates and protected provider delivery                                                                 | PR open; prior local 383-test verification                                  |
| Lumped qualification           | Tools #5071 / draft PR #5082   | Events, work/loss ledger, timeout/step contracts, law-consistent restitution, scaling counterexamples, parity and convergence | Implemented in `31ebe4993`; draft PR #5082 published; local push gates pass |
| Distributed shaft/grip         | Tools #5072                    | Prestressed rotating operators, passive impedance, beam limits, frame agreement, modal/mesh/time/FRF convergence              | Not implemented by this program                                             |
| Flexible contact               | Tools #5073                    | Off-center friction/contact and head/shaft modes; launch and ringdown; complete energy closure                                | Pending T3                                                                  |
| Acoustics                      | Tools #5074                    | Calibrated signals, identified transfer, qualified radiation and held-out validation                                          | Pending; generic audio tools are not sufficient                             |
| Reports/surfaces               | Tools #5075                    | Versioned provenance reports, consumer compatibility and truthful UI integration                                              | Pending qualified provider tiers                                            |
| Integration plan               | UpstreamDrift #9701 / PR #9706 | Source/state inventory and protected delivery                                                                                 | Merged PR #9706 at `dbc6727aa4f0d422b7adaf6957e658e8997f7f29`               |
| Swing adapters                 | UpstreamDrift #9703            | Compatible rigid, elastic, prestress and wrench transfer on exact provider pin                                                | Pending provider contract                                                   |
| Counterfactual studies         | UpstreamDrift #9704            | Registered matched-state and matched-input studies, reproducible results, uncertainty                                         | Pending verified coupled model                                              |
| Physical/perceptual validation | UpstreamDrift #9705            | Synchronized calibrated measurements, held-out validation, blinded sweetness analysis                                         | Data/equipment availability requested; no experiment run                    |

## Current Implementation: IA-T2

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

Finish the full provider checks, source/test inventory, handoff hashes and T2 PR.
Resolve review/CI on the existing foundation PRs through normal protections,
then proceed to #5072. Recheck GitHub heads before incorporating remote edits.
Do not silently overwrite the later UpstreamDrift PR head.

The full goal is active. A passing lumped model does not complete the distributed
shaft, acoustic, experimental, consumer or final theory milestones. Data and
hardware availability remain unknown; do not invent measurements or approvals.

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

## Provider Delivery Status

T1 PR #5077 remains blocked at `5932146f1` by both downstream consumer jobs.
The rate shard and aggregate Python 3.11 job passed after retry. The rate shard passed 2,889 tests with 29 skips, then failed
artifact upload with ETIMEDOUT; that failed workflow was retried. The UpstreamDrift
consumer passed 10 tests and failed the fresh-provider import check with missing
`src.shared.python.logging_pkg`. This is a real integration failure still to
resolve. Gasification_Model previously failed repository checkout. No Tools protection has been bypassed and no Tools merge is claimed.

## Delivery Environment and Next Integration Audit

The first push failed in system Python 3.13 when a MuJoCo plugin crashed a
parallel unit-test worker. Python 3.12 imported MuJoCo successfully but lacked
pytest-qt. An ignored worktree `.venv` now uses Python 3.12 system packages plus
pytest-qt 4.5.0; serial offscreen execution passed 1,614 tests, with 29 skips,
9 expected failures and one existing unexpected pass (161.32 s). Use this
process-local environment for push hooks: prepend `.venv/Scripts` to PATH,
set PYTEST_ADDOPTS=-n0 and QT_QPA_PLATFORM=offscreen. No hook was bypassed.

AffineDrift A1 and UpstreamDrift U1 have been protected-merged; their parent
issue checklists were updated and remain open. The later U1 branch commit was
a merge from main containing unrelated motion-capture changes; it was preserved.

The next inventory refinement must explicitly preserve the existing Tools
`swing_sim.impact_interval` facade, its rigid full-inertia club/ball state,
FREE/PINNED/TORSIONAL_GRIP boundaries, friction, trace queries and audit wire.
Its fixed-step solver is a useful T4 integration point, not a distributed shaft
or radiation solver. Read its contact geometry, termination and energy accounting
before extending it, and use the T1 tensor reference for independent limits.

The failing downstream import occurs in UpstreamDrift `cli_utils.py`, which
imports Tools-owned logging via `src.shared.python.logging_pkg.logging_config`.
UpstreamDrift has no physical logging_pkg there; the Tools alias finder deliberately
limits aliases under an external src namespace. Investigate an explicit canonical
provider import and fresh-process consumer tests, preserving downstream ownership.
Do not broaden namespace takeover to hide the failure. T3 has not been claimed yet.

## Current Checkpoint: T2 Published, Compatibility Reproduced

Draft Tools PR #5082 is open on `fix/5071-qualified-impact-coupling`; implementation
commit `31ebe4993`, delivery record `eb78179b5`. All local push hooks pass in the
qualified environment. First CI snapshot has running/queued checks, so no
protected delivery or completion of #5071 is claimed. Default dt remains
1e-7 s; the event/audit solver costs more than the old Euler loop. Keep this
runtime tradeoff visible when qualifying interactive consumers in T6.

Focused prerequisite UpstreamDrift #9735 is attached under #9700. Claim check
was free and the codex lease succeeded (session impact-acoustics-01a07d8a-provider,
expiry 2026-09-08T04:17:08Z). New worktree:
`C:/Users/diete/Repositories/UpstreamDrift-impact-provider`, branch
`fix/9735-impact-provider-imports`, base `dbc6727aa`. It has no production edits.
The existing failing fresh-provider contract was reproduced there: 1 failed in
4.67 s with the same missing logging_pkg exception as CI. Test process used the
Tools `.venv` Python, explicit TOOLS_REPO_ROOT and the same four provider
PYTHONPATH entries as CI, no initialized vendor, serial/offscreen execution.
A 90-second outer subprocess limit guarded collection; the failure returned
normally. Log: system temp `impact-provider-repro.log`.

Further source inspection found an existing installed/vendored fallback in
UpstreamDrift `src/__init__.py`. The canonical `shared.python` name can itself
resolve to the downstream shadow during initialization; examine that ordering
and the tests/conftest path setup before proposing a fix. A simple import rewrite
is not yet established as sufficient. Reuse and qualify the existing fallback,
retain ownership boundaries, and test both real layouts. No T3 implementation
has started, and all remaining scientific/empirical milestones remain active.
