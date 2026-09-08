# Impact and Acoustics Program Execution

The user authorized completion of the full planned program, not merely its
planning deliverables. All three parent epics remain open. Physical and
perceptual validation cannot be inferred from numerical fixtures.

## Requirement and Evidence Matrix

| Slice                          | Issue / Delivery               | Evidence Required Before Completion                                                                                           | Current State                                            |
| ------------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Theory review                  | AffineDrift #4254 / PR #4258   | Corrected rendered theory, source ledger and inventory; protected delivery                                                    | PR open at restart; local review evidence exists         |
| Final theory synthesis         | AffineDrift #4255              | Qualified downstream results with uncertainty and limits                                                                      | Not started; depends on evidence                         |
| Rigid reference                | Tools #5069 / PR #5077         | Analytic tensor/impulse gates and protected provider delivery                                                                 | PR open; prior local 383-test verification               |
| Lumped qualification           | Tools #5071                    | Events, work/loss ledger, timeout/step contracts, law-consistent restitution, scaling counterexamples, parity and convergence | Implementation and numerical tests in progress here      |
| Distributed shaft/grip         | Tools #5072                    | Prestressed rotating operators, passive impedance, beam limits, frame agreement, modal/mesh/time/FRF convergence              | Not implemented by this program                          |
| Flexible contact               | Tools #5073                    | Off-center friction/contact and head/shaft modes; launch and ringdown; complete energy closure                                | Pending T3                                               |
| Acoustics                      | Tools #5074                    | Calibrated signals, identified transfer, qualified radiation and held-out validation                                          | Pending; generic audio tools are not sufficient          |
| Reports/surfaces               | Tools #5075                    | Versioned provenance reports, consumer compatibility and truthful UI integration                                              | Pending qualified provider tiers                         |
| Integration plan               | UpstreamDrift #9701 / PR #9706 | Source/state inventory and protected delivery                                                                                 | PR open; remote head changed since initial delivery      |
| Swing adapters                 | UpstreamDrift #9703            | Compatible rigid, elastic, prestress and wrench transfer on exact provider pin                                                | Pending provider contract                                |
| Counterfactual studies         | UpstreamDrift #9704            | Registered matched-state and matched-input studies, reproducible results, uncertainty                                         | Pending verified coupled model                           |
| Physical/perceptual validation | UpstreamDrift #9705            | Synchronized calibrated measurements, held-out validation, blinded sweetness analysis                                         | Data/equipment availability requested; no experiment run |

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

T1 PR #5077 remains blocked at `5932146f1`: the Python 3.11 rate shard and
both downstream consumer jobs are red; the aggregate Python 3.11 job reflects
its failed shard. The rate shard passed 2,889 tests with 29 skips, then failed
artifact upload with ETIMEDOUT; that failed workflow was retried. The UpstreamDrift
consumer passed 10 tests and failed the fresh-provider import check with missing
`src.shared.python.logging_pkg`. This is a real integration failure still to
resolve. Gasification_Model previously failed repository checkout. No protection
has been bypassed and no merge is claimed.
