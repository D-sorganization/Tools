# Reference Placement and Calibration Recovery Handoff

## Prescribed Load Continuation #5073

- Worktree: C:/Users/diete/Repositories/Tools-impact-load-history; branch feat/5073-load-history; implementation SELF; parent event PR#5152. No load-history PR yet.
- Adds explicit observer/time-covered additional force/couple callbacks through the existing point-load and moving-grip work ports. Original loads remain once; no new inertia or force potential is added.
- Missing-module RED then18 controls pass; expanded20 new controls and12 temporal controls pass before the old event-refinement test hits its unchanged60s timeout. Cause remains unestablished; no JUnit completed. Four changed modules pass both mypy modes. Source976d7ff43 is archived; All406 Linux controls pass in270.59s, coverage58.86% above unchanged20%; Windows406 also pass in169.77s.
- Details: docs/development/impact-acoustics/LOAD_HISTORY_DEVELOPMENT.md. Event#5152 has a separate CI typing repair in progress; annotation-only source839ebe083 is now incorporated; all35 affected event/load-history controls pass in42.90s. Final protected provider/wheel and measured/acoustic evidence remain open.
- Next action: publish the verified load-history continuation for protected review. Development-log entry DL-#5073. Preserve peer camera records and all original test limits.

## Impact Event Continuation #5151 / Parent #5073

- Worktree: `C:/Users/diete/Repositories/Tools-impact-events`; branch `feat/5073-contact-events`; current integration commit `SELF`; PR#5152 https://github.com/D-sorganization/Tools/pull/5152 at publishedd212714c1.
- Objective: integrate the separately qualified adaptive normal-contact implementation9a8241015 with published temporal PR#5149 at0d6b99430, retaining camera/CLI source and every pre-existing API record. The calibration records below are preserved as peer-owned context.
- Exact archived source4e1b19810 passes386 Windows and386 Linux controls. Linux coverage58.64% exceeds the unchanged20% floor. NORMAL_EVENT_RESULTS.json records source/JUnit hashes, RED failures and preset independent event/work refinements. Four production modules pass NumPy-aware mypy; all3870 files pass root Ruff/format checks.
- This integration changes no event/shaft/impact Python implementation or tests. All15 CLI/service/sidekick/golf/swing API controls pass in8.04s. Generated inventory and all nine governance gates pass; normal commit/push checks pass; protected CI/review remains required.
- Publication repair at `SELF`: the first push at50935aefc was refused by three isolated mypy no-any-return errors. Explicit response typing and builtin scalar returns now pass the same hook; all15 affected event tests pass in28.94s. No equation, tolerance or gate changed. The archived386 receipts retain their original tree identity.
- Dense root work is not a nonnegative loss certificate; roots report state/response, while endpoint work remains strictly validated. Sign-change searches can miss repeated roots inside a step. No force-maximum or physical/acoustic approval is implied.
- Open dependencies: protected reviews#5146/#5149, private consumer checkout404, final reviewed Tools/UpstreamDrift pin and installed wheel. Launcher#5144 must be preserved alongside camera main in that final pin. GUI lifecycle failures remain under#5114.
- Remaining science: finite-duration friction, face/hosel modes, changing applied force/torque, independent mesh/mode/general event convergence, matched interventions, measured force/spin/radiation and blinded perception.
- No user-owned changes are discarded. Fleet-policy main92283cf3f is now preserved without changing its managed blocks. This merge changes instructions and continuation records only; event Python remains9a8241015's qualified implementation. Next action: resolve protected CI/review on#5152; additional prescribed force/couple histories now continue separately in Tools-impact-load-history under#5073, before full friction coupling.
- Development-log entries: DL-#5151 and DL-#5073. Canonical detailed evidence: docs/development/impact-acoustics/NORMAL_EVENT_DEVELOPMENT.md and NORMAL_EVENT_RESULTS.json.

## Active Placement Work #5137

- Worktree: `C:/Users/diete/Repositories/Worktrees/Tools-calibration-numerics`.
- Branch: `feat/5137-reference-placements`; integrating numerical repair45f3bd8b9 and main2c9a8d6c9. Prior published head7b6fbcdbb remains on PR#5140 until this merge qualifies.
- PR: #5140 https://github.com/D-sorganization/Tools/pull/5140; published d129737273800c8cb3c31e28c38c421f05993a95 after all normal hooks.
- Session: capture-product-01a08427-reference-placements; lease/presence renewed through02:03UTC.
- New reference_placements.py defines ordered physical rectangle/line references,
  explicit camera/profile revisions, placement/view identities and immutable evidence.
- New placement_solver.py initializes a connected camera/placement graph and jointly
  refines camera and target poses with fixed intrinsics and an explicit world anchor.
  Held-out views do not seed or fit; cancellation raises; no physical accuracy claim.
- All101 mocap/authority/API tests pass in17.03s;24 numerical/reference tests also pass on each actual OpenCV4.13/5.0 runtime. Independent synthetic views,
  noise, nonidentity anchor, disconnected graph, missing profiles, repeated placements,
  invalid point IDs/pixels, mutable inputs, cancellation and indirect camera connections
  are covered. Scoped Ruff/format/mypy and handoff/manual gates pass; normal commit/pre-push hooks pass; remote CI and protected merge remain.
- API baseline adds only the two new modules; prior entries are unchanged.
- Remaining before publication: generated inventory and publication gates; root handoff/SPEC/DL-#5137 now updated,
  full mocap regression, actual OpenCV4/5 runs and normal hooks. Do not publish without
  qualifying the adopted compatibility correction from the separate repair worktree below. That correction is now copied exactly into this owned branch;24 numerical/reference tests pass on4.13 (9.35s) and5.0 (7.28s).

Tools PR#5136 repair is now on `fix/5132-calibration-numerics` in
`Worktrees/Tools-calibration-opencv5`. Commit e92cacd3f supports the OpenCV5 iterative
API consolidation; all12 numerical tests pass on actual4.13 and5.0. Published e92cacd3f passes Python3.11/3.12 shared CI. Three pre-checkout permission failures are under #5139; current host ownership is correct and unchanged failed runs were retried.

Capture UX#9917, catalog#9919 and reference expansion#9918 are merged. My Clubs
#9923 is published as a draft and integrating current main. Everyday calibration,
club/wizard completion and fleet adoption remain active. The live test app remains
in capture-setup and must be preserved.

## Merge Qualification in Progress

The #5136 merge preserves impact#5133. Only the root handoff/digest conflicted;
combined handoff is149 lines. Numerical source changes are exactly the already
qualified repair. Regenerate inventory, rerun placement/API/calibration checks and
normal hooks before publication. Earlier rate timeouts and private Gasification
checkout failure must be qualified on fresh CI; no exclusions or thresholds change.

## Numerical Recovery History

- Repository: D-sorganization/Tools
- Worktree: `C:/Users/diete/Repositories/Worktrees/Tools-calibration-opencv5`
- Branch: `fix/5132-calibration-numerics`
- Baseline: `421889407a62fc541e8adc017a86dec2a7ab4f32`
- Implementation commit: `SELF`
- Issue: #5132; parent #4706; consumers UpstreamDrift #9897/#9898/#9899
- PR: #5136 OPEN: https://github.com/D-sorganization/Tools/pull/5136
- Session: `capture-product-01a08427-calibration-numerics`

## Objective and Scope

Qualify the existing shared numerical camera APIs before common-reference
calibration adopts them. Preserve the user's full goal: everyday references and
repeat placements, optical settings, club catalog/player bag, interactive wizard,
and the remaining fleet communication adoption. This repair does not close those
requirements. UpstreamDrift profile #9910 and capture UX #9917 are merged. Catalog
#9919 and reference expansion #9918 are merged; bag #9923 is published as a draft.

## Findings and Implementation

The prior PnP implementation caught every exception and returned identity rotation
with translation `(0, 0, 2)`. Bundle adjustment returned the input layout unchanged.
Inverse rays ignored nonzero lens distortion, and rational projection fell back to
undistorted coordinates. Six independent characterization cases reproduced these
failures; perturbed-layout residual stayed at 22.4076 pixels.

The existing public entry points now call `calibration_numerics.py`. OpenCV handles
declared pinhole/rational/fisheye distortion and PnP; iterative inverse rays must
reproject to the requested pixel. Camera poses use explicit world-to-camera rigid
transforms. SciPy minimizes robust pixel residuals against **known fixed world
targets**. This objective separates by camera: each pose block is optimized, with
only an explicitly named gauge camera held fixed. Unknown landmark coordinates
and moving target placements are not estimated by this API.

Sources: [OpenCV PnP](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)
and [SciPy Least Squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
The code uses APIs supported by the repository's dependency floor; no new package
dependency is introduced. Helper implementation exports no new public symbols;
the API baseline adds only the new module's empty surface.

## Validation

- Six numerical regressions failed before changes.
- Twelve numerical regressions pass: independent OpenCV projection checks,
  distortion inversion, skew, missing/failed solvers, collinear/nonfinite inputs,
  fixed gauge, missing observations, positive depth, fisheye pose recovery, and
  recovery from perturbed multi-camera poses checked on held-out points.
- Calibration/extrinsic contracts plus initial numerical tests: 19 passed.
- Final mocap/authority/API run: 89 passed (39 existing import-alias warnings).
  The API baseline adds only the helper module's empty surface and preserves all
  previous entries. Final run completed in19.12seconds.
- Focused Ruff and mypy pass for the three production modules.
- Reviewed inventory delta: one provisional helper, two changed implementation hashes,
  and automatic test associations to existing transform modules. No prior modules
  or classifications are removed. Textbook/exemplar/render checks pass; final
  inventory freshness/governance recheck remains.
- Clean git export of684fbfb06be6d066be2a005c752b63d63572f900:25 calibration tests
  passed. Pre-push found the older mypy needs an explicit TypeAlias declaration;
  that annotation is corrected and both the actual hook and local mypy pass.
- Exact14b35b57bc4ed44b607a21b6bc23bc7be7cd5c71 passed all normal push hooks and25 clean-export tests. CI then reproduced3 failures with OpenCV5.0.0.93: undistortPointsIter was folded into undistortPoints. SELF selects the supported iterative entry point without relaxing convergence checks. All12 numerical tests now pass separately on actual OpenCV4.13 and5.0. Both share the same test source and ordinary repo conftest; no test is skipped. Refreshed publication gates and exact-head protected CI remain.
- Temporary clean export is retained at TEMP/capture-tools-clean-05c2c95d640a468099b6b38fe505429e;
  automatic approval review rejected its removal with blocked-by-policy. Do not
  work around that rejection; logs remain in TEMP/capture-tools-clean-qualification.log.

## Limits and Coordination

Synthetic recovery is not physical camera qualification or publication approval.
Intrinsic uncertainty, planar ambiguity, unknown reference placements, target
scale/world axes and held-out physical evidence remain explicit consumer work.
The module inventory retains provisional calibration status and #5132 tracks its
qualification. The registered D-plane calculation is unchanged. Only the root
handoff digest is refreshed; historical check/artifact evidence stays historical. The unchanged quality categories
describe residual/count thresholds and do not certify physical observability.

The active Tools CLI PR #5121 was notified at
https://github.com/D-sorganization/Tools/pull/5121#issuecomment-5608073370.
Preserve its files and the concurrent impact-acoustics program. Fleet adoption was
39/41 at 19:53UTC; Tools and Gasification_Model had unresolved replacement queues.
Do not reopen their closed policy PRs repeatedly. Central issue #1579 has receipts.

## Next Steps

1. Finish OpenCV4/5 compatibility publication checks and require green PR#5136 before merging.
2. Publish #5132 through protected review and qualify the exact merged Tools SHA.
3. Adopt that SHA in the isolated UpstreamDrift capture setup worktree; implement
   common references and repeated placements through the shared authority.
4. Complete the club and wizard epics and fleet rollout before closing the goal.

## Change Log

- `SELF`: replace fabricated/no-op numerical behavior with tested backend recovery.

## Current Goal Additions

UpstreamDrift#9913 shipped in #9917 at8fce9f238 with445 capture/parity tests. Everyday calibration #9897, club/wizard #9902/#9906 and final fleet adoption remain open.

Tools#5137 moving-reference work is published in PR#5140 at7b6fbcdbb5e72a61ebcc8e71c9e50644c7e0a2cd, Worktrees/Tools-calibration-numerics.24 combined numerical/placement tests passed on OpenCV4.13/5;101 broader mocap/authority/API tests passed. Its main integration remains separate.

## Main Integration Receipt

Main2c9a8d6c9 includes merged impact PR#5133. Source merge was clean; only the
root handoff digest conflicted and was recomputed from the combined149-line
handoff. Inventory merge driver ran normally; final generator/freshness and hooks
remain.19 calibration contracts/numerics pass after merge;12 numerical tests also
pass against actual OpenCV5.0.0 via TEMP/capture-opencv5-runtime/Lib/site-packages.
The retained environment has no Python launcher, so global Python was used with
that isolated dependency path; no environment was recreated or removed. Prior protected aggregate
tests failed because the rate shards timed out at99%; private Gasification checkout
also fails before tests. No tolerance, timeout or gate was relaxed. A fresh run will
qualify the merged tree. User was asked to have the credential owner restore private
read access; this App cannot inspect/update Actions secrets.

## Combined Main Qualification Receipt

Merged repair45f3bd8b9/main2c9a8d6c9:97 mocap tests plus4 authority/API tests
pass;24 numerical/placement tests also pass on actual OpenCV5.0.0 using the
retained dependency directory. Handoff149-line/digest and module inventory
freshness checks pass. No solver or gate changed during merge. Normal commit
and push hooks remain before updating PR#5140.

Provider coordination: dieterolson closed standalone launcher PR#5144 unmerged at2026-09-10T04:26:49Z. Preserve this closure. Its exact correction is incorporated in combined provider#5141 atc80f2cf34; the context task retains final downstream pin ownership. No standalone reopening is requested. Load-history review#5153 passes406 Windows/Linux controls and35 after the annotation-only event repair; six affected production modules pass NumPy-aware mypy. Next action is normal publication and protected review.

Reference-integrity adoption at SELF: the two test files are byte-identical to event commit847e6927c. SI root-residual and forward-time checks strengthen the independent oracle, with a perturbed-root tripwire. All21 load-history/oracle controls pass in5.07s and the unchanged assertion gate passes. Production remainsc4e6d584d; the406-platform archive and35 annotation-integration receipts retain their source identities. Next action: publish the review child#5153.
