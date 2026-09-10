# Calibration Numerical Recovery Handoff

## Identity

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
