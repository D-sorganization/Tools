# Calibration Numerical Recovery Handoff

## Identity

- Repository: D-sorganization/Tools
- Worktree: `C:/Users/diete/Repositories/Worktrees/Tools-calibration-numerics`
- Branch: `fix/5132-calibration-numerics`
- Baseline: `421889407a62fc541e8adc017a86dec2a7ab4f32`
- Implementation commit: `SELF`
- Issue: #5132; parent #4706; consumers UpstreamDrift #9897/#9898/#9899
- PR: pending
- Session: `capture-product-01a08427-calibration-numerics`

## Objective and Scope

Qualify the existing shared numerical camera APIs before common-reference
calibration adopts them. Preserve the user's full goal: everyday references and
repeat placements, optical settings, club catalog/player bag, interactive wizard,
and the remaining fleet communication adoption. This repair does not close those
requirements. UpstreamDrift draft #9910 contains the profile UI and startup repair;
its latest published head is `b743eef0914073c9433c2816078d337e5d60b6e8`.

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
- Normal commit/push hooks, clean-export qualification and protected CI remain.

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

1. Finish numerical/API/manual inventory validation and review generated deltas.
2. Publish #5132 through protected review and qualify the exact merged Tools SHA.
3. Adopt that SHA in the isolated UpstreamDrift capture setup worktree; implement
   common references and repeated placements through the shared authority.
4. Complete the club and wizard epics and fleet rollout before closing the goal.

## Change Log

- `SELF`: replace fabricated/no-op numerical behavior with tested backend recovery.
