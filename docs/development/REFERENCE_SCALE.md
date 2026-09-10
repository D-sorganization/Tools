# Established-layout ruler scale (#5168)

Turnover: PR #5169 merged implementation/API repair3d7beb203 asd4ab52a926. All old
API entries are unchanged; exactly three module entries were added. Normal hooks
passed. CI's remaining private Gasification checkout failure occurs before tests
(job102883109039/run34479121462); coordinate access without bypassing the gate.
No consumer/UI completion is claimed. See the root and development handoffs.

The public entry point is `sidekick.lab.mocap.reference_scale.estimate_reference_scale`.
It accepts an immutable snapshot of an existing metre-coordinate camera layout,
its pinhole lens/zoom profile identities, two-endpoint reference definitions and
stationary placements observed by at least two cameras. It returns a candidate
layout with a new identity; it does not overwrite calibration or approve capture.

## Method and review contract

Each endpoint is undistorted through the existing lens contract and reconstructed
through the canonical multi-view triangulation API. Raw-image reprojection error
and ray parallax must pass explicit thresholds. For reconstructed lengths `m_i`
and known lengths `L_i`, the equal-weight least-squares solution is
`s = sum(m_i * L_i) / sum(m_i * m_i)`. Lengths are metres and `s` is dimensionless.
Held-out placements do not enter this objective. Every fit and held-out length
must satisfy the selected relative-error limit after fitting.

Camera centers transform as `C_new = A + s * (C_old - A)`, where `A` is the
explicit fixed world anchor. Rotations and lens profiles remain unchanged.
Camera translations are regenerated from those centers using the existing
world-to-camera convention. Scaling scene points about the same anchor preserves
their image projections. Consumers must review the evidence before saving a new
calibration revision, and invalidate dependent reconstruction/model results when
that revision changes. Previous files and their provenance must remain available.

The default software rejection limits are one degree of usable parallax, two
pixels of reprojection error and five percent relative length error. They are
not measured camera accuracy. Processing is bounded to 12 cameras and 32 distinct
placements. Duplicate views, identical repeated images, stale zoom profiles,
unusable geometry, absent fitting evidence and inconsistent lengths are rejected.

## Limits

This method assumes correct relative camera poses and lens calibration. It cannot
initialize camera orientation from a ruler, detect unreported camera movement,
or repair lens distortion. Optical zoom changes require a matching lens profile
and fresh observations. Current support is pinhole profiles, including their
supported distortion models. A stationary ruler must identify the same endpoints
across views; timestamps alone cannot establish that the object was stationary.

Reported repeatability is the sample standard deviation of individual length
ratios divided by the square root of the number of fitting placements. It is
absent for one fitting placement, and excludes reference-length, lens and manual
pixel-picking uncertainty. Unit pixel covariances are algorithmic triangulation
weights, not calibrated measurement noise. A missing holdout is explicitly
reported. Synthetic tests do not constitute physical-camera qualification.

## Validation and publication state

`tests/shared/python/sidekick/lab/mocap/test_reference_scale.py` uses independent
OpenCV projections with distortion to check metric recovery, anchored projection
invariance, noisy placements, holdout isolation, endpoint ordering and rejection
contracts. The initial missing-module test failed before implementation. The
expanded scale and existing placement suite passed 27 tests; changed source
passed Ruff and mypy. Exact metric recovery uses a preset `1e-9` tolerance;
projection preservation uses `1e-10`. Neither threshold was relaxed.

Generated module inventory provides provisional calculation discovery for the
three implementation modules. This development note is not engineering-manual
authority. A fully traced textbook chapter, measurement uncertainty qualification
and publication review remain unapproved; no manual release promotion is claimed.
The UpstreamDrift worker, review UI and revision integration remain tracked by
UpstreamDrift #9899 after the canonical provider is published.
