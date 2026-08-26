# TOOLS-M4 Intrinsic Calibration Delivery Reconciliation

**Issue:** [#4714](https://github.com/D-sorganization/Tools/issues/4714)  
**Decision:** `blocked_dependency_reconciliation`  
**Audit base:** protected `main` at
`cff2909f1585273e10fa49165bfab8521e889da1` (tree
`f9f08e57496849c9a1e80e73ff25fd7a5d589f91`)  
**Runtime delivery:** not authorized  
**Pull request:** not authorized while the capacity drain prohibits broad CI

The companion
[`mocap_intrinsic_calibration_4714_reconciliation.json`](mocap_intrinsic_calibration_4714_reconciliation.json)
is the machine-readable admission record. This document explains the decision
and the implementation contract that a later, dependency-correct M4 branch must
satisfy. It does not publish a camera-calibration API and does not qualify any
camera or calibration target.

## 1. Authority and decision

Tools is the sole runtime and schema authority for shared markerless-mocap
calibration. UpstreamDrift and AffineDrift may consume immutable Tools releases;
they must not recreate or silently amend the shared contracts.

Protected Tools `main` contains no
`src/shared/python/sidekick/lab/mocap` package. M0 and M1 are therefore not
release facts. Their combined PR #4734 remains open and conflicting, and its
Upstream consumer check failed because a separate package was unavailable.
PR #4753 demonstrates a focused sparse-check repair, but its broad protected
checks were cancelled during the operational capacity drain and it is not
merged. M2 and M3 exist only as local candidates with no PR; M3 is a sibling of
M2 rather than a descendant. Building M4 on these candidates would turn a
private dependency stack into an undocumented public authority.

The only truthful current-main delivery is consequently this fail-closed
reconciliation record. No runtime file from candidate
`619b23f27548dbd821b511f27a02b084d9d2ac63` is copied here.

## 2. Candidate 619b audit

The earlier candidate is useful design evidence, not releasable code. Its four
runtime files implement pinhole/fisheye projection, nonlinear fitting,
uncertainty estimates, and quality summaries, with deterministic synthetic
tests. It nonetheless misses required authority and acceptance boundaries:

1. It imports M1 `CameraIdentity`, which is absent from protected main.
2. It neither consumes an M2 captured-frame/provider contract nor preserves M3
   timestamp and synchronization provenance.
3. It accepts correspondences directly; there is no calibration-pattern
   detector protocol, provider seam, or typed detector failure.
4. Its target points are assumed already expressed in the camera optical frame.
   Ordinary planar-board calibration instead needs a pose for each image.
5. It lacks a versioned JSON Schema, canonical golden serialization, public API
   baseline change, and downstream fixture.
6. High residuals degrade a reference flag rather than producing a configured,
   typed rejection. Outlier, duplicate, bound, non-finite, and resource-limit
   behavior is not a complete public contract.
7. The covariance surface does not yet document parameter order, units,
   identifiability, conditioning cutoffs, or positive-semidefinite validation.

Cherry-picking the commit would bypass M0--M3 and still fail issue acceptance.
The later implementation must be rebuilt after its immutable dependencies
exist.

## 3. Required dependency order

The order is part of the design contract, not project bookkeeping:

1. **R1 — repair qualification.** Protected-qualify and merge #4753, or carry
   its sparse consumer-contract correction into a fresh M0/M1 replacement.
2. **R2 — M0/M1 authority.** Rebuild from then-current protected main and merge
   camera identity, capability, units, frames, serialization, and public surface.
3. **R3 — M2 acquisition.** Rebuild on the exact M1 release; expose bounded
   `CapturedFrame` and source-provider protocols. Split the oversized candidate
   so each module has one reason to change.
4. **R4 — M3 timing.** Rebuild on the exact M2 release; preserve timestamps,
   clock domain, synchronization uncertainty, drop/duplicate facts, and source
   provenance.
5. **R5 — M4 calibration.** Add M2 and M3 as explicit #4714 dependencies and
   implement the contracts below from current protected main using TDD.
6. **R6 — consumer projection.** Only after protected M4 and M5 releases,
   publish immutable sanitized schemas/fixtures and qualify AffineDrift #3962.

## 4. Mathematical contract for the future M4

### 4.1 Coordinate convention

For target point \(P_j=[X_j,Y_j,0,1]^T\) in the target frame and image
\(i\), the solver must estimate a rigid pose \(T_{C\leftarrow T,i}\):

\[
  p_{ij}=T_{C\leftarrow T,i}P_j=[x_{ij},y_{ij},z_{ij}]^T,
  \qquad z_{ij}>0.
\]

Normalized coordinates are \(a=x/z\), \(b=y/z\), and \(r^2=a^2+b^2\).
The frame direction and handedness must come from M0; code must not infer them
from array shape or filename.

### 4.2 Pinhole and distortion models

For the Brown-Conrady model, a bounded initial public contract may use radial
coefficients \(k_1,k_2,k_3\) and tangential coefficients \(p_1,p_2\):

\[
L=1+k_1r^2+k_2r^4+k_3r^6,
\]
\[
a_d=aL+2p_1ab+p_2(r^2+2a^2),\quad
b_d=bL+p_1(r^2+2b^2)+2p_2ab.
\]

Pixel prediction with skew \(s\) is

\[
\hat u=f_x a_d+s b_d+c_x,\qquad
\hat v=f_y b_d+c_y.
\]

Fisheye projection must be a distinct tagged model, not a coefficient-count
guess. For \(\theta=\operatorname{atan}(r)\), one conventional polynomial is

\[
\theta_d=\theta(1+k_1\theta^2+k_2\theta^4+k_3\theta^6+k_4\theta^8).
\]

The schema must freeze the exact convention, coefficient order, supported
version, pixel-center convention, and units. Unknown model tags fail closed.

### 4.3 Optimization

Let \(q\) contain intrinsics, distortion, and one minimal pose parameterization
per accepted view. With observed pixel \(y_{ij}\), prediction \(h(q,P_j,i)\),
and declared observation weight \(W_{ij}\), solve

\[
q^*=\arg\min_{q\in\mathcal B}
\sum_{(i,j)\in\mathcal A}
\rho\!\left((y_{ij}-h(q,P_j,i))^T
W_{ij}(y_{ij}-h(q,P_j,i))\right).
\]

Here \(\mathcal B\) is an explicit finite parameter box, \(\mathcal A\) is the
deterministically ordered accepted-observation set, and \(\rho\) is a named,
versioned loss. The result must include iteration/evaluation counts, termination
reason, active bounds, final accepted/rejected observation IDs, and residuals.
Wall-clock timing is diagnostic only and may not affect returned science.

### 4.4 Uncertainty and identifiability

For residual vector \(r(q)\), weighted Jacobian \(J\), \(p\) free parameters,
and \(n\) scalar residuals, a local approximation is

\[
\hat\sigma^2=\frac{r(q^*)^TWr(q^*)}{n-p},\qquad
\Sigma_q=\hat\sigma^2(J^TWJ)^{-1}.
\]

An implementation may use an SVD/pseudoinverse only with a documented singular
value threshold and returned effective rank. It must reject \(n\le p\), rank
deficiency, non-finite matrices, excessive condition number, or materially
non-positive-semidefinite covariance. Parameter names and units must accompany
every covariance row and column. A covariance estimate is local model
uncertainty, not camera certification.

### 4.5 Quality and rejection

Quality must be a typed outcome: `pass`, `degraded`, or `reject`, plus stable
reason codes. At minimum the future contract must test image-plane coverage,
view count, pose diversity, depth sign, residual distribution, parameter-bound
contact, Jacobian rank/conditioning, duplicates, non-finite inputs, and declared
resource caps. Thresholds belong in a versioned configuration carried into
result provenance; implementations may not silently discard observations.

## 5. Provider and pattern-detection seam

The domain layer should depend on a small protocol, not OpenCV:

- input: bounded M2 `CapturedFrame`, M3 timing/provenance, target definition,
  and detector configuration;
- output: immutable ordered observations with point IDs, pixel coordinates,
  optional covariance, detector version, and explicit reject facts;
- failure: stable typed codes for unsupported format, target absent, ambiguous
  ordering, partial target, invalid covariance, cap exceeded, or provider error.

An OpenCV chessboard, circles-grid, ChArUco, or AprilTag adapter may implement
the protocol. Synthetic tests must use a deterministic in-memory provider so
the solver and schema are testable without hardware or optional native
dependencies.

## 6. Required TDD and release evidence

The later M4 RED suite must precede runtime and cover pinhole and fisheye
known-geometry recovery, perturbed synthetic uncertainty, canonical ordering and
serialization, detector substitution, degenerate geometry, outliers, high
residual, non-finite values, bounds, duplicates, resource caps, and a sanitized
consumer fixture. Numeric tolerances must state scale, units, and rationale.

GREEN on a feature branch is necessary but not sufficient. Completion requires
protected merge of M0--M3, exact-head protected M4 checks, immutable schema and
fixture SHAs, post-merge evidence, and AffineDrift parity against those released
artifacts. Hardware accuracy, real-shop calibration quality, and commercial
fitness remain unclaimed until physical qualification data is reviewed.

## 7. Current evidence and nonclaims

- Valid RED admission run: Tools Actions run `33020373400`, job
  `98349035900`, OGLaptop-3, explicit `pytest -n 0`; five tests failed because
  this ledger and handoff state did not yet exist.
- An earlier wrapper-diagnostic run `33020311403` stopped before collection on
  inherited pytest addopts and is not counted as RED.
- Auto-triggered workflow-lint runs `33020307540` and `33020363344` were
  cancelled immediately under the capacity-drain rule.
- No local pytest, lint, render, or build command was run.
- No runner, service, route, secret, issue state, PR, or protected branch was
  mutated by this audit.

