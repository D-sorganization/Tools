# TOOLS-M5 Extrinsic and Flexible-Layout Delivery Reconciliation

**Issue:** [#4721](https://github.com/D-sorganization/Tools/issues/4721)  
**Decision:** `blocked_dependency_reconciliation`  
**Audit base:** protected `main` at
`cff2909f1585273e10fa49165bfab8521e889da1` (tree
`f9f08e57496849c9a1e80e73ff25fd7a5d589f91`)  
**Runtime delivery:** not authorized  
**Physical calibration authority:** not claimed

The companion
[`mocap_extrinsic_layout_4721_reconciliation.json`](mocap_extrinsic_layout_4721_reconciliation.json)
is the machine-readable admission record. This document defines the dependency
repair and future implementation contract. It does not publish an extrinsic
solver, approve a camera layout, or establish that a shop installation is safe
for motion-capture inference.

## 1. Authority and dependency decision

Tools owns the shared runtime and schema. Protected main contains no markerless
mocap package. M0/M1 PR #4734 remains open and conflicting; M2 and M3 are
local-only sibling stacks with no PR; M4 has only a current-main fail-closed
audit and no protected runtime. Issue #4721 explicitly depends on M3 and M4.
Those facts make M5 runtime inadmissible.

Prior M5 candidate `f371ce4f06ba9b904d1719a99221cffab44b4020`
(tree `569e6d703f0183b18bfe070a689b63022dbd00fa`) is ten commits
ahead of and ten protected-main commits behind merge base
`e76a7a21408db9ba55b18959f0fc513bf63ec579`. It descends from prior
M4 candidate `619b23f27548dbd821b511f27a02b084d9d2ac63`, but not from the
M2 or M3 candidates and not from the current M0/M1 PR head. It has no PR.
Cherry-picking it would bypass the exact dependency chain and import stale
private interfaces, so this branch carries none of its runtime paths.

## 2. Adversarial audit of candidate f371ce4

The candidate contains useful exploratory work: explicit
`T_world_from_camera` transforms, known-world reference points, weighted
residuals, a soft-L1 fit, condition/rank checks, covariance, arbitrary camera
records, and simple movement thresholds. Its synthetic tests exercise three
cameras, one outlier, disconnected point sets, degenerate geometry, and layout
changes. These are design inputs, not release evidence.

The central mathematical limitation is that the alleged global bundle
adjustment is separable by camera. All 3-D reference points are supplied in the
world frame, and each residual depends on only one six-parameter camera block.
No target pose or other shared unknown couples the camera blocks. The
optimization is therefore multiple absolute PnP refinements concatenated into
one vector, not the flexible-layout learning problem required by #4721.

Additional gaps are release-blocking:

1. The initializer requires a rank-eleven DLT design, which excludes the planar
   boards commonly used for flexible camera placement.
2. “Robust PnP” evaluates all correspondences and each leave-one-out set. It is
   not a bounded multi-outlier consensus method and emits no inlier/rejection
   authority.
3. Camera graph connectivity is required even though known world points make
   each camera independently observable; this masks rather than models the
   coupling needed for unknown target poses.
4. Soft loss downweights observations silently. Results do not identify which
   observations were accepted, rejected, or ambiguous and why.
5. Covariance is conditional on exact intrinsics and the supplied fixed-world
   geometry. The contract does not state that limitation, expose cross-camera
   covariance meaningfully, validate PSD, or define gauge-aware rank handling.
6. Movement monitoring compares caller-provided poses with fixed scalar
   thresholds. It has no relocalization observation/provider, timestamp,
   covariance, stale-evidence policy, or provenance linkage.
7. An invalid result can retain the prior active layout ID if the caller omits a
   new ID. There is no governed recalibration, approval, activation, rollback,
   or immutable segment transition.
8. No JSON Schema, canonical serialization, public export baseline, downstream
   fixture, cancellation contract, or complete resource cap is delivered.

## 3. Required R1–R7 reconstruction order

1. **R1 — sparse consumer repair.** Protected-qualify and merge #4753, or
   incorporate its repair into a fresh M0/M1 replacement.
2. **R2 — M0/M1 authority.** From then-current protected main, freeze frame,
   transform, unit, quaternion, uncertainty, provenance, availability, error,
   serialization, and qualification contracts; protected-merge them.
3. **R3 — M2 acquisition.** Rebuild on the exact M1 release and protected-merge
   bounded captured-frame and target-observation provider seams, lifecycle,
   cancellation, and source identity.
4. **R4 — M3 timing.** Rebuild on the exact M2 release and protected-merge clock
   domain, skew/drift uncertainty, sequence, drop/duplicate, and crash-safe
   recording provenance.
5. **R5 — M4 intrinsics.** Amend M4 to consume M2/M3, then protected-merge
   detector/provider, intrinsic, rejection, uncertainty, quality, schema, and
   sanitized fixture authority.
6. **R6 — M5 implementation.** Rebuild from then-current protected main and the
   exact M4 release using contract-first tests. Do not cherry-pick `f371ce4`.
7. **R7 — consumers.** After protected M5 release, project immutable schema and
   sanitized fixture identities into AffineDrift #3962 and qualify parity before
   any downstream layout claim.

## 4. Coordinate and gauge contract

Every transform must state direction. Let \(T_{A\leftarrow B}\) map coordinates
from frame \(B\) into frame \(A\). For camera \(c\), capture \(k\), and metric
target point \(P^T_j\), the future prediction is

\[
\hat u_{ckj} = \pi\!\left(K_c,D_c,
T_{C_c\leftarrow W}\,T_{W\leftarrow T,k}\,P^T_j\right),
\qquad
T_{C_c\leftarrow W}=T_{W\leftarrow C_c}^{-1}.
\]

Here camera poses \(T_{W\leftarrow C_c}\) and per-capture target poses
\(T_{W\leftarrow T,k}\) are shared unknowns. Their joint appearance couples
cameras that observe common target placements. The target definition supplies
metric scale; its coordinate convention and digest are provenance.

This state has a six-degree rigid gauge: applying the same world transform to
every camera and target pose leaves pixels unchanged. The schema must select
exactly one versioned gauge policy, such as fixing an anchor camera to identity
or fixing one approved target pose. It must record the anchor and exclude its
fixed coordinates from free-parameter covariance. Hidden post-solve alignment
is not acceptable. A free-scale target adds a scale gauge and must fail unless a
metric baseline or target dimension is authoritative.

## 5. Initialization and robust estimation

Planar targets require homography/IPPE-style pose initialization; non-planar
targets may use a documented PnP/DLT family. The public initializer contract
must state supported geometry, chirality/depth tests, minimum point layout,
normalized thresholds, deterministic ordering and seed, maximum hypotheses,
timeout/cancellation behavior, and stable failure codes.

For observation \(z_i\), prediction \(h_i(q)\), and declared covariance
\(\Sigma_i\), define the whitened residual

\[
e_i(q)=L_i^{-1}(z_i-h_i(q)),\qquad L_iL_i^T=\Sigma_i.
\]

The future bundle adjustment solves

\[
q^*=\arg\min_{q\in\mathcal B}\sum_{i\in\mathcal A}
\rho\!\left(e_i(q)^T e_i(q);\tau\right),
\]

where \(\mathcal B\) is a finite parameter domain, \(\rho\) and scale \(\tau\)
are versioned, and \(\mathcal A\) is a deterministically ordered accepted set.
Robust loss alone is not rejection. The result must return every input
observation with accepted/rejected/ambiguous status, reason code, residual, and
provenance. Iteration count, function evaluations, active bounds, termination,
and cancellation state are required result facts.

## 6. Observability and uncertainty

Admission needs both structural and numerical tests:

- the camera–capture bipartite graph must connect every solved camera and target
  pose to the declared gauge anchor;
- target geometry must support the chosen initializer;
- views need sufficient depth, coverage, baseline, and orientation diversity;
- the gauge-reduced Jacobian must have full effective rank;
- conditioning, leverage, residuals, and bound contact must meet versioned
  policy.

For weighted residual Jacobian \(J\), \(n\) scalar residuals, and \(p\) free
parameters after gauge removal,

\[
\hat\sigma^2=\frac{r(q^*)^TWr(q^*)}{n-p},\qquad
\Sigma_q=\hat\sigma^2(J^TWJ)^{-1}.
\]

Any SVD/pseudoinverse must publish its threshold and effective rank. Covariance
must carry parameter names, transform direction, tangent-space convention,
units, gauge policy, condition number, intrinsic-uncertainty policy, and PSD
validation. Per-camera marginal and cross-camera blocks must be extractable.
If intrinsics are treated as exact, the result must say “conditional on fixed
intrinsics”; otherwise their covariance must be propagated or jointly modeled.

## 7. Camera motion and recalibration

Movement cannot be inferred by comparing unaudited poses. A relocalization
provider must emit observations tied to M2 camera identity, M3 time/sync facts,
M4 intrinsics, target digest, and input hashes. Relative pose change should be
evaluated on \(SE(3)\), for example

\[
\Delta T_c=(T^{base}_{W\leftarrow C_c})^{-1}
T^{meas}_{W\leftarrow C_c},
\]

with translation and rotation evidence compared against both policy and
uncertainty. Missing cameras, unexpected cameras, stale evidence, ambiguous
relocalization, target movement, frame mismatch, and threshold exceedance must
all invalidate inference authority with distinct reason codes.

Recalibration is a governed state machine, not a boolean. At minimum it must
distinguish `valid`, `movement-suspected`, `invalid`, `calibrating`,
`candidate`, `approved`, `active`, `rejected`, and `rolled-back`. Activation
creates an immutable layout segment with predecessor, evidence, approver,
timestamps, schema/solver/config versions, and output digest. Invalidation may
never silently keep the prior segment active.

## 8. Required TDD and release evidence

Future M5 RED tests must cover arbitrary-N and two-camera recovery, planar and
non-planar initialization, gauge invariance, transform composition, multiple
outliers, accepted/rejected observation identity, disconnected/scale-free/
collinear/low-baseline geometry, uncertainty coverage, covariance permutation,
camera addition/removal/movement, stale/ambiguous evidence, recalibration
approval/rollback, caps, cancellation, canonical serialization, and an immutable
AffineDrift fixture.

GREEN on a private branch is necessary but not sufficient. Completion requires
protected M0–M4 releases, exact-head M5 checks, schema and fixture identities,
review and protected merge, post-merge qualification, Affine parity, and later
physical shop-layout evidence. No hardware accuracy or commercial fitness is
claimed by this audit.

## 9. Current evidence and nonclaims

- Valid RED: Tools Actions run `33020976236`, job `98350987415`,
  OGLaptop-2, explicit `pytest -n 0`; five tests failed because the ledger and
  required handoff state did not exist.
- Auto-triggered workflow-lint run `33020964714` was cancelled immediately.
- No local pytest, lint, render, or build command was run.
- No PR, merge, issue closure, runner/service/routing/secret change, or protected
  branch mutation was performed.

