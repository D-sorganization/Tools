# Section kinematics reuse and qualification

Review #5160 / PR #5162 continues parent #5073. Hosted Standard CI run
34464371054 at source 72b2efd47 failed both shared Python shards at the
unchanged 60-second limit in the zero-gap frictional compression/release
refinement test. The aggregate run still reported queued because another job
had not started. Job-level results, not that aggregate, establish the failure.

## Scope and exact identities

The timeout stack repeatedly evaluates section inertia inside nonlinear
finite-difference Jacobians. Every interior quadrature fraction previously
computed both the full-section and fractional Frechet exponential pairs.
The private `_SectionVelocityKinematics` now owns copied, read-only section
state and direction, and lazily prepares the full pair once per evaluation.
For N interior fractions this requires N+1 pairs instead of 2N. Endpoint-only
kinematics still uses the exact identity without an exponential. A subsequent
section evaluation creates a new object; no cross-state cache is introduced.
Returned arrays remain independently owned. Matrix-product association is
preserved, including storage of the full Jacobian derivative itself.

For A=ad(relative), write Jr=phi1(-A) and Jl=phi1(A), where
phi1(A)=(exp(A)-I)/A denotes its entire power-series extension. Then
phi1(-A)=exp(-A)phi1(A), and

```text
Jr^-1 - Jl^-1 = phi1(A)^-1 (exp(A)-I) = A.
```

The identity extends to singular A by analytic continuation wherever the
Jacobians are invertible. It does not require division by A. The existing
principal rotation chart remains the domain contract. `_relative_maps` thus
evaluates Jr once, solves for its inverse, and obtains Jl^-1 by subtracting A.
No small-angle cutoff or polynomial approximation is added. This equivalent
algebra can change floating-point roundoff and therefore requires the same
physical and numerical regressions as the previous calculation.

No quadrature order, material property, inertia weight, governing equation,
solver budget, residual bound, time grid, coverage floor or test deadline is
changed. This work does not resolve contact-event or physical/acoustic accuracy.

## TDD and current evidence

The first three new controls failed before implementation: the old inertia
path used 2N exponential pairs, and the owned kernel did not exist. The first
implementation passed 50 focused inertia, kinematics and public-API checks.
The inverse-map call-count control then failed because the old path evaluated
two Jacobians; four direct-exponential comparisons already passed. After the
identity change, all 55 focused checks passed in 6.41 seconds. Direct comparisons
cover zero rotation, 1e-10, 0.4 and 2.6 radians with the original tight bounds.
Input/output ownership and absence of cross-evaluation caching are explicit
controls. Both production files and the new test pass scoped Ruff, formatting
and NumPy-aware mypy.

A plain local baseline of the original zero-gap three-grid test passed in
28.54 seconds; full-pair reuse alone passed in 27.01 seconds. Serialized motion,
impulse, energy-defect and algorithmic-loss outputs were exactly identical.
These individual timings are diagnostic observations, not a statistically
qualified performance claim or a guarantee for hosted Linux. An attempted
module-name coverage baseline failed during collection with a NumPy duplicate
module-load ImportError and supplies no numerical evidence.

The full golf-club/impact regression with directory-path coverage passes
1,481 tests, with two optional build123d CAD/export collection skips, in
743.31 seconds. Coverage is93.51%, above the unchanged20% floor; the original
60-second thread deadline remains active. The zero-gap three-grid test takes
50.354 seconds, and the slowest entry240 test takes57.363 seconds. This narrow
local margin does not guarantee hosted success. Relative to the pre-change
plain zero-gap result, maximum recorded output difference is4.44089e-16;
scaled differences and mechanical defects are identical, and maximum
algorithmic-loss difference is6.77626e-21 J.

KINEMATICS_REUSE_RESULTS.json binds the exact source hashes, actual command,
JUnit, coverage, RED records, skip reasons and timings. The preceding
entry-reference commit36ad1cc528aeb02a989b829bd08f99fbda834235 retains its separate
282-test receipt. Both production files and the new test pass NumPy-aware
mypy and scoped Ruff/format. Fresh hosted CI remains required before claiming
the original timeout repaired. CONTACT_FORCE_REGULARITY.md records the next
contact-law/acoustic comparison requirements and explicit source-access limits.
