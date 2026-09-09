# Conditional Continuous Frequency-Band Coverage

## Scope and integration

Tools #5072 composes the existing `_shaft_frequency_interval` assessor over a
declared closed band. It uses the same length-scaled M/G/C/K coefficients,
separate gyroscopic and passive damping terms, nonsymmetric stiffness, computed
inverse residual and assumed uniform additive dynamic-stiffness error. There is
no new physical shaft or damping law. The API remains private with no exports.

The need is scientific: finite endpoint or grid samples can miss a resonance.
Successful sampled frequency responses therefore cannot establish a whole-band
bound. The new result is returned only when accepted cells cover the entire
requested band; a successful prefix cannot qualify an unfinished band.

## Conditional bound and complete coverage

For each symmetric cell with center w0 and radius h, the existing assessor
evaluates the Neumann contraction q from the computed inverse X, its center
defect, the first and second frequency coefficients, and declared pencil error.
If q is below the supplied threshold, itself strictly below one, it yields
R = ||X||F/(1-q) and delta = q R. In exact arithmetic these bound respectively
the spectral norm of D(w)^-1 and its difference from X throughout the cell.
See FREQUENCY_INTERVALS.md for the full derivation and units. This reuses the
Neumann-series norm argument described in [Cornell's numerical-analysis
notes](https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-01-30.html).

The band routine bisects only cells refused because their contraction exceeds
the declared threshold. It processes the left child before the right, and each
child pair shares exactly the same representable endpoint. A stack retains all
unprocessed cells. Every attempted assessment consumes one unit of the explicit
evaluation budget, including rejected parents. Exhaustion, an unrepresentable
interior split, an invalid plant, a singular solve or other numerical failure
raises without returning a partial cover. No regularization, added damping or
quiet budget increase is permitted. Exact singular centers are valid reasons
to refuse the band, even when adjacent samples are finite.

Symmetric interval construction requires care beyond simply computing
(a+b)/2 and (b-a)/2 in binary floating arithmetic. Endpoints a and b are treated
as exact binary rationals. The center c is the least representable float no
less than their exact midpoint. The radius is rounded upward from
max(c-a, b-c). Therefore [a,b] is contained in [c-h,c+h], including adjacent
floats and subnormal endpoints. Since c is at or above the midpoint and a is
nonnegative, h does not exceed c. Nonfinite resulting intervals are refused.
This bookkeeping prevents gaps; it does not certify floating matrix arithmetic.

The band result asserts bounds on each cell's nominal endpoints only. Rounded
enclosures can extend slightly beyond the requested band, but do not extend the
domain of an assumed coefficient-error bound or a physical model. The same
Neumann inequality holds on the nominal cell using its enclosing radius and
the error bound there; no outside-band error assumption is needed for that
restricted conclusion. The nested symmetric assessment must not be detached
and used to claim qualification outside its enclosing band cell.

For a returned cover, the maximum of the individual R values bounds the whole
band conditionally. For an input b and observation l in the same work-conjugate
coordinates, the response disk on each cell has center l\*Xb and radius
||l||2 ||b||2 delta. Magnitude is bounded above by the center magnitude plus the
radius. A phase bound requires the disk to exclude zero. At or near an
antiresonance, absolute response error remains meaningful while relative error
and phase may be undefined or uninformative. This module does not silently
replace those quantities with a favorable finite value.

## Evidence limits and next requirements

Results explicitly retain `conditional-numerical` evidence and `unqualified`
stability. Fraction arithmetic encloses endpoints only: matrix assembly,
inversion, norms and the final bound are ordinary floating calculations. The
declared coefficient-error bound is an assumption, not identified uncertainty.
The method does not certify exact-real arithmetic, modal truncation, mesh error,
nonlinear time-varying motion, physical shaft properties or acoustic bandwidth.

Continuous full/reduced port-error qualification must retain both models and
separate absolute error from near-zero relative/phase quantities. Rotating joint
mesh/mode convergence, moving-boundary work, flexible contact, calibrated
radiation, consumer integration and physical/blinded studies remain required.

## Verification record

TDD started with a collection failure because the band module did not exist.
The predeclared controls include an analytic damped oscillator and its interior
peak, a hidden undamped pole, uncertainty that subdivision cannot remove,
budget exhaustion after a successful cell, exact binary endpoint enclosure,
strict scalar/shape/integer contracts, unstable plants retaining unqualified
stability, an independent 2x2 adjugate with gyroscopic/circulatory coupling, and
the existing finite-grip spatial shaft. The 59 combined Windows band/interval
tests pass in 8.63 s. Actual mypy passes all three changed Python files, and
scoped Ruff passes. The narrow shaft interval needs only one accepted cell;
an incorrect test expectation of multiple cells was replaced with direct
coverage/adjacency checks, without changing the requested band or numerical
limits. Scalar and coupled controls separately verify actual subdivision.
The full Linux golf/signal/API run passes 956 tests with two optional CAD
skips and three absent-plugin configuration warnings in 93.83 s. Actual
scientific archive provenance is recorded in PROGRESS.md. Repository Ruff
passes 3,805 formatted files; all nine manual gates pass. Only one private
empty-export API entry is added, and canonical inventory keeps provisional
calculation/publication status. Commit 842c39890755a8fcfb419b3068bf0dc58b2e97f1
is published through all normal hooks and remote verified. The full T3 and
downstream program remain open; no physical bandwidth is inferred.

## Main integration follow-up

Main 0f2dfe3fd48381bc608baf208b54c7b0d9396465 adds mocap reconstruction,
temporal mapping and C3D work owned by the separate camera task. The merge
preserves every shaft source/test and the golf API baseline. All 145
Windows mocap/authority/API/band tests pass in 17.03 s; actual mypy passes
eight incoming source files, and root Ruff passes 3,815 formatted files.
Both root-handoff entries are retained and canonical inventory is
regenerated. The original 956-test Linux scientific result remains tied
to its recorded archive. All nine final governance gates and actual mypy for
the three incoming test files also pass. Normal merge publication remains.
