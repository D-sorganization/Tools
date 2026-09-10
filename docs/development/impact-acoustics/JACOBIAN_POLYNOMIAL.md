# Bounded SE(3) Jacobian evaluation

This numerical continuation belongs to Tools #5160 / PR #5162 under #5073.
Published source `279926e95e8a17cc42702a53c30ea03bf4638bec` passes the recorded
Windows regressions but fails the unchanged 60-second first-contact 240-step
test in both hosted shared shards of run 34471450139. A queued Rust job does
not change those completed failures. This document describes an evaluation
of the same matrix function; it introduces no constitutive or impact law.

## Algebra and domain

Use the existing linear-first twist q=(v,w), A=-ad(q), and E=-ad(dq). Define
t=w.w and theta=sqrt(t). For this SE(3) adjoint matrix the minimal polynomial
divides x(x²+t)². The two zero eigenvalues are semisimple when theta is
nonzero: the coupling along the rotation axis vanishes because w.[v]x.w=0.
The identity also extends to pure translation, for which A²=0. Thus the
entire function phi1(A)=sum A^k/(k+1)! has the representation

```text
J = I + b1(t) A + b2(t) A² + b3(t) A³ + b4(t) A⁴.
```

For theta>0, with s=sin(theta)/theta, its scalar coefficients are

```text
b1 = [2(1-cos(theta)) - theta sin(theta)/2] / t
b2 = [2 + cos(theta)/2 - 5s/2] / t
b3 = [1-cos(theta) - theta sin(theta)/2] / t²
b4 = [1 + cos(theta)/2 - 3s/2] / t².
```

These expressions satisfy the Hermite conditions at x=0 and x=±i theta:
the polynomial agrees with phi1 at all three eigenvalues and with its
derivative at the two potentially repeated nonzero roots. Directly computing
these quotients near zero would subtract nearly equal numbers. The code
instead evaluates their analytic Taylor coefficients in t:

```text
[t^j] b1 = (-1)^j (1-j)/(2j+2)!
[t^j] b2 = (-1)^j (1-j)/(2j+3)!
[t^j] b3 = (-1)^j (1+j)/(2j+4)!
[t^j] b4 = (-1)^j (1+j)/(2j+5)!.
```

The zero limits are 1/2, 1/6, 1/24 and 1/120. There is no small-angle
dead zone, finite difference, fitted coefficient, or artificial symmetry.
The coefficient polynomial and its analytic derivative are evaluated together
through degree 18 in t. Since dt=2 w.dw, the matrix variation is

```text
DJ = sum_i [bi'(t) dt A^i + bi(t) D(A^i)]
D(A) = E
D(A^(i+1)) = D(A^i) A + A^i E.
```

The variation direction must itself be an SE(3) adjoint direction. This
formula is not offered as a general matrix-function derivative for arbitrary
6-by-6 perturbations. Inputs remain finite real six-vectors, and outputs
remain independently owned. Reuse of the full section pair within one
inertia evaluation remains as qualified in KINEMATICS_REUSE.md.

The bounded path requires ||A||\_infinity <= 4 and theta <= pi. These are
algorithm qualification limits in the declared SI coordinate representation,
not material limits or objective physical quantities. The existing principal
rotation chart still applies at the public section boundary. Outside this
numerical domain, the original augmented exponential / Frechet routine is
used. SciPy documents its default as scaling, Pade and squaring and gives the
augmented block exponential identity used by the independent test oracle.
[Official SciPy documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.linalg.expm_frechet.html).

## Truncation and roundoff are distinct

For j>=19 and denominator offset d in {2,3,4,5}, every omitted scalar
coefficient has magnitude at most (j+1)/(2j+d)!. Since t<=pi²<10, a bound
for its evaluated term is u_j=(j+1)10^j/(2j+d)!. Consecutive ratios obey

```text
u_(j+1)/u_j = 10(j+2)/[(j+1)(2j+d+1)(2j+d+2)].
```

These ratios decrease with j, so each remaining tail is bounded by its first
term divided by one minus the first ratio. For the derivative tail, use
j(j+1)10^(j-1)/(2j+d)! and the corresponding ratio
10(j+2)/[j(2j+d+1)(2j+d+2)]. Exact rational arithmetic gives the following
conservative whole-matrix truncation bounds:

- ||J - J_degree18||\_infinity < 1.10e-27.
- ||DJ - DJ_degree18||\_infinity < 1.49e-26 ||E||\_infinity.

The second bound uses ||D(A^i)||<=i 4^(i-1)||E|| and
|dt|<=2 pi ||E||<7||E||. It covers the coefficient truncation and its
directional derivative in real arithmetic. It is not a floating-point
roundoff certificate. Matrix products, coefficient storage, Horner evaluation,
and cancellation remain subject to roundoff; tests compare against a separate
24-by-24 augmented exponential with tolerances 3e-14 relative and 2e-14
absolute. Directions scaled by 1e-120 and 1e120 are compared after removing
their scale so that an absolute tolerance cannot conceal a tiny derivative.

## TDD and rejected approaches

The polynomial-specific controls first failed 31 times because the bounded
evaluator was absent. The implementation then passed those controls plus the
existing work-map derivative and within-evaluation reuse controls, 58 total.
Cases include zero and tiny rotations, near-pi rotations, arbitrary axes,
noncommuting directions, exact zero limits, input/output ownership, and both
norm and angle fallback. A first complete coverage run still timed out in the
compression/release test while reaching the separate public derivative path;
its terminal log is retained and no full-suite pass is claimed for that source.
Two additional RED controls then required the public derivative to share the
bounded evaluator and return the exact zero-direction identity after checking
both inputs. Three invalid-input controls already passed. This implementation
passes 63 focused checks and NumPy-aware typing. The original zero-gap release
test passes in 51.07 seconds with coverage and the unchanged 60-second limit.
Existing reuse spies now observe the Jacobian-pair
boundary, preserving the required N+1 pair count without demanding a specific
library implementation. Separate tests still require the original general
routine outside the bounded domain.

Two earlier generic Taylor-series candidates were rejected as performance
repairs. Both passed the first-contact physical checks, but their coverage
runs took 115.04 and 137.16 seconds respectively. Python iteration and bound
evaluation overhead made the adaptive variant especially unsuitable under
instrumentation. Their source, RED/green records and timing receipts are
retained as rejected experiments; neither is production code.

The final fixed-degree kernel benchmark alternates baseline/candidate order
across five repetitions of 500 pairs at six rotations. The qualified source
uses 58-67% of baseline median pair time, with differences of a few units of
floating-point roundoff. This diagnostic benchmark is not a hosted runtime
guarantee, and overall trajectory speed does not scale directly with this
kernel speedup.

The complete directory-path coverage regression passes 1,517 tests with two
optional build123d CAD/export collection skips in 698.82 seconds. Coverage is
93.53%, above the unchanged 20% floor. The 240-step entry case takes 54.473
seconds, and the zero-gap release case takes 48.987 seconds, under the original
60-second limit. All original physical and numerical acceptance bounds pass.
The entry reference's recorded output changes are below 1.96e-12 in its declared
SI coordinate vector; separate work defects and losses retain their own units
in the comparison artifact. These are roundoff-sensitive numerical comparisons,
not measured accuracy or universal timing guarantees.

The completed full run precedes a structural-only direct-import cleanup:
NumPy's same polyder/polyval callables replace two deep namespace accesses,
and docstrings describe the final algorithm. All 63 affected controls pass
again after this cleanup. Scoped Ruff, formatting and NumPy-aware mypy pass;
all four touched Python files stay below 400 lines, all functions below 50
lines with at most four parameters, and attribute chains at most two levels.
JACOBIAN_POLYNOMIAL_RESULTS.json distinguishes the full-run source hashes from
the final source, binds the post-cleanup receipt, and retains rejected/failed
experiments and named motion/work comparisons. Fresh hosted qualification is
still required for the published revision.

No time grid, nonlinear budget, tolerance, material property, quadrature order,
coverage floor or deadline changes. Event-resolved work, useful relative
cutoff-work accuracy, general reversal/recontact, physical identification,
acoustic radiation and perceptual validation remain separate open work.
