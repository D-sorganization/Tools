# Shaft Prestress Implementation and Turnover

Parent #5068; T3 #5072 remains in progress. This checkpoint supplies the
prescribed tensile operator and its beam limits. It does not satisfy the full
distributed rotating shaft/grip milestone. No measured golf or acoustic effect
is established here. The engineering publication remains governed by
`manuals/tools`; this is an implementation and verification record.

## Contracts and Integration

`golf_club.shaft_prestress` exposes `ShaftPrestress`, `radial_shaft_tension`,
`shaft_geometric_stiffness`, and `solve_prestressed_shaft_modes`. It consumes
the existing immutable measured `ShaftProfile` and `ShaftModalSettings`.
The existing midpoint-property bending assembly, consistent mass and symmetric
generalized eigenvalue solver now live in private `_beam_fem.py`; the unloaded
public solver delegates to that same implementation. Its public signature and
reference behavior are retained.

Coordinates alternate transverse displacement in metres and slope in radians.
The raw shaft datum is converted to an exposed coordinate measured from the
trimmed butt. The exposed length excludes insertion. Inserted shaft mass is
not silently added to the tip: callers must supply attached mass explicitly.
The point tip mass has translational inertia only, and cannot represent the
clubhead's full spatial inertia or bending/torsional modes.

Load fields are explicit SI real values; booleans, nonfinite inputs, negative
hub radius/mass/tension and positions outside the exposed span are refused.
Both signs of angular speed yield the same centrifugal tension. Overflowing
tension/matrix results are refused. Modes require finite positive eigenvalues.
Nonzero station spine angles are refused by this uncoupled modal solver;
rotated anisotropic principal axes require a coupled bending operator.

## Independent Derivation

Let s run from zero to exposed length L, h denote hub radius, mu(s) the
piecewise-linear mass per unit length, Omega a constant rotation rate, m_t
the explicitly supplied point tip mass, and P a nonnegative extra dead tension.
Force equilibrium of the straight outboard segment gives

\[
N(s)=P+\Omega^2\left[m_t(h+L)+\int_s^L\mu(r)(h+r)\,dr\right].
\]

For uniform density, the integral is
mu [h(L-s)+(L^2-s^2)/2]. The implementation integrates the product of two
linear functions exactly on each density segment using endpoint values. It
avoids subtracting nearby cubic antiderivatives at the tip. A trimmed/tapered
test independently evaluates the polynomial integral in exposed coordinates.

For small transverse motion w, expansion of the length change gives the
quadratic prestress energy and its finite-element Hessian:

\[
U_g=\frac12\int_0^L N(s)(w'(s))^2\,ds,
\qquad K_g=\int_0^L N(s)B'(s)^T B'(s)\,ds.
\]

With cubic Hermite interpolation, slope products are quartic and N is cubic
within each linear-density segment. Four-point Gauss quadrature therefore
integrates the degree-seven integrand exactly, splitting at every density
knot even when it is inside an element. This is spatial integration accuracy,
not an assertion of exact continuum modes for a finite mesh.

For constant N on an element of length ell, independent integration yields

\[
K_g=\frac{N}{30\ell}
\begin{bmatrix}
36&3\ell&-36&3\ell\\
3\ell&4\ell^2&-3\ell&-\ell^2\\
-36&-3\ell&36&-3\ell\\
3\ell&-\ell^2&-3\ell&4\ell^2
\end{bmatrix}.
\]

Nonnegative N implies nonnegative quadratic energy. Rigid transverse
translation is a null vector. The tests independently check these properties,
the constant matrix, linear-displacement energy, and cubic-displacement
energy across an interior density knot by reversing the integration order.

The modal problem is (K_b + K_g) phi = omega^2 M phi after eliminating the
clamped butt displacement and slope. The point tip mass adds only to the
last translation diagonal of M. Existing EI and mass midpoint interpolation
is retained deliberately, so profile refinement still requires mesh studies.

## Reference Evidence and TDD

- Initial RED: collection fails because `shaft_prestress` does not exist.
- Initial GREEN: 24 new/existing shaft tests pass in 8.21 s.
- Facade RED: the explicit public-export assertion fails; additive exports fix it.
- Scope RED: a rotated-spine profile was accepted; the modal solver now refuses
  this unsupported coupling rather than returning misleading uncoupled modes.
- Final focused/API run: 38 tests pass in 15.79 s using isolated Python 3.12,
  pytest-qt, serial execution and the additive API baseline regeneration.
- Constant-tension energy, unloaded parity, large-tip-mass static-spring limit,
  speed-sign invariance, independent matrix storage, trim/density integration,
  mesh refinement and numeric refusal are separately exercised.

The rotating reference uses the first out-of-plane frequencies in Table 5
of [Rodrigues et al., arXiv:2401.17519v1](https://arxiv.org/html/2401.17519v1).
With L = EI = mass per length = 1 in SI, the spin parameter equals Omega;
the listed dimensionless frequencies are angular frequencies. Rates
0, 3, 6 and 12 give reference values 3.5160, 4.7973, 7.3604 and 13.1702.
The test uses a predeclared absolute tolerance of 0.00006, including the
table's four-decimal rounding, and 24 elements. Third-mode refinement uses
4/8/16/32 elements and the table's 79.6145 reference at rate 12. These are
numerical beam benchmarks, not measurements of a golf shaft. We do not use
the paper's coarse one-element high modes as exact continuum values.

The standard weak-form bending and mass construction is also documented in
the [TU Delft beam dynamics teaching material](https://teachbooks.tudelft.nl/computational-modelling/dynamics/Exercises/str_elem_dyn_workshops/Workshop_FEM_dyn_beam.html).
The implementation reuses this repository's existing matrices; it does not
copy the teaching page's example code or treat that page's execution as a test.

Broader provider validation passed 334 tests and skipped two, with one expected
failure in the exact facade export list. Adding the four new symbols retained
all previous exports. The subsequent focused facade/shaft/API run passed all 62
tests in 14.26 s. Pinned Ruff reports 3,700 files formatted with no lint errors;
three-module mypy and all nine manual/governance checks pass. Implementation
checkpoint: `f1f8da112`. Existing publication approval blockers remain.

## Remaining T3 Requirements

The symmetric tensile Hessian alone is not a full rotating operator.
Two reported axes are alternative prescribed-tension bending directions;
they are not simultaneous predictions of in-plane and out-of-plane rotating
frequencies. The restricted radial out-of-plane limit is the only rotating
interpretation verified here. Centrifugal load depends on the actual
equilibrium geometry; large extension or deflection invalidates this reference.
The existing profile does not supply axial EA or polar rotary mass inertia,
so the model cannot check that domain or infer missing composite properties.

Next work must add explicit axial/torsional constitutive and inertia data,
coupled bending orientation, consistent rotating-base kinematics with Coriolis,
Euler and spin-softening contributions, full clubhead spatial inertia, and a
passive six-axis grip impedance. It must verify frame agreement, rigid and
stationary limits, modal/mesh/time/FRF convergence and work/energy closure.
Time-varying prestress requires its parametric work term; this checkpoint
does not supply a transient solver or imply that a varying K_g is passive.
Grip damping requires measured or explicitly synthetic parameters; do not
convert hand pressure into acoustic damping by analogy alone.

T4 must integrate finite off-center contact/friction and head/shaft modes;
reuse `swing_sim.impact_interval` contracts where applicable. T5 needs measured
transfer/radiation and held-out calibrated acoustics. T6 and UpstreamDrift
must consume qualified versioned provider contracts, followed by registered
studies, empirical/blinded validation and AffineDrift's final synthesis.

## Continuation State

Worktree `Tools-impact-shaft`, branch `feat/5072-prestressed-shaft`, base
`2f975d06ee9e4192a4b35ba5172bb7b3b8950e8a` (T2). The free claim was followed by a
successful codex lease, session `impact-acoustics-01a07d8a-t3`, expiring
2026-09-08T04:51:16Z. The standard claim label was missing and was created;
the earlier failed lease is not the active claim evidence.

UpstreamDrift PR #9745 fixes the provider-test bootstrap and source ownership;
its 13 no-vendor and 72 pinned-vendor checks pass locally, plus an installed
Tools wheel smoke and an eviction-mutation negative control. Protected CI is
still required. Tools #5077/#5082 remain open; #5082 is ready for review.
Its downstream run reproduces the same UpstreamDrift test failure. Gasification
checkout returns Not Found before tests; its credentials/access are unchanged.
The Worker E2E job passed 23 PyQt tests but failed visual baseline comparisons
across seven existing views. Inspect candidate images and existing visual
baseline work before attributing or changing this gate; no tolerance is waived.
