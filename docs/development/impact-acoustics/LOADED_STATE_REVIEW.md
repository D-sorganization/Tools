# Loaded-State Formulation Review

Tools #5072 is still in progress. This note records the next implementation
decision, verified private kinematics, elastic energy derivatives and source access;
no nonlinear equilibrium solver is implemented. The existing
rotating transport and radial references remain independently useful checks.

## Objective Section Kinematics

[Sonneville, Cardona and Brüls (2014), author manuscript](https://orbi.uliege.be/bitstream/2268/159471/1/pa_SonnevilleCardonaBruls2014_GeometricallyExactBeamFEOnSE3.pdf)
was inspected in sections 3–7, particularly equations 35, 55–56, 63–66 and 73.
Their two-node SE(3) interpolation couples positions and orientations through
the relative transform. It represents constant material strain and pure bending
without introducing the shear locking of independently linear position fields.
The strain energy uses a linear constitutive law in small section strains;
geometric exactness does not remove this material restriction. The relative
rotation logarithm also needs an explicit domain below its pi branch boundary.
Section 7.1 retains only the material tangent in its examples; the omitted
geometric and external-load derivatives must be derived for this prestress work.
Their material-frame tangent and velocity coordinates must not be substituted
unchanged for the current canonical small-rotation operators.

The first private kinematics are now implemented in `_shaft_se3.py` and checked
by `test_shaft_se3.py`. TDD first failed because the module did not exist;
26 isolated tests then passed (7.44 s). The initial broad run found the new
test's use of the alternate top-level package alias caused duplicate turf
registration; the test now follows the suite's `shared.python.golf_club` imports.

Twists are linear-first, with translation generator in metres and rotation in
radians. Poses map section material axes into a common observer frame. For
angular generator A, the upper-right block of `expm([[A, I], [0, 0]])` gives
`J = integral_0^1 exp(t A) dt`. Thus `Exp([v,w]) = [Exp(A), J v; 0, 1]`;
the inverse obtains w from SciPy's principal SO(3) logarithm and solves J v = p.
This uses existing SciPy numerical routines rather than importing or copying
the leaf rotation-converter algorithms. There is no small-angle dead zone.
The kernel uses existing strict numeric and proper-rotation validators.

The tests compare screw motion with an independent 4x4 matrix exponential,
preserve 1e-12-radian rotations, and recover exact circular pure bending at
four material positions after an arbitrary common rigid motion. The resulting
section twist per length is `[0,0,1,0,curvature,0]`, with no spurious axial or
shear strain. Non-rigid transforms, invalid fractions and coerced numeric
values are refused. Rotations at or within 1e-6 rad of pi are explicitly outside
the local chart; this numerical margin does not establish a material strain limit.

The broader golf suite passes 424 tests with two optional build123d skips; all
nine API checks pass after recording the new module with an explicitly empty
export tuple. No existing API signature changed. Repository Ruff 0.14.10 passes,
and changed-module mypy with `--follow-imports=silent` passes. All nine manual
governance checks pass after staging the new module and regenerating inventory.
Validation used `python -m pytest tests/shared/python/golf_club tests/test_shared_package_api_stability.py -q -n0 --no-cov`;
after the additive API baseline update, the nine API checks were rerun separately.
The optional CAD skips are missing build123d, not physics test failures.

These results qualified the initial kinematics checkpoint `52d6ec791`. The
following private elastic element extends that checkpoint; equilibrium, inertia,
applied loads, stability, contact and acoustics remain separate open gates.

## Elastic Energy and Complete Internal Tangent

`_shaft_section.py` now defines a uniform section element with explicit positive
length L, reference twist d0 and symmetric positive-definite coupled stiffness C.
No shear stiffness or anisotropic coupling is inferred from EI, GJ or diameter.
For `d = Log(H_A^-1 H_B)`, strain is `e = (d-d0)/L` and internal energy is
`U = L e^T C e / 2`. The linear components of e are dimensionless; angular
components are inverse metres. The resultant `n = C e` contains force [N] and
moment [N m], so C's translation, coupling and rotation blocks have units N,
N m and N m² respectively. This small-material-strain quadratic law permits
finite rotations; it does not establish a material strain limit or identify a
real shaft. Input arrays are copied into immutable tuples, invalid numerical
types and nonfinite data are refused, and stiffness is never symmetrized or
projected onto a positive cone. Exact input symmetry is required.

The following is this implementation's fixed-chart derivation. For a linear-first
twist `d = [v,w]`, `ad(d) = [[cross(w),cross(v)],[0,cross(w)]]`.
Define `J_r(d) = integral_0^1 exp(-t ad(d)) dt`, `J_l(d) = J_r(-d)` and
`P = [-J_l(d)^-1, J_r(d)^-1]`. At the nominal configuration, an independent
local nodal variation gives `delta d = P delta q`. Therefore the internal
gradient is `g = P^T n`, and its material contribution is `K_m = P^T C P/L`.

For each column direction a, let `h = P a`, `A = J_l^-1`, `B = J_r^-1`.
The relative-coordinate derivative is
`DP[h] = [A DJ_l[h] A, -B DJ_r[h] B]`. A fixed nodal exponential chart is
`H_i(q_i) = H_i(0) Exp(q_i)`, so its coordinate-rate mapping also changes:
`D J_r(0)[a_i] = -ad(a_i)/2`. The complete energy-Hessian column is
`K a = K_m a + DP[h]^T n - blockdiag(ad(a_A),ad(a_B))^T g/2`.
Omitting that final chart term differentiates a moving material force basis,
which is not the requested fixed-chart Hessian. The implementation does not
enforce output symmetry to conceal an omitted derivative.

The Jacobians use the upper block of a matrix exponential; their directional
derivatives use SciPy's existing
[matrix-exponential Fréchet derivative](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm_frechet.html).
This differentiates the defining integral without finite-difference steps or
small-angle cutoffs in production. It adds no dependency and shares the existing
SE(3) block-exponential helper. Scalar-energy finite differences are used only
as an independent test oracle.

TDD first failed collection because the section module did not exist. The initial
22 section tests plus 26 kinematic tests then passed. Expanded verification now
passes 57 tests (13.61 s): all six single-strain energy/length scalings; a coupled
anisotropic loaded fixture's virtual work; all 144 fixed-chart Hessian entries;
observer invariance of energy, gradient and tangent; six unloaded rigid modes;
zero work and scalar energy curvature along six exact loaded rigid-motion paths;
pure bending at 1e-12, 0.2 and 2 rad/m without artificial shear/axial energy;
and strict input and copy-isolation contracts. Virtual-work differences use a
1e-6 coordinate step with rtol 2e-7/atol 2e-9. Energy second differences use
2e-4 with rtol 3e-5/atol 3e-6; analytic symmetry is checked independently at
2e-12 absolute tolerance. These heterogeneous coordinate tolerances describe
the synthetic fixture, not an error guarantee for every physical section.

The loaded fixture's geometric contribution has norm greater than 0.5 in its
declared SI coordinates, ensuring that dropping all prestress terms fails the
test. This magnitude is not a measured golf effect. A loaded free element need
not have six zero eigenvalues of its internal Hessian alone: physical equilibrium
and stability require the applied-load derivatives and boundary constraints.
No negative eigenvalue is clipped, and no internal tangent is advertised as a
complete loaded-system stiffness. Next assemble consistent applied loads,
residuals, reactions, strain-domain checks and equilibrium before dynamic use.

## Energy and Time Integration

[Zupan and Zupan, published online 2018, journal volume 2019](https://link.springer.com/article/10.1007/s11071-018-4634-y)
was read in full-text HTML through sections 2–4 and selected benchmarks in
section 5. It provides a geometrically exact Cosserat formulation with a
symmetric section constitutive matrix and an energy-preserving velocity-based
time discretization. Kinematic compatibility, stress evaluation and rotation
updates must be considered together. Its conservative benchmarks motivate
independent energy and rigid-motion gates; they do not qualify this project's
integrator or a golf club. Numerical dissipation must stay separate from
identified physical damping and measured acoustic decay.

The Invernizzi–Dozio 2016 rotating-beam paper
(DOI 10.1016/j.jsv.2016.01.049) remains abstract/search-excerpt access only;
publisher and institutional full-page retrieval returned 403. Do not claim
its complete derivation or numerical tables have been reviewed.

## Integration Constraints and Required Work

- Keep `ShaftProfile` as the property/provenance authority. Supply missing
  shear and coupled section stiffness explicitly; no isotropic material law
  follows from diameter, EI and GJ. Preserve full head inertia and COM offset.
- The existing shared `rotation_transforms.Rotation` wraps SciPy for SO(3).
  SE(3) exponential/logarithm code exists in the leaf
  `rotation_converter/modern_robotics_pkg/se3.py`, using angular-first vectors.
  Shared golf physics must not acquire a leaf-tool dependency or silently use
  that order in its linear-first ports. Qualify shared reuse before adding
  duplicate transformation algorithms. Existing small-angle branch behavior
  also requires derivative tests before a Newton tangent relies on it.
- Derive static residual, geometric tangent, mass and transport from the same
  nominal shape and kinematics. Include applied-wrench coordinate derivatives;
  a fixed physical torque is not necessarily a constant canonical load.
- Solve and report residuals, reactions, strains and source loads. Reject
  unconverged, unstable, compressive or out-of-domain states under the declared
  model policy. Do not repair a negative physical stiffness by projection.
- Retain the current tensile/Rayleigh reference for independent low-strain
  limits. Add shear/rotary bandwidth, mesh/time/modal and full/reduced FRF
  magnitude-and-phase checks. No universal golf bandwidth is established by
  a synthetic uniform beam.
- Separate inertial work by the moving grip from relative-port dissipation.
  A hand impedance or imposed trajectory must carry its own provenance and
  power-conjugate frames. Contact and ringdown need a complete work ledger.

The private kinematics use existing SciPy; no dependency was added or upgraded.
No nonlinear equilibrium or time-integration backend is selected. All
later contact, acoustic, consumer, experimental and synthesis gates remain
required in PROGRESS.md.

## AffineDrift Theory Consistency Follow-up

A read of AffineDrift main at `4748e674db5c3f7d0c981282f6cf4a8ea9dd86ed`
confirmed that
[`ch29_joint_damping_friction.qmd`](https://github.com/D-sorganization/AffineDrift/blob/4748e674db5c3f7d0c981282f6cf4a8ea9dd86ed/articles/The_Physics_of_Golf/quarto/ch29_joint_damping_friction.qmd)
still categorizes Coriolis terms as implicit damping in two places. This must be
corrected under the theory program before final synthesis. For a conservative
Lagrangian model with a consistent Coriolis representation,
`v^T C(q,v) v = v^T Mdot(q,v) v/2`; this contribution balances the changing
kinetic metric and is not viscous heat loss. Linearization about a moving
trajectory can change perturbation dynamics without identifying dissipation.
In state coordinates `x=[q;v]`, viscous damping enters as
`[0; -M(q)^-1 D(q)v]`, with a separate input/state block and constraint treatment.
The book's unembedded effective-damping expression should be replaced with
that explicit state mapping. No book files were changed by this Tools checkpoint.
