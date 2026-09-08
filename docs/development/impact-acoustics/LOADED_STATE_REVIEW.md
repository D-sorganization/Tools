# Loaded-State Formulation Review

Tools #5072 is still in progress. This note records the next implementation
decision, verified private kinematics and source access; no nonlinear solver is implemented. The existing
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

The implementation remains kinematics only. Next derive section strain energy,
axial/torsional limits, virtual work and the **complete geometric tangent** before
equilibrium solving. A common material-frame tangent is not automatically the
Hessian in a fixed nodal exponential chart: its moving nodal coordinate bases
also contribute derivatives. Preserve this distinction when checking symmetry
and prestressed vibration. No loaded-shape, stability, inertia, contact or
acoustic result follows merely from passing these kinematic tests.

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
