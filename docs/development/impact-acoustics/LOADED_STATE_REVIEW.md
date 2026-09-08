# Loaded-State Formulation Review

Tools #5072 is still in progress. This note records the next implementation
decision and source access, not an implemented nonlinear solver. The existing
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

This is a candidate for the loaded-state and shear extension. Its first TDD
gates should establish rigid-motion objectivity, exact circular pure bending,
axial/torsional energy, and virtual-work derivatives before equilibrium solving.
No test or implementation of this candidate has yet been run in this project.

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

No new dependency or nonlinear backend has been selected or installed. All
later contact, acoustic, consumer, experimental and synthesis gates remain
required in PROGRESS.md.
