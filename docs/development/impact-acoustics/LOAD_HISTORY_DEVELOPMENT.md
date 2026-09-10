# Prescribed Force and Couple Histories

Parent: Tools#5073/#5068; normal-event foundation: PR#5152. This private
continuation permits explicit additional loads on existing shaft nodes during
motion and contact. It does not infer a hand controller or material properties.

## Existing Integration and Ownership

`SpatialPointLoad` already owns observer-resolved force and free couple, a
material point offset, its body wrench and instantaneous power. Shaft dynamics
already assemble these loads with distributed/head inertia and finite moving
grips. `MovingTrajectoryProblem` previously varied only prescribed anchors;
the normal-contact trajectory reused it and retained fixed external loads.

`PrescribedPointLoads` now adds a named inertial observer, declared closed SI
time interval and explicit callback returning a complete additional-load tuple.
Each evaluation appends that tuple to the original baseline exactly once.
The common append helper is also used for internal face reactions. It adds no
head inertia and does not accumulate prior stage or contact loads. Both fixed
Lie RK4 and adaptive normal-contact paths reuse this composition.

## Equations and Interpretation

For node pose `(R,p)`, material offset `r`, body velocity `(v,omega)`, spatial
force `f(t)` and spatial free couple `tau(t)`, the existing canonical load gives

    Q = (R^T f, r cross R^T f + R^T tau)
    P = f dot R(v + omega cross r) + tau dot R omega = Q dot V.

The couple is additional to the force moment. The shaft/ball mechanical ledger
uses this instantaneous input power, retaining anchor output and grip/contact
losses separately. No force potential is inserted into mechanical storage, and
no `df/dt dot position` term is required in this external-work convention.
Time-varying material or grip coefficients would require additional storage
terms and are outside this API. Offsets identify supplied material points;
their variation is not a measured sliding or migrating hand-contact model.

At identical state and constitutive law, an added force/couple changes the
generalized acceleration through the existing mass solve. Over a trajectory it
can change the contact state and subsequent impact outcome. This establishes a
mechanism within the assumed model, not a magnitude for any player's swing.
Matched-state studies must retain external and anchor work, normal contact
history and event-refinement evidence. Comparisons that silently change these
inputs cannot isolate geometric stiffness, gyroscopic effects or hand damping.

## Verification and Limits

The missing module is retained as the initial RED. Eighteen initial controls
pass, including an independent polynomially forced three-mass matrix exponential
and work quadrature, preset fourth-order refinements, adaptive agreement,
nonplanar offset-force/couple power, ownership and missing-data refusal. Two
additional controls reject untyped histories at both problem boundaries.

The expanded Windows run passed the20 new controls and12 existing temporal
controls, then reached the unchanged60s thread timeout inside the old event
refinement test; no JUnit was completed. Its stack was in SciPy's matrix
exponential Frechet evaluation. Cause is not established and no deadline or
tolerance is relaxed. Source976d7ff43 is archived for separate Linux and Windows
qualification. Four changed production modules pass NumPy-aware and isolated
pre-push mypy. The API RED adds only one empty-export private module; ordinary
regeneration passes the two package controls.

The exact archive976d7ff43 passes406 Linux controls in270.59s with58.86%
coverage against the unchanged20% requirement. All101 existing golf-club and236
existing swing API records are identical; the added module exports no public
symbols. The independent synthetic example uses additional axial force
`0.4 + 500 t + 100000 t^2` N with time in seconds. At0.4ms, the predicted ball
velocity changes from-0.2151976813 to-0.2151688602m/s at the same initial state.
The fine RK4 state error is2.43e-10 and its energy defect is-1.48e-13J. These are
numerical reference results for artificial parameters, not a golf prediction.
The total external work is negative (-7.65e-5J) because the loaded point moves
against the positive force; additional force does not imply added energy.

Callbacks must be deterministic, consistent and sufficiently resolved. Time
coverage is a declaration, not proof of measured data completeness. Invalid
time, observer, node, load type or callback failure is refused; no interpolation
or partial successful trajectory is substituted. Baseline loads remain, so a
caller must avoid double-counting forces already represented by a finite grip.
No feedback, friction-history trajectory, deformable face/hosel, measured
force/spin, acoustic radiation or perceptual qualification is supplied here.

The same numerical source passes406 Windows controls in169.77s, with no failures or skips. Both platforms give identical independent-study values. The state norm uses numerical SI components (positions in metres, velocities in m/s); it is a test monitor, not a measurement uncertainty. The event CI annotation repair839ebe083 is incorporated separately without changing numerical expressions; its affected integration check is recorded separately.

After adopting the two event ndarray annotations from839ebe083, all35 affected load-history/event controls pass in42.90s. This later receipt is separate from the406-test archive; no numerical expression changed.
