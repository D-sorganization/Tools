# Coupled tangential friction trajectory — review #5160

This continuation of #5073 joins existing shaft/ball mechanics and the private
elastic/Coulomb history law. It is a numerical model with synthetic coefficients.
It does not establish a player's impact efficiency, ball deformation, acoustic
radiation or perceived sweetness. The parent scientific requirements stay open.

## Existing implementation and integration

`_normal_shaft_contact` already combines full shaft/head inertia and rigid-ball
response at a common contact point. `_normal_contact_trajectory` and its event
companion integrate normal contact with explicit grip and applied-load histories.
`_tangential_contact_work` already supplies objective elastic history, a Coulomb
return map and a discrete work identity. It previously supplied no coupled
trajectory. None of these laws or the mass solve is copied into a consumer.

The new implementation shares body response and prescribed-history resolution
with normal contact. It adds separate transport, contracts, response, nonlinear
step and trajectory modules. Histories are immutable during solver evaluations.
Only a converged endpoint becomes the next accepted mechanical/history state.
No head inertia is added again, and no empirical gear-effect correction is used.

## Coupled endpoint equations

For step h, each material pose H and body twist V satisfy the first-order scheme

```text
H_new = H_old Exp(h V_new)
V_new - V_old = h a(H_new, V_new, z_new, t_new).
```

The right side uses the full canonical shaft mass/convective/grip response and
the ball's Newton–Euler response. Ball and face forces act at the same projected
contact point, so the compression-dependent ball lever arm is retained. Force
power uses material-point velocity, not the velocity of a migrating geometric
intersection. Normal force follows the existing explicit unilateral spring/
dashpot/cutoff convention and its force ceiling/domain checks.

Let n be the endpoint unit normal and v_rel the endpoint ball-minus-face material
velocity. The slip increment is h n cross (v_rel cross n). This is the tangential
projection; its cross-product form avoids subtractive cancellation for almost
normal motion. The original strict constitutive tangency check remains active.

With stiffness k_t, previous elastic history z_old, declared rotation Q, and
endpoint normal force F_n, the existing return map gives

```text
z_trial = Q z_old + h v_t
z_new = projection onto |z| <= mu F_n / k_t
tau = k_t z_new
F_ball = F_n n - tau.
```

The same endpoint unknowns determine geometry, normal bound, slip, history and
both accelerations. Trial evaluations always start from the accepted old history;
they cannot accumulate fictitious slip. Initial law, observer, normal and force
cap must agree; an invalid initial history is refused instead of erasing storage.

## Contact-frame convention

Face transport is Q_face = R_face,new R_face,old^T. It maps the normal exactly.
The alternative mean-normal-spin convention adds a rotation about n_new by
h n_new dot (omega_ball,new - omega_face,new)/2, applied after Q_face.
This is a declared first-order discretization of a different constitutive spin.
Neither choice is physical calibration, and neither resolves arbitrary
within-step spin history exactly.

Both maps transform covariantly under a common constant observer rotation.
Independent tests verify normal mapping, orthogonality, pure twirl and the
distinction between conventions. Relative spin can change traction direction
even for an isotropic spring with unchanged stored energy. The
[3D contact-update study by Kuhn, Suzuki and Daouadji](https://arxiv.org/abs/2002.10231)
motivates retaining rotation and twirl explicitly. Its granular contact model
is not a golf-ball friction calibration or a proof of this entire time integrator.

## Nonlinear solve and limits

Unknown body velocities are scaled by the existing mechanical length/time
scales: L/T for translations and 1/T for angular components. The solver uses
[SciPy's MINPACK hybrid root method](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html).
A forward-difference Jacobian uses sqrt(machine epsilon) times max(1, |x_j|)
in these dimensionless coordinates. The unit floor prevents vanishing
perturbations when a material velocity component is nearly zero. Repeated
Jacobian requests at exactly the same point reuse an owned copy within one
step; each actual response evaluation still consumes the caller's budget.

The relative-iterate stopping criterion is sqrt(machine epsilon), following
[MINPACK's recommendation](https://www.math.utah.edu/software/minpack/minpack/hybrd.html).
It is distinct from the caller's mechanical equation-residual tolerance.
Acceptance requires both a successful solver exit and a fresh independent
endpoint evaluation with max(abs(scaled mechanical residual)) <= that tolerance.
The test tolerance remains 1e-10; no test oracle, deadline, response budget or
force/geometry domain was loosened. The original 0.1-times-residual iterate
criterion could stall at residuals below 3e-17. Solver failure, an excessive
fresh residual, exhausted budgets and invalid arithmetic still refuse the
whole requested trajectory without returning a partial result.

Backward Euler introduces mechanical numerical damping. It is first order,
including between contact events. A small nonlinear residual does not establish
small time-discretization error, a unique nonsmooth solution, correct peak force
or correct event timing. The fixed grid does not resolve all within-step contact
or stick/slip transitions. Positive-gap onset is checked on a synthetic three-grid compression/release
control. Grazing/repeated events, arbitrary within-step mode transitions and
general mesh/mode convergence remain separate unresolved qualifications. The existing higher-order normal-only
solver remains available and is not replaced by this first-order method.

## Work and dimensions

Lengths/slip are in metres, forces in newtons, body angular velocity in radians
per second, stiffness in N/m, impulses in N s and energies in joules. The return
map retains its exact discrete constitutive identity

```text
tau dot ds = Delta(k_t |z|^2/2) + tau dot dp
             + k_t |z_new - Q z_old|^2/2.
```

The plastic term is nonnegative. The final term is a separately reported
tangential algorithmic loss, including storage removed by a collapsing normal
bound. It must converge away and must never be labelled sound or measured heat.
Normal elastic storage, viscous loss and cutoff storage removal remain separate.

The trajectory integrates external input, anchor output, grip dissipation and
normal loss powers with endpoint quadrature. Its uncorrected defect is final
mechanical plus contact energy minus initial energy and external work, plus all
reported output/loss channels. Adding the tangential algorithmic term leaves a
mechanical integration defect; it does not make the trajectory energy-exact.
Normal and vector tangential impulses are retained alongside ball spin and the
complete shaft response for later matched-intervention studies.

## Current evidence and outstanding qualification

Two missing-module RED failures preceded implementation. The independent
zero-friction three-mass test exposed subtractive slip-projection cancellation;
the cross-product form corrected it without weakening tangency. A rotated
observer case then exposed a real residual of 4.52e-6. Solving predictor
corrections removed that failure but caused false-negative solver exits in other
cases, so it is not the delivered parameterization. Physically scaled Jacobian
increments fix the original problem. Caching duplicate Jacobian requests keeps
the full release study within the original test deadline; the initial uncached
version timed out and its diagnostic is retained.

At the cached-Jacobian intermediate source, 239 affected tests pass in 51.22 s.
Positive-gap onset/release passes separately in 30.65 s, including independent
constant-velocity/zero-impulse ball motion before contact. New integrated elastic
reversal and nonplanar momentum refinements first exposed stagnation at residuals
1.17e-18 and 2.29e-17. Separating the documented MINPACK iterate criterion from
the unchanged mechanical-residual check yields ten passing mode, momentum and
invariance tests. Exact final source hashes and results belong to
`FRICTION_RESULTS.json`; intermediate records retain their actual identities.

Independent controls cover: linear three-mass endpoint equations and a matrix
exponential reference; full material ball Newton/Euler equations including
transport and off-center torque; whole free-system spatial momentum from
canonical mass and independent world moment arms; common observer transforms;
normal rotation and relative twirl; elastic reversal, monotone Coulomb sliding,
shrinking normal-cap release, input immutability and solver refusal.

The momentum check shares the canonical mass interpolation. It verifies the
coupling and spatial moment balance, not physical quadrature accuracy. Its
finite-step momentum defect is reported and must decrease with refinement;
backward Euler is not asserted to conserve momentum exactly.

The smooth reference uses 8/16/32 steps and the original preset first-order
ratio range. Earlier 4/8/16 steps lay outside that asymptotic range; the grid
was refined rather than changing the order criterion. Compression/release
controls use 30/60/120 steps, synthetic mu=0.3, independently retained normal
and tangential impulses, ball speed/spin, mechanical defect and tangential
algorithmic loss. First touch and an initial 0.1 mm gap are separate cases.
They do not establish event-exact peak force or an error bound for arbitrary
club trajectories.

Final Windows verification passes 253 tests in 97.32 s with no skips or
failures. Both seven-file typing modes pass; all prior API records are unchanged.
Independent normal-force errors decrease from 0.1460 to 0.07337 to 0.03677 N;
five-channel work errors decrease from 6.80e-6 to 3.47e-6 to 1.75e-6 J.
The free spatial momentum defect decreases from 4.82e-7 to 2.39e-7 to 1.19e-7
in the declared unit momentum scales.

The positive-gap combined output norm decreases, but its final y-spin values
1.423818, 1.418119 and 1.441712 rad/s are not yet componentwise asymptotic.
A norm dominated by normal impulse does not establish spin convergence.
Keep review #5160 open for finer contact-transition and componentwise spin
verification; do not close it merely because the aggregate test passes.
Source/API/inventory/turnover and platform/review evidence remain required. Parent requirements include face/hosel/ball material behavior, measured
friction, mesh/mode and grazing/repeated-event qualification, radiation,
held-out force/spin recordings and blinded sweetness. No numerical loss is
promoted to physical heat, acoustic energy or perceived quality.

## Finer-grid diagnostic and exact one-step failure

The same published production source completes 240 steps from the positive gap:
y-spin 1.440045294 rad/s, normal impulse 0.004537435 N s, mechanical defect
-0.000110616 J and tangential algorithmic loss 0.00000451942 J. These are
additional diagnostics, not a componentwise asymptotic certificate.

The 480-step run refuses the interval [0.0014625, 0.00146875] s. Its previous
gap is -0.000145563 m and normal force 2.94956 N, so this occurrence is inside
compression, not contact onset. MINPACK reports stagnation with a residual of
2.602085214e-18 even at its recommended iterate tolerance. The full requested
trajectory is not returned. The exact accepted mechanical/tangential state is
embedded in FRICTION_RESULTS.json; reconstructing that one step reproduces the
refusal in 0.344 s, avoiding a full trajectory in the next RED regression.

Review #5160 and draft PR#5162 remain open. Next, add that exact regression and
repair the distinction between mathematical residual convergence and backend
termination reporting. Preserve actual evaluation/domain limits, independent
fresh residual and force/work/momentum controls; report termination evidence
explicitly. Do not adjust tolerances merely to make this fixture pass. Finer
individual spin convergence remains a separate requirement after that repair.
