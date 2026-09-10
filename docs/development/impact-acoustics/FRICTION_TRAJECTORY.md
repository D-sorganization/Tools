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
The application also stops a function evaluation when its maximum scaled
residual is at most min(caller tolerance, machine epsilon times
max(1, maximum absolute scaled velocity)). This roundoff-scale stopping
threshold is not a rigorous floating-point backward-error bound. It avoids
requiring further iterate progress after the endpoint equations are already
satisfied. Jacobian difference probes cannot trigger this stop.

An immutable record distinguishes this application stop (`roundoff-residual`)
from a successful backend return (`backend-iterate`). Supplied initial data has
its own `initial` record and makes no solved-equation claim. Every candidate,
from either numerical criterion, requires a fresh budgeted endpoint evaluation
with max(abs(scaled mechanical residual)) <= the caller tolerance. A backend
that actually returns failure is still refused; no backend success flag is
fabricated. Tests reject both misleading backend reports and a deliberately
false residual callback. The test tolerance remains 1e-10; no test oracle,
deadline, response budget or force/geometry domain was loosened. An excessive
fresh residual, exhausted budgets and invalid arithmetic refuse the whole
requested trajectory without returning a partial result.

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

The earlier implementation75f8328bb passes 253 Windows tests in 97.32 s with no skips or
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

## Preserved earlier failure and residual-stop qualification

Implementation75f8328bb completes 240 steps from the positive gap:
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

That captured state first fails the new regression, then passes after the
application residual stop described above. A separate contract-import RED
precedes the termination record. The repaired source passes 258 affected
Windows controls, including the five convergence cases, in 94.25 s. Both
production typing modes pass. Historical source and failure receipts stay in
FRICTION_RESULTS.json; FRICTION_CONVERGENCE_RESULTS.json identifies the repair,
new reference tests and all finer-grid results separately.

An independent continuous sticking reference integrates world history using
zdot = omega_transport cross z + v_t and local Lie-chart pose rates with
[SciPy DOP853](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
For mean-normal-spin transport, omega_transport is omega_face plus half the
normal component of omega_ball minus omega_face. This reference shares the
canonical mechanics, force evaluation and chart differential, but uses neither
the discrete history return map nor the endpoint root solver. Both conventions
remain strictly compressed and below half the Coulomb cap throughout. Raw
history tangency is monitored before removing roundoff normal drift.

The adaptive reference agrees with a tenfold tighter tolerance to 1e-10 in the
declared scaled outputs. Against it, the 4/8/16-step backward-Euler errors are
3.8841e-4/1.9419e-4/9.7093e-5 (face transport) and
3.8614e-4/1.9306e-4/9.6526e-5 (mean spin). Both satisfy the preset 1.7–2.3
first-order ratio bounds with zero plastic loss. These two additional tests
qualify smooth sticking time convergence, not sliding or physical coefficients.

The positive-gap compression/release diagnostic now completes all finer grids:

| Steps | Final y-spin (rad/s) | Normal impulse (N s) | Mechanical defect (J) | Actual evaluations |
| ----- | -------------------- | -------------------- | --------------------- | ------------------ |
| 240   | 1.4400452942         | 0.0045374347         | -1.1061583e-4         | 5938               |
| 480   | 1.4461078036         | 0.0045717032         | -5.5394438e-5         | 11754              |
| 960   | 1.4456943845         | 0.0045892411         | -2.7720634e-5         | 23110              |
| 1920  | 1.4454844460         | 0.0045980250         | -1.3866184e-5         | 46084              |

All accepted steps use the explicitly recorded residual stop, with maximum
fresh scaled defects below 2.22e-16. The 480/960/1920 spin-difference ratio is
about 1.969; normal impulse, normal speed and energy defects also approach
first-order refinement. This is observed evidence in one synthetic case, not
an a priori error bound or proof of general contact-transition convergence.
The 240-to-480 reversal and all earlier nonmonotone spin values remain visible.
Tangential linear velocity has much smaller, non-asymptotic differences; a
decreasing mixed norm cannot certify every output component.

These standalone scientific runs explicitly allow 60000 total evaluations
for the larger grids, with the same 150 per-step limit and 1e-10 residual
tolerance. They are not a relaxation of regression-test budgets or deadlines.
The earlier published cb7219f38 Linux Python3.11 shard timed out at the unchanged
60-second limit in the zero-gap release test (job102797151274); its Python3.12
shared shard passed. The repaired source still needs hosted qualification.
Review #5160 and draft PR#5162 remain open for contact-transition/componentwise
qualification, CI and review. All physical/acoustic parent requirements remain.

## Independent planar sliding reference

The test-only `_friction_sliding_reference.py` supplies a continuous saturated
sliding oracle in addition to the sticking reference. It shares the canonical
shaft/ball mechanical response, normal law and Lie-chart differential. It
does not use the endpoint nonlinear solver or discrete tangential return map;
tests disable both while checking its continuous work balance.

Let v_t = s e with s > 0, |e| = 1. In planar motion with fixed slip sign,
e rotates with the face frame. With constant mu and k_t, a saturated elastic
history is z = r e, r = mu F_n/k_t. The objective history rate then gives

```text
lambda = s - r_dot > 0
r_dot = mu F_n_dot / k_t
F_t = -mu F_n e
D_plastic_dot = mu F_n lambda
E_t_dot = k_t r r_dot
mu F_n s = E_t_dot + D_plastic_dot.
```

The explicit positive-lambda check is essential: if the normal cap expands
faster than slip loads the spring, this saturated branch is invalid and the
reference refuses it. A shrinking cap releases stored tangential energy;
plastic work consequently cannot be replaced by mu F_n s alone. These are
properties of the declared model, not measurements of heat or acoustic energy.

While compression and raw normal force remain positive,
F_n_dot = -k_n g_dot - c_n g_ddot. For a fixed material plane, let d be the
ball-center minus face-origin vector in world axes. Its material plane offset
contributes a constant to the gap and drops out of the derivatives:

```text
g_ddot = n_ddot dot d + 2 n_dot dot d_dot + n dot d_ddot
n_dot = omega_face cross n
n_ddot = alpha_face cross n + omega_face cross (omega_face cross n).
```

World origin acceleration is R (v_body_dot + omega_body cross v_body).
The reference retains these rotating-frame terms; three independent expanded
world-coordinate controls check them with nonzero plane offsets and body spin.
The terms affect the changing contact force bound. They do not introduce a
second centrifugal potential or turn Coriolis action into material damping.

Two synthetic 40-microsecond cases use mu=0.02 and 5 m/s tangential ball speed,
with normal ball velocities -0.4 and +0.4 m/s. One has growing tangential
storage and one has shrinking storage. The original reference refused loss of
compression or force; the release extension below permits zero force after
cutoff while retaining planar, slip-sign and positive plastic-rate checks. No onset, reversal, stick/slide transition, force cutoff or
separation lies inside these reference intervals.

DOP853 and a tenfold tighter reference agree in each reported output to 1e-10
in that output's stated SI unit. Against them, the 4/8/16-step production
solutions satisfy the preset first-order ratio bounds separately for normal
force, world tangential x-force, world ball x-speed, world y-spin, normal and
tangential x-impulse, and plastic work. All five other work ports satisfy their
separate 1e-6 J finest-grid bounds. The combined shaft/ball pose, twist and
history error decreases, and tangential algorithmic loss decreases; that
combined state norm is not a componentwise certificate for every shaft mode.

The continuous total-energy defects are about 1.38e-15 and 2.10e-15 J. The
tangential storage changes are +1.17e-6 and -3.58e-6 J, respectively. The seven
initial reference controls pass after the missing-helper and new-parameter
RED records; exact final source, typing and regression receipts belong to
FRICTION_SLIDING_RESULTS.json. Final expanded Windows verification passes267 tests
in104.24s at3a46d6d0 with no failures or skips. A later direct-component access
refactor passes all nine sliding/sticking reference controls in21.17s, with
equations unchanged. Both modified reference files pass NumPy-aware mypy and
the file/function/attribute-depth contracts. This strengthens sliding qualification within
the stated branch and retains the general event/mesh/mode, material,
radiation and perception requirements.

## Independent force cutoff and full separation

The same planar saturated branch now continues through unloading to 3 ms.
At positive normal force its moving Coulomb radius and plastic power retain
the equations above. At zero normal force, tangential traction, elastic
history and plastic power are zero. The canonical normal law continues to
account for stored compression until the geometric gap clears. These are
initially compressed, constant-sign sliding cases, not a general hybrid
contact solver: onset, reversal, recontact and stick/slide switching remain
outside this oracle. An initial history off the saturated branch is refused.

DOP853 independently locates the raw normal-force zero and signed-gap zero.
Both event families are retained in owned, read-only trajectory records.
Detection uses accepted-step sign changes; more than one detected crossing
per family is refused, but that refusal cannot establish completeness for
unobserved repeated or grazing events. The reference shares the canonical
mechanical equations, normal constitutive law and Lie-chart differential,
while independently integrating time, sliding force, impulses and work.

| Initial normal ball speed (m/s) | Force cutoff (ms) | Separation (ms) | Stored normal energy at cutoff (J) |
| ------------------------------- | ----------------- | --------------- | ---------------------------------- |
| -0.4                            | 2.1385984324      | 2.2869284692    | 0.0001757327312                    |
| +0.4                            | 1.0914164802      | 1.2397682334    | 0.0002112765680                    |

The normal-force cutoff occurs while the gap remains negative. Subsequent
cutoff-work accumulation equals the stored normal energy at that event to
1e-9 J. After separation, independent free-ball checks verify constant world
linear velocity, linear center motion and constant world angular velocity to
1e-8 in the respective SI units. The reference ends with zero contact force
and tangential storage, positive clearance and positive separation speed.

The complete continuous energy check is

```text
(E_mechanical+normal,final - E_mechanical+normal,initial - E_t,initial)
- W_external + W_anchor + D_grip + D_normal_viscous + D_normal_cutoff
+ D_plastic = 0.
```

Here the final tangential storage is zero. The two residuals are approximately
-2.31e-13 and -3.91e-13 J, comfortably inside the preset 1e-8 J bound. A tenfold
tighter reference agrees separately in all twelve reported SI channels to
1e-8. Cutoff and plastic losses remain model work channels; neither is a
measurement or prediction of heat, radiated sound, or perceived sweetness.

Production backward Euler uses 30/60/120 steps across the same 3 ms interval,
with the unchanged 30000 total / 150 per-step evaluation limits and 1e-10
requested residual tolerance. Against the continuous endpoint, separate
x-velocity, y-spin, normal impulse, x tangential impulse, plastic work and
z-velocity errors satisfy preset successive-error ratios of 1.5 to 2.5.
Their finest-grid absolute limits are respectively 0.01 m/s, 0.05 rad/s,
0.0005 N s, 0.00001 N s, 0.0001 J and 0.02 m/s. Each of the five other work
channels decreases in error and satisfies its own 0.0005 J finest-grid bound.
Full scaled-state error, mechanical energy defect and positive tangential
algorithmic loss also decrease. No individual shaft-mode or peak-force
certificate follows from a combined endpoint norm.

The missing unloading behavior and missing event/history contracts were
recorded as RED before their test-only implementation. Continuous release,
event/free-ball and production comparison checks then passed before fixture
consolidation. The final module shares immutable coarse/fine traces so all
assertions reuse the same expensive reference integrations. The invalid-history
control constructs a fresh reference. No production source, accuracy bound,
test deadline or solver budget changes. Exact sources, JUnit identities,
component errors and final regression results are in FRICTION_RELEASE_RESULTS.json.

Final expanded Windows verification at SELF passes all274 tests in195.17s,
with no failures or skips. Both modified files pass NumPy-aware mypy, Ruff
and file/function/attribute-depth contracts. Hosted qualification remains open.

The subsequent [first-contact and sliding-entry reference](FRICTION_ENTRY.md)
extends verification to an initially separated planar case and documents the
remaining event-work limits. Its282-case receipt has its own source identity;
no earlier result is relabelled as a new run.
