# Moving Grip: Finite-Pose Kinematics and Power-Conjugate Ports

This T3 #5072 checkpoint adds private kinematics, not a nonlinear human grip
model. `grip_impedance.py` remains a fixed-frame, small-rotation constitutive
port. Its states must not silently be reinterpreted as finite poses. Loaded
shaft/grip balance, the constitutive extension, tangent/transport assembly,
time evolution, stability and calibrated bandwidth remain open.

## Reuse and State Convention

`_grip_moving_kinematics.py` reuses `_shaft_se3._rigid_pose`, `log_pose`,
`right_jacobian` and its analytic Frechet derivative, plus the shared strict
array/identifier contracts. No separate rotation-series approximation or
numerical Jacobian is introduced. The existing principal-log margin below pi
is retained. State arrays are copied into immutable tuples; result arrays are
fresh. The observer identifiers must agree.

Each pose H=[R,p;0,1] maps material coordinates into a common observer frame.
Its linear-first body twist V=[v;omega] satisfies Hdot=H hat(V). The supplied
twist_rate is the derivative of these material components. In particular,
vdot=R^T pddot-omega cross v; it is not simply R^T pddot. A state record cannot
prove consistency with a measured trajectory. A fixed rigid observer change
preserves the following relative outputs; an arbitrary moving observer needs
its own consistent transport definitions.

## Project Derivation: Actual Separation and Rotation Coordinates

Subscripts r and a denote the shaft-root point and anchor point. Define
Q=Ra^T Rr, d=Ra^T(pr-pa), phi=Log_SO3(Q), and q=[d;phi]. The translation d
is the actual relative point separation, not the translational generator of
Log_SE3(Ha^-1 Hr). This distinction matters whenever rotation is finite.

With Jr(phi) the SO(3) right Jacobian and h=omega_r-Q^T omega_a,

```math
\dot d=Qv_r-v_a+d\times\omega_a,\qquad
\dot\phi=J_r(\phi)^{-1}h.
```

The right-Jacobian rotation block is taken from the existing SE(3) kernel at
[0;phi]. Differentiating the same maps, rather than treating their coefficients
as constant, gives

```math
\dot h=\dot\omega_r+h\times(Q^T\omega_a)-Q^T\dot\omega_a,
\qquad
\ddot\phi=J_r^{-1}(\dot h-\dot J_r\dot\phi),
```

```math
\ddot d=Q(h\times v_r+\dot v_r)-\dot v_a
        +\dot d\times\omega_a+d\times\dot\omega_a.
```

The derivative of Jr is evaluated in direction phidot by the existing
matrix-exponential Frechet implementation. No Euler-angle rates or omitted
moving-frame terms are substituted. At zero relative rotation the Jacobian
has its regular identity limit. At the principal branch boundary the result
is refused, not continued through an arbitrary axis flip.

## Two Physical Ports and Boundary Work

Writing qdot=Ar Vr+Aa Va gives

```math
A_r=\begin{bmatrix}Q&0\\0&J_r^{-1}\end{bmatrix},\qquad
A_a=\begin{bmatrix}-I&[d]_\times\\0&-J_r^{-1}Q^T\end{bmatrix}.
```

Let g be an effort conjugate to these coordinate rates. It is not generally
a physical wrench that can be applied unchanged at either material point.
The physical root and anchor reactions are wr=-Ar^T g and wa=-Aa^T g. Therefore

```math
w_r^T V_r+w_a^T V_a=-g^T\dot q.
```

Their world forces sum to zero. Their world torques also close when the root
force's moment arm about the anchor is included. This supplies the correct
power accounting boundary for a future constitutive law. For example, a newly
declared constant-coefficient coordinate law g=M qddot+C qdot+K q has storage
E=(qdot^T M qdot+q^T K q)/2 and loss D=qdot^T C qdot when M,C,K are symmetric
positive semidefinite. Then root-plus-anchor power is -Edot-D. If anchor motion
is prescribed, the anchor-port contribution into the attached system is
-wa^T Va in the declared observer; it must be retained separately from
dissipation. Individual port power depends on the observer. Use inertial
motion states for an inertial energy ledger; if the observer moves, retain
its frame-work terms before interpreting this contribution as actuator power.
The sum of the two internal-port powers is invariant under a common rigid
observer motion because their total force and moment close. This law was a
candidate at the kinematic checkpoint; the subsequent private implementation
and independent tests are documented in `FINITE_GRIP_RESPONSE.md`. It is not
an identification of hand mass or a reinterpretation of the local grip API.

The geometric-control literature supports deriving force laws from declared
pose potentials instead of treating finite orientation coordinates as ordinary
Cartesian errors. [Seo et al. (2024), sections III–V](https://arxiv.org/html/2401.13190v1)
compare group- and logarithm-based potentials and their different behavior.
That robot-control comparison does not calibrate golf grips or establish
stability for the present shaft system. The project derivation above uses
actual Cartesian separation plus a rotation vector to preserve the existing
meaning of translation; it is not a copy of their SE(3) logarithmic potential.

## TDD and Independent Checks

The initial test collection is RED because the new module is absent. The first
six controls pass (6.04 s), followed by extraction of relative geometry to keep
every function within the repository's 50-line limit. Eight final new controls
and the 23 existing local-grip tests pass together (31 passes, 6.25 s).

The derivative oracle constructs finite pose curves with SciPy's matrix
exponential and extracts actual point differences and rotation vectors. It
does not call the production rate/Jacobian functions. Fourth-order centered
differences check velocity to absolute 2e-10 and acceleration to 3e-7 at a
2e-4 s step. Other controls independently verify world force/moment closure,
power duality, fixed-observer invariance, rigid common-motion cancellation,
zero-angle behavior, principal-branch refusal, immutable inputs, mismatched
observers, strict real inputs and arithmetic-overflow refusal.

Scoped Ruff and actual pre-push mypy pass. The API baseline adds only one empty
private-module entry; all existing public surfaces remain identical. The first
broad run times out in the unchanged rotating-rod convergence control inside
small-matrix linear algebra. An isolated run with one BLAS thread passes in
54.72 s. The complete Linux golf/API run then passes 628 tests in 174.65 s,
with two optional CAD skips and three unavailable-plugin configuration warnings.
Both runs set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS to 1;
equations, tolerances and the 60-second per-test limit remain unchanged. This
runtime observation does not identify the original worker's BLAS thread count.
All nine final manual gates pass. Repository Ruff 0.14.10 passes (3,742 Python
files), as does the actual pre-push mypy hook. Implementation is published at
`0d45c4b7f32e0d2917c93e9ef6e492c032580dd7`, with its remote SHA verified and
every normal commit/push hook passing. The published evidence is recorded in
[issue #5072](https://github.com/D-sorganization/Tools/issues/5072#issuecomment-5591328984).
The subsequent finite law and stationary loaded balance/tangent are documented
in `FINITE_GRIP_RESPONSE.md` and `GRIPPED_CHAIN.md`. Retaining anchor work in
time evolution remains required. Do not bolt a finite moving anchor onto the
frozen clamped response or claim that local positive damping proves swing
stability.
