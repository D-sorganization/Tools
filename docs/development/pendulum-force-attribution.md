# Pendulum Force Attribution and Impulse Optimization

Issue [#4698](https://github.com/D-sorganization/Tools/issues/4698) defines the
canonical Tools contract for explaining motion-dependent terms in pendulum and
movement-optimization models. The implementation is
`shared.python.swing_sim.force_attribution`, schema
`force-attribution/v1`.

## The Equation Being Explained

For fixed generalized coordinates \(q\), Tools writes the model as

\[
M(q)\ddot q+h(q,\dot q)+g(q)+d(\dot q)=\tau.
\]

The terms have one declared meaning:

- \(M\ddot q\) is generalized inertia.
- \(h\) is the complete velocity-dependent bias.
- \(g\) is the gravity generalized-force term.
- \(d\) is the declared dissipative term.
- \(\tau\) is the applied generalized control.

For plotting a source as a contribution to acceleration, the code uses the
equal-and-opposite generalized drive. Thus the Coriolis drive is
\(-h_{\mathrm{cross}}\), the gravity drive is \(-g\), and the control drive is
\(+\tau\). This sign choice closes exactly:

\[
M\ddot q=\tau-h_{\mathrm{cross}}-h_{\mathrm{squared}}-h_{\mathrm{residual}}-g-d.
\]

## What “Coriolis” and “Centripetal” Mean Here

The first-kind Christoffel symbols are

\[
\Gamma_{ijk}=\frac12\left(
\frac{\partial M_{ij}}{\partial q_k}+
\frac{\partial M_{ik}}{\partial q_j}-
\frac{\partial M_{jk}}{\partial q_i}
\right).
\]

Tools expands \(h_i=\sum_{jk}\Gamma_{ijk}\dot q_j\dot q_k\) into two
auditable monomial groups:

- `squared_speed`: all \(\Gamma_{ijj}\dot q_j^2\) terms;
- `coriolis`: all \(2\Gamma_{ijk}\dot q_j\dot q_k\), \(j<k\), terms.

Squared-speed terms describe the generalized effects associated with curved
motion. In an inertial-frame mechanism description, **centripetal** names the
inward acceleration or real resultant force required to bend a path. A
**centrifugal** force is the equal-and-opposite inertial term introduced when
the same motion is described in a rotating frame. The UI or a paper may show
both interpretations, but it must not add them together as two physical
forces.

The cross-versus-squared split is not coordinate invariant. Changing from a
relative wrist angle to an absolute club angle changes the monomials and can
move a term between named groups while leaving the total \(h\) and predicted
motion unchanged. Every result therefore records the coordinate names and the
`christoffel_first_kind_cross_vs_squared_speed` convention. The independently
implemented model bias is compared with the Christoffel sum; any disagreement
is retained as `velocity_residual`, never hidden.

For the Tools relative-coordinate double pendulum,

\[
k=m_2l_1l_{c2}\sin q_2,
\]

\[
h_{\mathrm{cross}}=
\begin{bmatrix}-2k\dot q_1\dot q_2\\0\end{bmatrix},\qquad
h_{\mathrm{squared}}=
\begin{bmatrix}-k\dot q_2^2\\k\dot q_1^2\end{bmatrix}.
\]

These two vectors sum to the existing reference implementation's complete
Coriolis/centripetal bias.

## From Generalized Terms to Force Along the Hand Path

A generalized force is not automatically a Cartesian hand force. For an
endpoint Jacobian \(J\), virtual work gives

\[
Q=J^TF.
\]

The contract computes the least-squares force-only equivalent and always
returns

\[
Q=J^TF+r,
\]

where \(r\) is the unreconstructed generalized residual. It also reports the
Jacobian rank and one of `exact_force_only`, `rank_deficient_force_only`, or
`least_squares_force_only`. This is essential at the wrist: a joint couple can
do generalized work but cannot be silently converted into a net linear hand
force.

When endpoint speed is nonzero, the tangent and along-path force are

\[
e_t=\frac{J\dot q}{\lVert J\dot q\rVert},\qquad
F_\parallel=F\cdot e_t.
\]

At zero speed the tangent is undefined, so the result is unavailable rather
than guessed from neighboring samples.

## Impulse, Power, and Work Are Different

For each source, the trajectory contract reports:

- signed generalized impulse \(\int Q\,dt\);
- absolute generalized impulse \(\int |Q|\,dt\);
- signed tangent impulse \(\int F_\parallel\,dt\);
- absolute tangent impulse \(\int |F_\parallel|\,dt\);
- generalized power \(Q^T\dot q\) and work \(\int Q^T\dot q\,dt\);
- mapped endpoint power \(F^TJ\dot q\) and work;
- tangent-impulse cancellation; and
- mapping residuals.

Impulse measures accumulated force over time. Work measures energy exchange
and includes velocity. A large centripetal force perpendicular to motion can
have large absolute impulse and nearly zero work. Signed values can also hide
large opposing phases, which is why absolute values and cancellation accompany
every signed integral.

## Optimization Contract

`component_impulse_objective` returns the negative signed or absolute tangent
impulse so an ordinary minimizer can maximize the chosen component. Movement
Optimizer exposes the specific
`coriolis_hand_path_impulse_cost` adapter.

Maximizing that scalar alone is an instructive mechanism experiment, not a
recommended swing. A qualified study must report at least:

- achieved Coriolis impulse and total along-path force;
- clubhead or distal speed;
- negative and total work;
- peak force and torque;
- joint/range/contact feasibility;
- mapping rank and residual;
- sensitivity to bounds, coordinate convention, and timing; and
- robustness on held-out or perturbed cases.

MacKenzie-style average force along the hand path is linear work divided by
path length. It is neither time-average force nor impulse. A correlation
between that net inverse-dynamics quantity and clubhead speed does not identify
which coordinate term caused the force, and none of these model terms identify
muscle activation, co-contraction, metabolic cost, or perceived effort.

## Verification

The TDD gate checks the analytical two-link formula, zero-velocity limits,
force-only rank/residual behavior, equation closure, acceleration closure,
trajectory integrals, objective sign, and invalid input contracts. Downstream
repos must pin both the Tools commit and `force-attribution/v1`; they must fail
closed on an unknown schema.
