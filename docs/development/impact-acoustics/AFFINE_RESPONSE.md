# Residual-Forced Linear Response: Tools #5072

The private `_shaft_affine_response` companion extends the existing homogeneous
Lyapunov assessment to a declared constant affine ODE. It retains the nominal
residual load, a separately assumed input-error bound, and the original
homogeneous evidence. It does not identify a golfer, infer a constant operating
history from a snapshot, or qualify contact, acoustics or a nonlinear swing.

## Coordinates and model

Let physical nodal perturbations be q=S y, where S uses the declared translation
length and unit rotation scales. For unscaled residual r and physical operators,
the length-scaled quantities supplied to this kernel must be

```text
rs = S.T r
Ms = S.T M S, Gs = S.T G S, Cs = S.T C S, Ks = S.T K S.
rs + Ms ydd + (Gs+Cs) yd + Ks y = 0.
```

The matrices undergo congruence; the residual undergoes the single left
transformation. Applying the matrix transformation to the vector is incorrect.
With tau=t/T and x=(y,dy/dtau), the existing dimensionless generator A gives

```text
dx/dtau = A x + [0, -T^2 Ms^-1 rs]
dx/dt   = (A/T) x + u0,       u0 = [0, -T Ms^-1 rs].
```

The code stores `input_per_s=u0`, with its sign, and the copied left-side
residual. No balance tolerance sets a small residual to zero. A nearly balanced
configuration can have a persistent response. This kernel does not solve for
a shifted equilibrium or prove that it remains within a nonlinear strain
domain. It preserves the regular positive-mass domain of the existing kernel;
singular descriptor systems require another formulation.

## Conditional response envelope

Reuse the existing assessment of the unchanged A, including its resolved
Lyapunov candidate and assumed dimensionless operator-error bound. Its constants
K >= 1 and c > 0 give a physical-time homogeneous propagator envelope
K exp(-c t). The following variation-of-constants derivation is for this model.
For measurable additional input e(t), assume a uniform scaled-state bound
norm(e(t)) <= epsilon_u over the interval. The user supplies epsilon_u in
scaled-state units per second; it is separate from the operator uncertainty.
Set U=norm(u0)+epsilon_u. Then

```text
x(t) = Phi(t,0)x(0) + integral_0^t Phi(t,s)[u0+e(s)] ds
norm(x(t)) <= K exp(-c t) norm(x(0))
              + K U integral_0^t exp[-c(t-s)] ds
           = K exp(-c t) norm(x(0)) + K U (1-exp(-c t))/c.
```

The nominal operator is constant. The inherited common quadratic storage
argument also bounds a transition matrix under the declared uniform additive
operator perturbation; this is an assumption on that perturbation, not a
qualification of arbitrary time-varying swing coefficients.

`ForcedEnvelope.bound` returns the initial-state and input terms separately.
The input term approaches K U/c, not zero. This is an upper envelope, not an
exact steady-state displacement or response simulation. U=0 recovers the
existing homogeneous result exactly at the model level. The Euclidean norm
depends on S and T and mixes scaled displacement and velocity. It is not
physical energy, sound pressure, subjective sweetness, an H-infinity transfer
norm, or a measured error bound.

The distinction between free decay and externally supplied energy follows the
input/storage framework discussed in
[Tedrake's MIT notes on dissipation inequalities](https://underactuated.mit.edu/robust.html).
Those notes explain why storage can increase when inputs act. The particular
norm envelope above is derived here from our existing propagator bound; the
notes' L2-gain discussion is not substituted for this uniform-input bound.

## Numerical domains and TDD

All control scalars, residual entries, time and initial norm must be strictly
real and finite; booleans, strings and complex values are refused. Time, initial
norm and input-error bound are nonnegative. Residual/result tuples own their
values. Missing homogeneous decay evidence produces `not_established`, without
an instability verdict or a forced-response envelope.

`hypot` retains small nonzero input norms that a naive squared norm can lose.
The convolution uses `expm1` and its small-time limit, including when c\*t itself
underflows. Multiplication order avoids an unnecessary overflow before a zero
or small-time factor. If a nonzero residual produces an entirely zero nominal
input by underflow, qualification is refused. Positive response terms that
underflow to zero are also refused, as are nonfinite outputs. These checks can
conservatively refuse extreme values even when a differently scaled computation
would succeed. They are not directed rounding or interval certification; the
existing Lyapunov/eigenvalue numerical-evidence limits still apply.

The initial test failed to import the absent module (RED). The first
implementation passed 32 cases. Two additional controls reproduced zero-time
overflow and a false zero-input claim after solve underflow; both were repaired.
Two further controls reproduced silent loss of positive bound terms, now
refused. The final 36 affine tests and 61 existing decay/spectral tests pass
together (97 tests, 8.17 s on Windows Python 3.12.10). Scoped Ruff 0.14.10 and
the actual mypy 1.13 pre-push configuration pass. No function exceeds 50 lines. Linux Python 3.11.15 with NumPy 2.3.5 and
SciPy 1.15.3 passes the same controls plus all shared-package API checks:
106 tests in 222.44 s, with three unavailable-plugin configuration warnings.

The independent trajectory oracle is qdd+2qd+q=3, q(0)=qd(0)=0:
q=3[1-(1+t)exp(-t)], qd=3t exp(-t). The tests compare its scaled state to the
envelope at three declared time scales. Other controls cover copied residuals,
sign, zero input, nonzero input uncertainty, operator uncertainty, neutral
systems and strict invalid domains. Sampling checks the implementation against
the analytic oracle; the conditional all-time statement follows from the
derivation, not from sampling. All fixtures are synthetic.

## Integration and remaining qualification

This is a private reusable kernel with empty exports. Existing frozen-spectrum
status remains `unqualified`. The next integration must explicitly prescribe
constant frame angular velocity, appropriate origin motion, stationary grip
anchors and constant load laws before constructing a constant gripped model.
A sample with zero angular acceleration does not establish a constant history.
The nominal residual must use the same coordinate scaling as M/G/C/K.

After that, qualify full/reduced modal and FRF bandwidth, time/mesh convergence,
moving boundary work, flexible contact, acoustic radiation and measured/blinded
validation. All nine final manual/inventory/handoff gates and repository Ruff 0.14.10
pass (3,789 formatted Python files). The current inventory labels this new
numerical module non-calculation: a confirmed false negative, not accepted
scientific categorization. The #5103 classifier fix is still required before
combined delivery. Manual gates remain structural and unapproved;
this research note is not an approved engineering-manual chapter.
