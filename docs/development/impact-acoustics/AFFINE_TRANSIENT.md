# Constant Affine Transient Motion: Tools #5072

The private `affine_state_at` evaluator returns finite-time motion of the
explicitly prescribed constant local shaft model. `ConstantGrippedModel`
delegates through `scaled_state_at`, retaining its owned pencil, residual and
scales. This adds motion to the existing spectra and conditional response
bounds. It does not change the strain/balance checks or establish that a real
swing has constant inputs. See CONSTANT_GRIPPED_MODEL.md for those prescriptions.

## Equation and coordinates

Use the same work-conjugate length scaling q=S y, coefficient matrices
S.T operator S, and left-side residual r=S.T r_physical as the existing model:

```text
M ydd + (G+C) yd + K y + r = 0
x = [y; T yd]
dx/dt = (A_T/T) x + u_s
A_T = [[0, I], [-T^2 M^-1 K, -T M^-1 (G+C)]]
u_s = [0; -T M^-1 r].
```

Here t and T are seconds. Both input and returned state are x, with translations
and rotations in the declared length scaling. In particular, the second half
is T times the scaled physical velocity; it is not velocity in m/s. The return
is an owned tuple. Initial conditions are at t=0, not at the first sampling time.

For constant coefficients and residual, append a unit coordinate:

```text
H(t) = [[(t/T) A_T, t u_s], [0, 0]]
[x(t); 1] = exp(H(t)) [x(0); 1].
```

Differentiating the augmented exponential yields the stated affine equation
and its initial condition. This formulation does not invert K or A_T and does
not diagonalize the generator. A free particle, neutral rigid mode or defective
critical-damping block therefore remains representable. Finite growing motion
is retained as a diagnostic result, not relabeled as stable motion.

The evaluator reuses the existing damped-generator validation and residual
input conversion, so there is no second mass/gyro/damping or force-sign kernel.
It retains nonsymmetric K. The existing positive resolved mass, skew G and
passive C checks remain numerical coefficient checks with their declared
tolerance, not identified material or hand properties.

## Numerical method and limits

SciPy's dense `expm` evaluates the augmented matrix. Its documented method is
variable-order Pade approximation with scaling and squaring; see the
[SciPy 1.15.3 primary documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.linalg.expm.html)
and its cited Al-Mohy/Higham algorithm. The package already uses this numerical
library for SE(3) calculations. No new dependency or custom exponential is added.

There is no time-step discretization in this evaluator. Requesting a denser
output grid is not an independent time-convergence demonstration. Floating
point exponential evaluation still has conditioning and roundoff limits;
this interface supplies no certified forward-error bound. Dense cost also
limits large discretizations. Separate modal/mesh/bandwidth qualification and
independent transient comparisons remain required for the distributed model.

Inputs must be finite strictly real numbers, with a nonnegative elapsed time.
Zero time returns an owned copy after plant and residual validation. Nonfinite
evaluation is refused. Underflow while forming a nonzero exponent coefficient
is refused; a positive time ratio that rounds to zero is checked explicitly.
Representable tiny residual motion is preserved. These checks do not amount
to a complete floating-point error certificate for `expm`.

For the conservative scalar oracle, energy is
E=(v^2+9q^2)/2. Its numerical conservation is an independent analytic check.
For a general nonsymmetric K, using y.T K y/2 as a conserved potential would
discard circulatory work. No such general energy claim is made here, and the
scaled-state norm is not mechanical energy or acoustic amplitude.

## Verification and integration evidence

The initial test run failed because the transient module did not exist.
Thirty-four initial cases then passed. An additional regression exposed a
positive-time ratio rounding to zero: 35 passed and one failed before the
explicit check, followed by all 36 transient cases passing.

Analytic oracles cover a forced critically damped oscillator across three time
scales, a forced free particle, an undamped oscillator and its energy, and an
unstable oscillator. A separate physical-coordinate DOP853 integration checks
a coupled system with non-diagonal mass, gyroscopic coupling, damping and
nonsymmetric stiffness at three time scales. The gripped adapter preserves the
same coefficients, residual and coordinate convention. Strict input, ownership,
semigroup, overflow and tiny-input tests complete the local contract.

The combined transient/operating/affine suite passes 88 Windows tests in
11.74 seconds. Scoped Ruff 0.14.10 and actual pre-push mypy 1.13 arguments pass.
The broader Linux golf/signal/API regression passes 860 tests in 274.37 seconds,
with two optional CAD skips and three unavailable-plugin configuration warnings.
It uses the preserved Python 3.11.15, NumPy 2.3.5 and SciPy 1.15.3 environment
and one BLAS/OMP/MKL thread. All nine manual/inventory/handoff gates pass;
repository Ruff 0.14.10 passes with 3,794 formatted files. The API baseline adds
one empty-export private module; every prior API entry is unchanged. Normal
commit/push publication remains.

The current inventory classifier still misses scientific imports and labels
this new calculation module non-calculation. That known false negative is
tracked by #5101 / PR #5103, which must be integrated before combined delivery.
The canonical inventory is regenerated without hand-editing its classification;
its structural success is not scientific approval or a complete textbook pathway.

This is synthetic numerical evidence. Contact excitation, moving nonlinear
work, radiation, measured grip/shaft parameters and blinded sweetness results
remain separate required work. The model continues to report physical/nonlinear
stability as unqualified.
