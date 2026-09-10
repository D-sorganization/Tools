# Frozen Shaft Spectra: Diagnostics and Limits

Tools #5072 remains partial. The loaded M/G/K checkpoint is published at
`78253741f`. The private `_shaft_spectrum.py` now diagnoses those operators
without replacing nonsymmetric stiffness, deleting growing modes or inferring
a stability certificate. All prior public signatures remain unchanged.

## Reuse and Physical Preconditions

`clamped_chain_spectrum` recomputes `linearized_chain_dynamics` at supplied
proper poses, reuses the equilibrium strain-domain checks, and verifies every
free-node force and moment against the caller's declared tolerances. It clamps
only node zero and retains its support-on-shaft material wrench. A mutable
candidate record is never accepted as proof of balance. No mass regularization,
straight-shaft substitution, modal clipping or inferred damping is performed.

The mass must be finite, symmetric within the numerical tolerance, positive
definite and resolved above the declared reciprocal-condition floor. Gyroscopic
transport must be skew within that tolerance. K can be nonsymmetric, including
circulatory loading. These restrictions describe this private ODE solver;
singular descriptor systems require a separate formulation, not mass repair.

## Coordinate and Eigenproblem Derivation

For free material increments q, start with M q'' + G q' + K q = 0. Let
q=S x, with S repeating (L,L,L,1,1,1) for each free node, and let t=tau theta.
The congruent operators are Mbar=S.T M S, Gbar=S.T G S, Kbar=S.T K S.
The dimensionless state z=(x,dx/dtheta) therefore obeys

```text
dz/dtheta = A z
A = [[0, I], [-tau² solve(Mbar,Kbar), -tau solve(Mbar,Gbar)]]
A z_j = nu_j z_j; physical rate lambda_j = nu_j/tau.
q_mode = S z_j[:n]; velocity_mode = S z_j[n:]/tau.
```

Length/time scales are positive finite caller inputs, not fitted parameters.
They change numerical conditioning; resolved physical rates are invariant.
The time-square underflow boundary is refused explicitly. Matrix overflow,
nonfinite results, unresolved mass and failed eigensolutions also fail closed.
The solver uses the complete nonsymmetric eigenproblem and linear solves.

Two independent residual diagnostics are required. The state residual is
norm(Az-nu z)/((norm(A)+abs(nu))*norm(z)). The original quadratic residual is
norm((lambda² Mbar+lambda Gbar+Kbar)x) divided by
(abs(lambda)² norm(Mbar)+abs(lambda) norm(Gbar)+norm(Kbar))*norm(x).
Matrix norms here are Frobenius; vector norms are Euclidean. Both must satisfy
the explicit residual tolerance. Nonzero displacement modes are required;
the identically zero free-particle pencil at lambda=0 has residual zero.

The reciprocal condition of the scaled eigenvector matrix is also returned.
This is not an individual eigenvalue sensitivity estimate or a physical
uncertainty interval. Repeated eigenspaces have no unique basis; modes have
arbitrary complex normalization. Small backward errors can coexist with large
forward errors, especially near defective or clustered eigenvalues. Linearized
state and quadratic residuals must not be conflated. These numerical concerns
and gyroscopic eigenproblems are treated in Tisseur and Meerbergen (2001),
sections 3.10 and 4–5, [The Quadratic Eigenvalue Problem](https://eprints.maths.manchester.ac.uk/466/1/38198.pdf).
The implementation's explicit scaling, restrictions and fixtures are derived
above; this citation does not qualify a golf model or its measured parameters.

## Falsifying Controls

- Two uncoupled oscillators recover exact positive/negative imaginary rates,
  velocity/displacement consistency and small residuals.
- M=I, K=-I and G=0 retain exponential rates +/-1. With G=3J, where
  J=[[0,-1],[1,0]], the characteristic polynomial is lambda^4+7lambda²+1.
  Its roots are +/-i(3+sqrt(5))/2 and +/-i(3-sqrt(5))/2. Negative stiffness
  alone therefore does not classify a gyroscopic system as unstable.
- M=1, K=G=0 has zero eigenvalues but defective state evolution:
  q(t)=q(0)+v(0)t. Zero real parts alone do not establish bounded motion.
- K=[[4,1],[-1,4]] with M=I, G=0 retains roots whose squares are -4+/-i;
  symmetric projection would erase the growing solutions.
- A deliberately corrupted fast mode in a widely separated two-oscillator
  fixture passes the state residual tolerance but fails the quadratic residual.
- The unloaded two-element rod agrees with an independent symmetric
  generalized eigensolution. Changing length or time scales preserves rates.
  Rotating unbalanced and strain-exceeding states are refused; a solved loaded
  root retains its computed support reaction.

These controls, malformed inputs, output isolation and numerical refusal cases
are in `tests/shared/python/golf_club/test_shaft_spectrum.py`. The initial 14
cases fail on a missing module and then pass. A boundary extension catches
time-square underflow before its fix. The two-residual extension first fails
on the missing diagnostic and deliberately corrupted mode; all 22 final cases
pass on Windows (6.20 s). The preceding broader Linux checkpoint passes 588
golf/API tests, two optional CAD skips and three optional-plugin configuration
warnings (220.42 s). The final two-residual implementation passes all 589
golf/API tests on Linux, with the same two skips and three warnings (245.81 s).
Ruff 0.14.10 passes repository-wide (3,735 Python files), scoped mypy passes,
and all nine manual gates pass before and after. The deterministic inventory
currently gives this module a false-negative `non-calculation` label: its
scientific-import regular expression omits whitespace after `import`/`from`,
and the filename contains none of its calculation path markers. This module
does perform calculations; the automatic label cannot establish otherwise.
Tools #5101 tracks a separately reviewed correction and regenerated inventory.
No publication approval or registered textbook pathway is inferred.

## Remaining Qualification

Every result still says `stability_status="unqualified"`. Frozen linear
eigenvalues do not establish nonlinear stability, time-varying swing stability,
robustness to parameter uncertainty, physical damping, transient amplification,
mesh bandwidth, contact validity or acoustic radiation. A balanced snapshot
with zero instantaneous frame acceleration does not prove autonomous future
motion. Moving-boundary power, loaded grip/head integration, convergence,
forced response and experimental/blinded controls remain required. In
particular, these synthetic spectra cannot explain a player's sweeter sound.
