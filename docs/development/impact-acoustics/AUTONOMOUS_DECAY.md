# Constant-System Decay Assessment: Tools #5072

The private `_shaft_autonomous_decay` companion assesses only a constant,
homogeneous, regular finite-dimensional linear ODE. It shares plant validation
and the scaled generator with `_shaft_damped_spectrum`. Frozen spectrum status
remains `unqualified`. No driven swing, nonlinear contact, physical energy,
golfer uncertainty, sound quality or experimental validation follows from this
operation.

## Coordinates and Derivation

Supply already length-scaled coefficients M/G/C/K for q=S y and t=T tau.
The shared generator acts on x=(y,dy/dtau):

```text
A = [[0, I], [-T^2 M^-1 K, -T M^-1 (G+C)]]
```

The existing kernel requires finite, resolved positive mass, skew G and
symmetric semidefinite C within explicit tolerances. It retains nonsymmetric K
and the original coefficients without projection or regularization. A finite
grip spectrum does not itself establish that this constant-system scope holds.

Choose P using the Lyapunov equation A.T P+P A=-I. The verified
[SciPy 1.15.3 API](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.linalg.solve_continuous_lyapunov.html)
solves aX+Xa.H=q, so the call uses a=A.T and q=-I. The candidate's symmetry
defect is recorded before explicitly choosing its symmetric part. This changes
a free candidate, never the plant. Independently recompute Q=-(A.T P+P A),
check ||Q-I||\_F/||I||\_F, and require resolved positive P and Q. Solver warnings,
invalid output and unresolved candidates yield `not_established`, never an
automatic instability verdict.

For the stated ODE, V=x.T P x satisfies dV/dtau=-x.T Q x. If the declared
dimensionless operator error obeys ||delta A||\_2 <= d, then
||delta A.T P+P delta A||\_2 <= 2||P||\_2 d. Consequently the retained margin is
m=lambda_min(Q)-2||P||\_2 d, and the envelope is

```text
||x(t)||_2 <= sqrt(lambda_max(P)/lambda_min(P))
             * exp[-m*t/(2*lambda_max(P)*T)] * ||x(0)||_2.
```

Require m to exceed the declared definiteness floor times ||Q||\_2, as well as
resolved P/Q eigenvalue ratios. This avoids promoting an unresolved positive
cancellation remainder. The norm depends on coordinate scaling and can admit
transient amplification. It is not mechanical energy. The error bound is
assumed by the caller, not estimated from parameters, experiments or solver
residuals. Zero concerns the computed generator alone. Floating-point matrix
formation and eigenvalue checks are not interval-verified proof.

## Independent Controls and TDD

For A=[[0,1],[-1,-1]], direct scalar equations give
P=[[1.5,.5],[.5,1]]. Critical damping A=[[0,1],[-1,-2]] instead gives
P=[[1.5,.5],[.5,.5]], even though the state eigenbasis is defective.

For M=I, C=2I and circulatory K=[[1,4],[0,1]], the independent trajectory
starting at [0,1,0,0] is q2=exp(-t)(1+t), q1=exp(-t)p(t),
p(t)=-2t^2-2t^3/3, with velocities obtained by differentiation. Its norm
exceeds 1.5 before decay and remains below the reported envelope over the
declared sampled interval. The algebraic bound supplies the all-time result
under the stated assumptions; sampling alone does not.

The initial new test file fails collection because the implementation is absent.
Implementation then passes 82 new/existing decay and spectrum controls in
23.81 s. A further cancellation-margin test fails on a falsely qualified tiny
positive margin; the explicit relative margin gate repairs it. All 39 decay
cases pass in 10.91 s. Controls also cover physical-time conversion,
neutral/undamped/unstable systems, damping-induced gyroscopic instability,
semidefinite damping with/without coupling, invalid solver output, immutable
results, strict input domains, singular mass and overflow.

Broader Linux golf/API regression passes 731 tests in 284.12 s, with two optional
CAD skips and three unavailable-plugin configuration warnings. The same Python
3.11.15, SciPy 1.15.3 and single BLAS/OMP/MKL thread configuration is retained.
Hook-style mypy 1.13.0 passes both modified source modules, and scoped Ruff
0.14.10 lint/format passes. The private API baseline retains empty exports.
All nine final manual/inventory/handoff gates and repository Ruff 0.14.10 pass
(3,759 formatted Python files). These remain structural, unapproved gates.
The old inventory classifier labels this numerical module non-calculation;
this is a known false negative, not accepted scientific categorization. Apply
the AST classification correction in #5103 before combined delivery, retaining
provisional calculation authority and publication blockers. Normal delivery
hooks follow.
Keep the full T3 and impact/acoustic program open. Next: attach qualification
to explicitly selected operating models, time-varying/transient response,
validated modal bandwidth, nonlinear contact and physical acoustic evidence.
