# Finite-Grip Shaft Spectra

## Ownership and Scope

Tools #5072, parent #5068. This follows the published finite-support response
`12bcf3d83` and turnover `37af35dcc`. The new private damped pencil and all-node
spectrum retain the existing shaft, finite-grip and loaded-equilibrium models.
They do not introduce a hand calibration, acoustic model or hidden clamp.
`GRIPPED_SPECTRUM_PLAN.md` records the independent benchmark derivation.

## Equations and Shared Implementation

Use M q''+(G+C)q'+Kq=0. M is the retained inertia; G is gyroscopic transport;
C is the relative-grip loss coefficient; K includes the loaded material
derivative. The new `DampedPencil` owns finite real equal-size coefficient
arrays as immutable tuples. It does not combine G and C in its stored record.

Evaluation requires positive, numerically resolved mass, skew G and symmetric
positive-semidefinite C within the explicitly declared numerical tolerances.
No matrix is projected, symmetrized or regularized. Tolerance acceptance is
not an exact passivity certificate, physical uncertainty estimate or permission
to omit unresolved negative loss. Nonsymmetric K, growing modes and deficient
eigenbases remain visible. Singular mass is refused by this ODE route even
when the separate frequency-response route can solve an algebraic pencil.

For coordinate map q=S y and t=T tau, the first-order generator is

    A = [[0, I], [-T^2 Ms^-1 Ks, -T Ms^-1 (Gs+Cs)]],
    Ms=S^T M S, Gs=S^T G S, Cs=S^T C S, Ks=S^T K S.

S scales each translation by the declared length and leaves rotation in
radians. Physical rates are generator eigenvalues divided by T. Returned
displacement and velocity modes are mapped to node material coordinates;
their complex normalization is arbitrary. The all-node entry recomputes every
force/moment balance and the section strain domain before solving. Individual
equilibrium grip wrenches and the observer frame are retained; no single
clamped-root support wrench is invented.

The legacy skew-G-only entry retains its input contract. Both paths share the
positive-mass generator kernel and eigenpair diagnostics. State residuals
and original quadratic-polynomial residuals must pass. For each mode x and
physical rate s, the latter uses

    r = (s^2 M+s G+s C+K)x,
    denominator = (|s|^2 ||M|| + |s| ||G|| + |s| ||C|| + ||K||)||x||.

Coefficient norms are kept separate. Eigenbasis reciprocal condition is
reported; a small backward residual does not guarantee a small forward error
near clustered or defective roots. No automatic stable-swing status is assigned.

## Independent Numerical Controls

Scalar damped roots are compared with the independent characteristic
polynomial. Critical damping retains its ill-conditioned eigenbasis. Negative
stiffness keeps a growing root despite positive damping. The dimensionless
M=I, K=-I, G=[[0,-3],[3,0]], C=0.1 I example has characteristic polynomial
s^4+0.2 s^3+7.01 s^2-0.2 s+1 and two growing roots; adding passive loss does
not justify automatic stabilization. Semidefinite damping retains an undamped
mode, and circulatory stiffness is not replaced with its symmetric part.

The independent two-node axial rod has M=[[a,b],[b,d]], root-only damping c,
and K=[[k11,k12],[k12,k22]]. Its scalar determinant coefficients are

    [ad-b^2, cd, a k22+d k11-2b k12, c k22, k11 k22-k12^2].

The synthetic rod uses the same SI inputs as `GRIPPED_FREQUENCY_RESPONSE.md`:
EA=1000 N, mu=0.2 kg/m, L=1 m, mg=0.03 kg, mt=0.1 kg, kg=400 N/m,
c=2 N s/m. Its four axial roots are compared as an unordered multiset;
axial modes are selected by axial displacement participation, not by sorting
all bending/torsional frequencies into an assumed axial order.

For the distributed rod, kappa=s sqrt(mu/EA), dg=kg+c s+mg s^2. The boundary
determinant is

    EA kappa sinh(kappa L)+dg cosh(kappa L)
    +mt s^2 [cosh(kappa L)+dg sinh(kappa L)/(EA kappa)].

Solve its real and imaginary parts independently of the finite-element model,
check solver success and residual, and compare 2/4/8-element poles. The two
positive-frequency reference poles are approximately
-2.221586591+32.220241752i and -6.741190153+153.557574487i per second.
Both frequency and decay converge; these are numerical controls, not golf
measurements or an identified useful impact-frequency band.

## TDD and Delivery Evidence

Initial collection fails for both missing modules (2 errors, 6.09 s). The first
implementation passes all 49 new/legacy spectral controls in 20.22 s. Additional
checks cover immutable input ownership, time/length scaling, rotating preload,
separate equilibrium reactions, balance and strain refusal, malformed matrices,
indefinite/asymmetric coefficients, underflow/overflow and inaccurate eigenpairs.
Three-module hook-style mypy and scoped Ruff pass after an explicit return-type
cast in the shared kernel. No checker suppression or tolerance change is used.
Full Linux golf/API regression passes 692 tests in 161.71 s, with two optional CAD skips and three unavailable-plugin configuration warnings. Python 3.11.15, one BLAS/OMP/MKL thread and the unchanged 60-second test limit are retained. All nine final manual gates and repository Ruff 0.14.10 pass (3,757 formatted files); all modified functions remain at most 50 lines. Normal publication is pending.

## Completion Boundary

These remain frozen spectra. Stability of an autonomous model requires an
explicit autonomy/regularity declaration and the appropriate energy or spectral
argument; a driven/time-varying swing requires its own evolution analysis.
Positive M/K and PSD C yield a Lyapunov-energy argument only with the stated
constant/skew assumptions; asymptotic decay also needs the invariant-zero-loss
condition. AffineDrift #4295 / PR #4298 explains the counterexamples.

Next work is explicit stability assessment, transient/modal and bandwidth
qualification, nonlinear evolution with anchor/frame work, time convergence,
contact and flexible-head radiation. Physical/blinded acoustic validation
remains separate. UpstreamDrift #9825 tracks the actual claim-reconciliation
preservation bug on protected main; #8920/#8556 remain bilateral-wrench and
parameter-identification gates. Tools inventory #5103 must be integrated before
final combined delivery, without manufacturing scientific approval.
