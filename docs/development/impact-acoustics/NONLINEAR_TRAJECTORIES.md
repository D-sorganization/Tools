# Inertial moving-grip trajectories — Tools #5072

This private candidate integrates the nonlinear acceleration and two-port energy
assembly in NONLINEAR_MOVING_CHAIN.md. It retains fixed shaft, head, grip and
spatial applied-load laws. An explicit callback supplies each prescribed anchor's
pose, body twist and body-twist derivative at every evaluation. The callback's
smoothness, determinism and kinematic consistency are assumptions, not facts
established by sampling. Time-varying material or grip coefficients would require
additional storage/work terms and are outside this interface.

## Equations and time method

For every node, H maps its material axes to the declared inertial observer,
V is its linear-first physical body twist, and a is its body-twist derivative:

    H_dot = H hat(V),       V_dot = a(t,H,V).

The acceleration reuses the existing finite-geometry section/head/grip assembly.
No node is clamped. At each step of size h, explicit Lie midpoint uses:

    H_m = H_n Exp((h/2) V_n)
    V_m = V_n + (h/2) a_n
    a_m = a(t_n+h/2, H_m, V_m)
    H_(n+1) = H_n Exp(h V_m)
    V_(n+1) = V_n + h a_m.

The implementation uses the representable cell midpoint and its actual elapsed
stage time. It refuses collapsed endpoints/midpoints before physics evaluation.
Each local exponential retains the existing principal-chart rotation guard.
Proper rotations are checked after multiplication; no orthogonal projection,
velocity rescaling, clipping or energy correction is performed.

For smooth finite solutions, expansion of the pose update gives
H_n[I+h hat(V_n)+h²(hat(a_n)+hat(V_n)²)/2]+O(h³). Expanding the midpoint
acceleration likewise gives the correct velocity Taylor terms through h².
This establishes local third-order truncation and global second-order accuracy
under the usual smoothness/resolution assumptions. It does not make an explicit
method stable for arbitrary stiff modes or establish convergence at a chosen h.

Iserles, Munthe-Kaas, Nørsett and Zanna, [_Lie-group methods_, Acta Numerica 9
(2000), section 3](https://sites.math.rutgers.edu/courses/549/Lie_methods.pdf),
explain exponential-coordinate integration and the distinction between group
preservation and integration order. The local Taylor argument above is specific
to this implemented second-order scheme. Higher-order commutator methods,
symplectic methods and exact energy preservation are not implemented here.

## Work and failure contracts

Midpoint quadrature separately accumulates applied work, work delivered by the
grip to its prescribed anchor, and nonnegative dissipated energy. At every saved
endpoint the unaltered diagnostic is:

    energy_error = E(t)-E(0)-W_applied+W_anchor+D.

E includes shaft kinetic energy, elastic shaft energy, and relative grip elastic
and inertial storage. External load work is not also counted as potential energy.
Positive anchor work is energy leaving this shaft/grip system for the driver.
The diagnostic's convergence must be checked independently of state convergence.

Exactly 2N+1 acceleration and history evaluations produce N+1 endpoint samples.
Strict positive integer controls reject booleans and coercion. Insufficient
budget is refused before history calls. A missing/invalid anchor, strain-domain
exit at a stage, unresolved mass solve, chart violation or numerical failure
raises without returning a partial trajectory. Strains are checked only at
stages/endpoints; this is not a continuous-time enclosure or event detector.
The result remains `time-discrete-unqualified`, with stability `unqualified`.

## Reuse and verification in progress

Missing-module RED preceded implementation. The independent axial reference is
a two-mass system with consistent rod mass, tip mass, root inertance, a Kelvin
Voigt grip, constant tip force and quadratic prescribed base displacement.
A seven-state augmented matrix exponential includes the polynomial input exactly;
independent quadrature supplies each reference work port.

The initial 8/16/32-step grid converged in state/work, but the signed energy
defect crossed zero on coarse steps. The grid was refined to 32/64/128 with
unchanged interval, material parameters and error criteria. State errors are
1.37031e-4, 3.40835e-5 and 8.49940e-6 in the declared scaled norm. Work errors
are 1.44222e-6, 3.59847e-7 and 8.98686e-8 J; energy-balance errors are
-1.32313e-6, -4.01596e-7 and -1.09238e-7 J. All 23 initial trajectory controls
pass in 30.46 s before the work-only optimization.

A nonplanar, moving screw-anchor reference integrates spatial translations and
quaternions with independently coded kinematic equations and DOP853, reusing
the physical acceleration. Two tighter reference tolerances are compared.
The 8/16/32-step candidate converges at second order but its 32-step error is
0.0178002, above the predeclared 1e-4 target. Refinement to 128/256/512 steps
passes the same accuracy/rate criteria. The first refined Windows run reaches
the unchanged 60 s test limit. Limiting BLAS/OMP/MKL to one thread passes all
five geometry/history controls in 34.14 s (29.19 s for the demanding study).
Neither that time limit, accuracy criterion nor physical interval is relaxed.

That cost exposed redundant tangent assembly during each acceleration call.
New work-only section/chain pathways share the constitutive and force assembly
with the full-tangent pathway. A monkeypatched curvature refusal first fails,
then exact force/energy/rate parity passes without evaluating unused curvature.
The full-tangent pathway retains its fields and curvature. All 105 work,
section/chain/load and instantaneous moving controls pass in 15.89 s. Actual
mypy passes nine changed Python files. The final optimized axial/API run passes
32 Windows tests in 16.12 s. Full Linux golf/signal/API validation passes 1,018
tests with two optional CAD skips and three absent-plugin warnings in 122.27 s.
Repository Ruff passes 3,824 formatted files and all nine final governance gates
pass. Inventory remains provisional and existing manual approvals remain absent.
Normal publication is the remaining delivery step for this checkpoint.

The native run uses Python 3.11.15, NumPy 2.3.5 and SciPy 1.15.3, with one BLAS,
OMP and MKL thread. Its exact src/tests/conftest/pyproject archive comes from
tree a4027d37b2b3b92b3f6a25deac860503c705ff5d, contains 60,815,360 bytes and has
SHA256 a377fcf07d90bf04ac076acc6bfe9fc70266911f41aab74c68416d4a120752bf.
The retained native log is
`/home/dieterolson/.cache/codex-impact/trajectory-a4027d37b/scientific.log`.

## Remaining program

Complete this trajectory qualification, joint time/mesh/rotating convergence,
continuous full/reduced port errors and versioned physical parameter provenance.
Resolve flexible contact and events, acoustic radiation/calibration, exact-pin
consumer integration and physical/blinded experiments before equipment,
sound-quality or final AffineDrift causal conclusions. A numerical shaft history
alone does not establish why one player's hit sounds sweeter.
