# Time, mesh and corotation qualification — Tools #5072

These tests extend the published nonlinear trajectory at
e6b616308e8469ce71797b6e3aefa02cbe5e269a. Production implementation and public
APIs are unchanged. The new tests qualify specified synthetic subproblems of
the existing full shaft model; they do not close T3 or establish physical
impact, sound quality, arbitrary rotating disturbances or unconditional stability.

Machine-readable results from both platforms, test identities, runtime versions
and source/archive hashes are in [JOINT_TRAJECTORY_RESULTS.json](JOINT_TRAJECTORY_RESULTS.json).
The time integrator, work signs, input contracts and numerical-method references
are derived in [NONLINEAR_TRAJECTORIES.md](NONLINEAR_TRAJECTORIES.md).

## Independent axial continuum

The existing synthetic fixture has length L=1 m, axial rigidity EA=1000 N,
mass per reference length mu=0.2 kg/m, a 0.1 kg tip mass, a 400 N/m root
spring, and 0.03 kg relative root inertance. Damping and applied loads are zero
for this study; the anchor is stationary. All transverse/rotational nodes and
their original properties remain in the numerical model. Pure axial motion is
an invariant subproblem; the tests check that it remains so without clamping.

For displacement u along the reference coordinate x, continuum balance and
dynamic boundary conditions are:

    mu*u_tt = EA*u_xx
    EA*u_x(0) = K_root*u(0) + B_root*u_tt(0)
    EA*u_x(L) = -m_tip*u_tt(L).

Integration by parts gives the conserved energy

    E = integral_0^L [mu*u_t² + EA*u_x²]/2 dx
        + K_root*u(0)²/2 + B_root*u_t(0)²/2 + m_tip*u_t(L)²/2.

No boundary storage is omitted. For u=A*phi(x)*cos(omega*t), let
k=omega*sqrt(mu/EA) and

    phi(x) = cos(k*x) + (K_root-B_root*omega²)/(EA*k)*sin(k*x).

The remaining tip condition determines omega. A bracketed root in [5,60] rad/s
has omega=32.241776665090526 rad/s and characteristic residual below 1e-10 N.
The amplitude is A=0.001 m. The exact continuum energy is
2.34646190138e-4 J, independently verified at several phases of the motion.

An independent scalar FEM assembly uses consistent rod mass, axial stiffness,
root spring/inertance and tip mass. Its first-order matrix exponential supplies
the exact semi-discrete reference, starting from the sampled continuum mode.
Production matrices are not borrowed for this reference. Comparing that result
with the full nonlinear time integrator isolates temporal error; comparing the
semi-discrete reference with the continuum isolates spatial error.

The state error is the maximum over all nodal displacement errors divided by A
and velocity errors divided by A\*omega. This norm does not grow merely because
a mesh contains more nodes. Results at t=0.01 s on Linux are:

| Elements | Steps | Spatial error | Temporal error | Combined error | Energy-balance error (J) |
| -------- | ----- | ------------- | -------------- | -------------- | ------------------------ |
| 2        | 32    | 1.22728e-3    | 7.08413e-6     | 1.23398e-3     | 1.93988e-11              |
| 4        | 64    | 2.94654e-4    | 1.67258e-6     | 2.95880e-4     | 2.41976e-12              |
| 8        | 128   | 7.69756e-5    | 4.53833e-7     | 7.73897e-5     | 3.02501e-13              |

Temporal error remains below 10% of spatial error at each resolution. Both
spatial and combined errors decrease at second order. A separate fixed two-
element time refinement gives errors 2.84213e-5, 7.08413e-6 and 1.76805e-6
for 16, 32 and 64 steps. The initial discrete energy also converges to the
continuum value at second order.

The positive numerical energy drift is retained and checked against independent
discrete energy. For a conservative linear oscillator explicit midpoint can
show third-order energy drift alongside second-order state error. It is not an
energy-preserving method, and longer undamped ringdown requires its own error
control. This observation matters before interpreting a computed acoustic decay.

## Loaded rotating continuum and finite history

For a radial rod spinning at constant Omega=3 rad/s, write w=x+u. Its static
rotating continuum balance and finite-root boundaries are:

    w_xx + (mu*Omega²/EA)*w = 0
    EA*(w_x(0)-1) = K_root*w(0)
    EA*(w_x(L)-1) = m_tip*Omega²*w(L).

Solving the two coefficient equations for w=a*cos(kappa*x)+b*sin(kappa*x)
provides an independent continuum reference. The synthetic head has zero COM
offset and a principal inertia axis along the spin axis. These assumptions are
essential to the radial reference, although the production inertia remains full.
At 2, 4 and 8 elements, maximum nodal position errors are 1.07722e-7,
2.71437e-8 and 6.79924e-9 m, respectively. The finite root moves and the rod
extends. This is not a claim about the sign of every modal-frequency shift.

The loaded discrete state is then propagated in an inertial observer with the
exact rotating anchor history. The reference is rigid corotation of each loaded
pose; its physical body twist is constant. Over 0.002 s, the 2/4/8-element
models with 8/16/32 steps preserve reference positions within 2e-10 m and
dimensionless rotation-matrix components within 2e-10. Body twists agree within
2e-8 m/s or rad/s.
Applied work and dissipation remain zero, anchor work stays below 1e-12 J,
and energy-balance error stays below 1e-10 J.

Relative grip coordinates are constant in exact corotation. Thus the ideal
relative inertance has zero stored energy despite nonzero absolute root motion.
Adding it as an absolute mass rotating about ground would change the model.
A physical hand model may also contain absolute inertia, which must be specified
and identified separately; this test does not identify a real player's hand.

## Evidence and remaining scope

All nine final qualification cases pass on Windows in 43.31 s and Linux in
29.77 s, using one BLAS/OMP/MKL thread and unchanged 60 s per-test limits.
Linux reports three unavailable-plugin configuration warnings and no skips.
Actual mypy passes all three new Python files. Repository Ruff passes
3,827 formatted files, as do all nine final governance gates. Existing manual
approvals remain unapproved. Normal publication remains for this checkpoint. Existing production code already
satisfied these new checks, so no artificial implementation change or claimed
RED-to-GREEN repair was needed. The earlier 1,018-test broad Linux result remains
tied to the published implementation's earlier archive; it was not rerun or
relabelled as this new qualification run.

The new exact scientific tree is eda0b3efbaee46cf4ebdc57e6e67b131d31e6cdc;
its archive has 60,835,840 bytes and SHA256
f7c92000a51e3bd94a79b3130b47a2b101f849aeca976f18a0de08679daf95fd.
Raw Linux evidence is retained in
`/home/dieterolson/.cache/codex-impact/joint-eda0b3efb/qualification.xml`
and `qualification.log`. Windows XML is retained at
`C:/Users/diete/AppData/Local/Temp/impact-joint-windows.xml`.

Disturbed rotating/bending continuum convergence, full/reduced continuous-band
port errors, identified physical parameters, flexible contact/event convergence,
calibrated radiation, exact-pin consumer studies and physical/blinded final
AffineDrift synthesis remain required. These qualified synthetic cases cannot
by themselves explain differences in players' impact sound.
