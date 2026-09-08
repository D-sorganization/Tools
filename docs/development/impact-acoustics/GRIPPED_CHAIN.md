# Grip-Supported Loaded Shaft Balance and Operators

## Scope and Existing Capability Reuse

This private composition connects the finite law in `FINITE_GRIP_RESPONSE.md`
to the existing elastic or rotating section chain. A grip can act at any node;
multiple grips at the same node add separate ports. No node is implicitly
clamped. Anchors are stationary in their declared common observer for this
operating-point slice. A nonzero anchor twist/rate is refused, not discarded.
A rotating shaft additionally requires exact observer identity agreement.

Existing distributed inertia and attached rigid head/body samples remain in
`RotatingSectionChain`. The old clamped solver keeps its public/private-call
contract and exact root pose. Both solvers now use one bounded Newton and
backtracking iteration through a minimal work/strain interface. Material
strain limits, rotation chart guards, per-node step caps, force/moment scaling,
iteration budgets and failure refusal remain unchanged.

The new all-node solver returns separate physical grip reactions and combined
shaft/grip elastic storage. A force-balance candidate establishes neither
uniqueness nor stability. Multiple independent grip ports are not identified
physiological coupling between two hands; pressure distribution and a measured
frequency-valid model remain separate inputs.

## Finite Preload Tangent

At relative rest, the grip effort is g=Kq. With anchor prescribed and root
material increment eta, q'=Ar eta and root left-side force is r=Ar^T g.
The full material derivative is

```math
D r[\eta]=(D A_r[\eta])^Tg+A_r^TKA_r\eta.
```

The first term carries preload geometry. For Q=Ra^T Rr and the inverse SO(3)
right Jacobian Jinv, Ar=diag(Q,Jinv). Only the angular part of eta changes Ar:

```math
D Q=Q[\eta_\omega]_\times,\quad
D\phi=J_{\rm inv}\eta_\omega,\quad
D J_{\rm inv}=-J_{\rm inv}(D J_r)J_{\rm inv}.
```

The existing analytic matrix-exponential Frechet derivative evaluates D Jr;
production code adds no finite-difference tangent or small-angle dead zone.
The moving material derivative is converted to the chain's fixed-chart
convention by subtracting the existing connection whose action is
ad(eta)^T r/2. This produces the conservative grip potential Hessian without
imposing numerical symmetry. Rotating and nonconservative load terms retain
their existing nonsymmetric behavior.

## Full-Node Dynamic Snapshot

At the declared relative-rest point, grip inertance and damping contribute
Ar^T M Ar and Ar^T C Ar at their attached nodal blocks. They are added to the
existing distributed/rigid-body model, with all nodes retained:

```math
r+M_{\rm total}\delta\ddot x+(G_{\rm shaft}+C_{\rm grip})\delta\dot x
+K_{\rm total}\delta x=0.
```

The result stores gyroscopic transport and damping separately, as well as the
exact prescribed frame sample. Missing distributed inertia is a TypeError;
a static chain is never silently converted into a zero-mass dynamic model.
No mass regularization, pseudoinverse or stiffness symmetrization is introduced.
The grip's relative-coordinate inertance is still not an absolute hand mass.
Nonlinear velocity-dependent terms and time-varying anchors require the full
trajectory equations; these arrays are a frozen linearization only.

## Independent Numerical Controls

A 1 m synthetic axial rod with EA=300 N under 3 N has 0.01 m shaft extension.
A root spring of stiffness k independently displaces by 3/k m. Tests at
k=100, 1000 and 10000 N/m verify both compliances and the energy
9/(2 EA)+9/(2 k). Pure torsion similarly verifies the sum of grip rotation
and shaft twist. Interior-node and repeated-node grips verify unconstrained
upstream motion and additive support reactions. A nonplanar load closes total
world force and moment independently of local material residual bookkeeping.

An offset finite-orientation anchor exercises the nonzero preload derivative.
Centered physical pose perturbations verify the material tangent, while the
conservative fixed-chart tangent agrees with its transpose without repair.
A separate coordinate-rate energy oracle verifies added grip inertance and
damping, preserving the shaft gyroscopic matrix exactly. Moving anchors,
observer mismatch, invalid nodes, missing inertia and unbalanced free support
have explicit refusal controls.

The rotating rod control uses L=1 m, EA=1000 N, mass/length=0.2 kg/m,
angular speed 10 rad/s and root stiffness 400 N/m. Its independent consistent
axial mass matrix is rho L [[2,1],[1,2]]/6. Solving
(Krod-omega² Mrod+diag(k,0)) z=[-EA,EA] gives positions
[0.02649943, 1.03347761] m, support force 10.59977034 N and axial strain
0.00697818. The coupled solver matches that result. This verifies rotating
force balance and support compliance; it is not a golfer parameter estimate
or a general conclusion about transverse dynamic stiffening.

## TDD, Compatibility and Remaining Work

Initial chain tests are RED because the new composition is absent (5.39 s).
All 28 first coupled/clamped tests pass (12.34 s). The separate dynamic tests
are initially RED (module absent, 6.80 s); their first combined run passes
36 tests (17.10 s). Final coupled/dynamic/clamped/finite-grip controls pass
38 tests (19.75 s). Actual five-module hook mypy passes after explicit casts
at the validated, skipped-import delegation boundary; no arithmetic changes
were needed. All production functions remain within 50 lines.

The nine API tests pass (5.40 s), with only four empty private-module entries
added. Repository Ruff 0.14.10 passes (3,751 files). The tracked inventory is
regenerated; all nine manual gates pass before edits. Complete Linux golf/API
regression passes 646 tests (187.68 s), with two optional CAD skips and three
unavailable-plugin warnings. One BLAS thread and the unchanged 60-second
per-test deadline are retained. All nine final manual gates pass. Published at eeea63b47 with all normal hooks passing.

Required next steps are driven/frozen stability qualification, full grip-
supported frequency response and bandwidth/mesh/modal convergence, then
nonlinear time evolution with complete anchor/frame work and time convergence.
Contact/impact, flexible-head response, radiation/identified acoustic transfer,
physical validation and blinded sweetness remain open. Inventory #5103 remains
unmerged because its private consumer lookup fails before tests; integrate the
protected classifier and regenerate inventory before final combined delivery.
