# Finite-Grip Frequency Response

## Ownership and Status

Tools #5072, parent #5068. Branch `feat/5072-prestressed-shaft` extends the
published all-node balance/operators at `eeea63b47`. Lease codex /
impact-acoustics-01a07d8a-t3 expires 2026-09-09T00:13:36.131825Z.
This is private numerical verification, not a qualified golfer impedance,
impact band, stable swing trajectory or measured acoustic result.

## Model and Coordinates

The stationary-relative grip composition retains every shaft node. Before
response evaluation, recompute the full force/moment residual and the existing
section strain limits. Root balance is required; there is no implicit clamp.
Anchors remain prescribed in the common declared observer. Observer motion is
returned explicitly and can be driven.

For the positive-frequency convention exp(i omega t), form

    D(omega) = K - omega^2 M + i omega G + i omega C.

Keep the four coefficients separate when assessing cancellation. K includes
the finite-grip preload derivative; G is shaft transport and C is the
stationary relative-grip damping. Singular mass is permitted for an algebraic
particular solution, without an ODE/descriptor regularity claim.

At the material tip offset r, L maps nodal motion to point motion:
translation becomes v + angular_velocity cross r. Its transpose maps point
force/couple to nodal wrench. H=L D^-1 L^T and Y=i omega H, with force/torque
input columns and translation/rotation output rows. No pseudoinverse is used.
Translations are scaled by the declared length before solving, then mapped
back to SI. Reciprocal condition, original-coefficient resolution and direct
residual limits all must pass. Overflow/nonfinite results are refused.

Each attachment retains its own incremental grip-on-shaft material wrench:

    delta_w_g = -(K_g - omega^2 M_g + i omega C_g) delta_x_node.

Repeated or interior nodes are allowed; equilibrium reactions remain separate
from these harmonic transfers. These are node-frame increments, including
the preload derivative, not an unqualified derivative in fixed world axes.

## Independent Axial Reference

For a uniform rod, EA u'' + mu omega^2 u=0. Let kappa=omega sqrt(mu/EA),
length L, tip mass mt, and root dynamic stiffness
dg=kg+i omega cg-omega^2 mg. Boundary conditions are
EA u'(0)=dg u(0) and EA u'(L)-mt omega^2 u(L)=F.
Writing u=A cos(kappa x)+B sin(kappa x) gives B=dg A/(EA kappa).
Therefore define

    shape = cos(kappa L) + dg sin(kappa L)/(EA kappa)
    denominator = -EA kappa sin(kappa L) + dg cos(kappa L)
                  - mt omega^2 shape
    u(L)/F = shape/denominator
    u(0)/F = 1/denominator.

The static tip compliance is 1/kg+L/EA; root reaction is -dg u(0).
The independent one-element reference is the two-node pencil with
Krod=(EA/L)[[1,-1],[-1,1]], Mrod=(mu L/6)[[2,1],[1,2]], plus root
kg/cg/mg and tip mt. Synthetic tests use L=1 m, EA=1000 N, mu=0.2 kg/m,
mt=0.1 kg, kg=400 N/m, cg=0 or 2 N s/m, and mg=0.03 kg. These numbers
do not identify a player's hand, shaft material or useful golf bandwidth.

For a stationary conservative rod with passive grips, cycle-average supplied
power equals loss: Re(F^H v_tip)/2 = omega^2 q^H C q/2. Reciprocity is
checked for the stationary symmetric system, not assumed for a spinning,
nonsymmetric loaded pencil. Mesh checks compare 2/4/8 elements with the
continuum formula at 10/25/50 rad/s, checking second-order convergence.

## TDD and Review Evidence

Initial collection fails because the response module does not exist
(RED: 1 error, 8.03 s). First implementation passes 36 new/clamped response
tests in 49.29 s. Refactoring shares the scaled pencil, numerical solve and
point-motion mapping with the existing clamped entry. The final 50 focused
response, rotating-loaded, coupled-balance and grip-operator tests pass in
60.89 s. All modified functions remain at most 50 lines; Ruff passes.

Controls include static/complex two-node response, continuum mesh convergence,
offset force/torque power duality, cycle loss, reciprocity in its valid case,
repeated interior support reactions, length-scale invariance, fresh arrays,
root imbalance, strain violation, singular supports, coefficient cancellation,
overflow and invalid-control refusal. A rotating loaded case retains nonzero
G, recomputed preload reactions and the physical SI balance. That last check
is an integration check, not an independent continuum oracle.

Full Linux golf/API regression passes 665 tests in 218.62 s, with two
optional CAD skips and three unavailable-plugin configuration warnings. Python
3.11.15, one BLAS/OMP/MKL thread and the unchanged 60-second per-test timeout
are retained. All nine final manual gates, three-module hook-style mypy and
repository Ruff 0.14.10 pass (3,753 formatted files). Published at 12bcf3d83a5b33c14a21c998879b892b84ee57d8; normal commit/push hooks pass and remote SHA is verified. The public facade is unchanged; the new module declares empty
`__all__` and receives only an empty private-module baseline entry.

## Remaining Work

Frozen eigenanalysis must accept G and C separately and check polynomial
residuals/conditioning. A frequency solution does not prove stability;
AffineDrift #4295 gives passive-damping counterexamples. Autonomous versus
driven/time-varying stability, modal and bandwidth qualification, nonlinear
evolution with anchor/frame work and time convergence remain. Impact-point
contact, flexible-head radiation, microphone calibration and blinded sweetness
tests are distinct requirements. Integrate protected inventory PR #5103 and
already merged contact/wire changes before the final combined delivery.
