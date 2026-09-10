# Relative magnitude and phase of reduced shaft responses

Tools #5072, 2026-09-09. This development derivation extends
[REDUCED_TRANSFER_BANDS.md](REDUCED_TRANSFER_BANDS.md). It is conditional
floating numerical evidence for prescribed finite models, not an approved
engineering-manual pathway, measured club model, stability certificate or
acoustic prediction. The two new modules have empty public export lists.

## A response floor is necessary

Use the existing dimensionless displacement transfer H=C D^-1 B with exactly
one input and one output. The full response polynomial and its residual bound
give Hf(c+delta)=a+delta b+e(delta), |delta|<=h, |e|<=eta, where
a=C P0, b=C P1 and eta=norm(C) times the full polynomial remainder bound.
Computed inverse defects and prescribed coefficient uncertainty remain included.

Let s=h b. The minimum of |a+x s| for real x in [-1,1] is obtained by projecting
the origin onto the complex line and clipping to the segment endpoints. For
nonzero s the unscaled minimizer is

    x = clip(-Re(a conjugate(s))/|s|^2, -1, 1).
    f = max(0, |a+x s|-eta).

Implementation normalizes before projection and uses hypot norms, avoiding
squared huge/tiny complex values. Unrepresentable intermediate arithmetic is
refused. This is not outward-rounded interval arithmetic: rounding in matrix
assembly, segment geometry and final bounds remains an explicit limitation.

If f>0, and the existing full/reduced absolute error bound is E, then

    |Hr/Hf-1| <= E/f = r.
    ||Hr|-|Hf||/|Hf| <= r.

For r<1, Hr/Hf lies in a disk centered at 1 that excludes the origin. Tangency
from the origin to that disk gives |Arg(Hr/Hf)|<=asin(r). Consequently both
responses are nonzero under the conditional bound. This is the principal
relative phase, not subtraction of separately wrapped angles or a global
phase unwrap. A MIMO matrix has no single phase in this contract.

At an antiresonance, a finite absolute error can coexist with undefined relative
error and phase. A nonpositive floor therefore refuses qualification. Failure
to establish a floor does not prove that the true response has a zero. Likewise,
a conservative bound exceeding a target does not prove the actual error does.

## Complete bands and contracts

The new assessor reuses the existing exact-endpoint subdivision engine,
full/reduced interval assessment, normalized ports and residual polynomial.
Every accepted cell must meet the explicit absolute, relative and phase targets.
Attempted paired cells count against the original budget, including rejected
parents; exhausted budgets or unresolved cells never return a partial cover.
Relative targets lie strictly in (0,1), phase targets in (0,pi/2) radians.
Full and additional reduced coefficient-error prescriptions remain attached.
No port, basis, material, measurement or radiation uncertainty is inferred.

## Independent checks and retained failures

Missing-module RED tests preceded both implementations. A geometric underflow
regression then failed because Python scalar division silently returned zero;
NumPy checked division now refuses that unresolved input. Independent controls
include complex-plane segment geometry at scales 1e-200 to 1e200, analytic
omitted modes, exact antiresonance, a coupled gyroscopic/circulatory adjugate,
prescribed coefficient perturbations and deliberately corrupted inverses.
Another control has absolute phases straddling +/-pi while its relative phase
is small; subtracting the wrapped angles would give the wrong conclusion.

The original axial fixture retains its undamped transverse-pole refusal. Its
0–8 rad/s band meets absolute 0.05, relative 0.01 and phase 0.01 rad targets
with 2/4/5 axial modes in 31/21/21 attempted cells, respectively.

A **separate synthetic prescription** damps all six grip axes: translational
damping is 2 N s/m and rotational damping is 0.02 N m s/rad. It retains the
existing finite grip stiffness/inertance, shaft and tip assembly. Four elements
give 30 full coordinates. Tip force/displacement or torque/rotation references
are 1 N/0.001 m or 1 Nm/0.001 rad; these normalize transfer units and do not
claim that those finite forcing amplitudes remain in a linear physical domain.
The full 0–40 rad/s band (about 6.37 Hz) must meet absolute 1, relative 0.02 and
phase 0.02 rad targets within 4,095 attempted pairs. This is not an audible or
impact-band qualification.

Sixteen low modes initially fail the absolute target in bending and torsion,
despite small sampled relative errors. Explicit counterexamples at 10.2 and
23.4 rad/s retain this failure. Neither the target nor the band was relaxed.
Tip-only static enrichment was insufficient. In the accepted test fixture,
augment mass-orthonormal modes V with static responses X=K^-1 F for the tip
and relevant grip force/torque columns. Remove V(V.T M X), select independent
residual directions in the M metric, and mass-orthonormalize the combined span
by QR. The test verifies that the selected static responses remain in that span.
This is explicit fixture construction, not a general basis-selection product.

The numerical study supports considering interface motion and omitted static
flexibility when choosing reduction bases. The indexed abstract of NASA's
[Alternate Methods of Model Reduction to Avoid Dynamic Modal Truncation Error](https://ntrs.nasa.gov/citations/20240008223)
discusses related static-residual/interface issues. Only that abstract was
accessible; the full document was not reviewed. No spacecraft frequency-cutoff
heuristic or numerical result is transferred to golf here.

Windows accepted bending uses 19 coordinates (16 modes plus three static
directions), torsion 18 (16 plus two), and axial 16. Their respective maximum
relative bounds are 0.01973494, 0.00599532 and 0.01968981, in 1,437, 2,017 and
243 attempts. Complete 30-coordinate controls pass the same targets. These are
conservative bounds, not measured errors or proof of monotonic convergence;
even a complete basis can require many cells because of bound conservatism.
Between-sample coverage comes from residual bounds; sampled solves are
supplementary comparisons. See SISO_TRANSFER_RESULTS.json for provenance.

## Remaining scientific work

Disturbed rotating/bending time and mesh qualification, identified parameter
sets, flexible contact, head/face radiation, calibrated complex transfer data,
exact-pin consumers and physical/blinded perceptual evidence remain required.
Phase agreement of these mechanical models does not establish acoustic phase,
pleasantness, sweetness, or a causal explanation of player-to-player sound.
