# Continuous first contact and sliding entry

Review #5160 / draft PR #5162 continues parent #5073. This is test-only
numerical evidence for one initially separated planar, positive-slip case.
It neither calibrates the impact law nor predicts radiated sound or sweetness.
The canonical mechanics, normal law and local Lie-chart differential are
shared; the continuous phase/history equations and adaptive time method do
not use the production endpoint solve or discrete tangential return map.

## Phase equations and boundary values

The case starts with 0.1 mm clearance, zero elastic tangential history,
5 m/s tangential ball velocity, -0.4 m/s normal ball velocity, mu=0.02 and
k_t=1000 N/m. Before first touch, the ball is unforced and the shaft follows
its prescribed grip/load problem. An existing independent three-mass matrix
exponential supplies the first-touch time; that comparison uses no friction
or contact feedback before entry.

The normal port defines F_n=0 at nonpositive compression. Its damped law has
a nonzero incoming limit at a closing first touch: F_n(0+)=c_n delta_dot.
The phase reference uses that right-hand limit at the resolved boundary,
with zero elastic normal storage and viscous power c_n delta_dot squared.
It refuses application of this limit in unresolved positive clearance.
The instantaneous force jump does not create a momentum impulse or a finite
energy jump. State and accumulated work pass continuously to the active phase.

In this planar positive-slip branch, write v_t=s e, where e rotates with
the face and s>0. Elastic loading integrates a scalar q with q_dot=s and
z=q e, giving F_t=-k_t q e and zero plastic dissipation. The phase terminates
at k_t q=mu F_n. The saturated continuation uses q=mu F_n/k_t and
lambda=s-mu F_n_dot/k_t>0, as derived in FRICTION_TRAJECTORY.md. The event
checks continuity of history and force before changing representations;
no finite storage correction is added to the work ledger.

The shared `SlidingReference.respond` applies each reference's chosen force
through the existing common-point shaft/ball response. This extraction avoids
duplicated mechanics without sharing the production history update. DOP853
locates terminal first-touch and elastic-boundary events, then continues the
existing sliding reference through force cutoff and geometric separation.
Each entry root has an explicit SI residual check; event records are owned
and read-only. Phase extensions inside root location are not physical states
accepted past the phase boundary. Sign-change detection still does not prove
completeness for arbitrary grazing, repeated contact or reversal.

## Reference results and independent checks

| Event                | Time (ms)      |
| -------------------- | -------------- |
| First touch          | 0.367303647791 |
| First sliding        | 0.370291056692 |
| Raw-force cutoff     | 2.585013738545 |
| Geometric separation | 2.728275693654 |

Incoming normal force is 0.7337854620514 N. The elastic loading interval lasts
about 2.9874 microseconds. The first-touch time matches the independent
free three-mass solution within the preset 1e-10 s bound. History/force are
continuous at sliding entry within 1e-10 m / 1e-7 N; plastic work is zero
before entry. Full continuous work closes with a defect of about -5.54e-12 J,
inside the preset 1e-8 J bound. Tenfold tighter ODE tolerances preserve all
four event times to 1e-9 s and every reported endpoint channel to 1e-8 SI.

The continuous reference resolves this short elastic phase. The production
30/60/120 and 60/120/240 grids have steps much longer than that interval.
Endpoint convergence must therefore remain distinct from resolving the
complete event history, normal-force jump or force peaks.

## Retained production resolution findings

The original 30/60/120 comparison fails its preset finest normal impulse,
tangential impulse and plastic-work bounds at120 steps: errors are
1.3846851e-4 N s, 2.7474490e-6 N s and 1.3910565e-5 J. The limits remain
1e-4 N s, 2e-6 N s and 1e-5 J. Refining to60/120/240 brings all six separate
velocity/spin/impulse/plastic-work channels within their original bounds,
and their successive error ratios satisfy1.5 to2.5. The other work channels
meet their original1e-4 J absolute limits. No production source, solver
budget, residual tolerance or test deadline changes.

A provisional assertion that every work-channel error must decrease at
every halving fails for cutoff loss: absolute errors at60/120/240 steps are
3.28347e-6 / 2.21414e-7 / 7.24041e-7 J. This failed result is retained.
Source inspection shows endpoint quadrature of cutoff power; an analytic
counterexample below demonstrates why monotonic absolute error is not a
necessary property of that quadrature. It does not prove that quadrature
alone explains the coupled solver's error; state and event-time errors also
contribute. The invalid monotonicity requirement is replaced by the existing
absolute-accuracy check for cutoff loss, while monotonic checks remain on the
four other work ports. The cutoff absolute allowance exceeds the approximately
9.17e-6 J reference cutoff energy in this case, so passing it does not establish
useful relative cutoff accuracy. This is a correction to a newly authored test
assumption, not a claim of complete cutoff-work convergence. Event-resolved
work qualification remains open under #5160/#5073.

## Why a discontinuous pulse need not refine monotonically

Consider the independent 1 W triangular power pulse
P(t)=(b-t)/(b-a) W for a<t<b and zero otherwise, with
a=0.002585 s and b=0.002728 s. Its exact integral is (b-a)/2 J.
The right-rectangle approximation on [0,0.003] has signed errors
-2.79371e-6 J at120 steps and +3.76224e-6 J at240 steps. A further
480-to960 halving also increases absolute error. These values are an
analytic quadrature example, not fitted club powers or an energy correction.

For a function of bounded variation on a partition of maximum width h,

```text
|sum_i h_i P(t_i) - integral P(t) dt|
<= sum_i integral_cell |P(t_i)-P(t)| dt
<= h sum_i TV(P; cell_i)
<= h TV(P).
```

The pulse has total variation2 W, so its quadrature error is bounded by
2 h J/s and tends to zero despite nonmonotonic individual errors. The control
checks that bound and both error reversals. This bound concerns quadrature
of the prescribed pulse only; it is not a certified total-error bound for
the coupled shaft/ball trajectory. A small nonlinear residual, a passing
mixed norm or three favorable grid values cannot supply that missing proof.

## Test workload and retained timeout

The first combined three-grid test passes its numerical core bounds in the
focused run (53.914 s test body), then hits the unchanged60-second deadline
inside the broader regression. That interrupted regression has no completed
JUnit receipt and is not a passing qualification. The test now compares
60-to120 and120-to240 steps in separate parameterized cases, sharing a live
read-only120-step result through a module fixture. Each pair still checks the
original velocity/spin/impulse/plastic-work ratios, valid work requirements,
energy defect and algorithmic loss; full scaled-state refinement is also
checked. The240-step case retains every original finest-channel absolute
bound. No grid, production calculation or deadline is removed or relaxed.

## Continuation and qualification boundary

Exact source/JUnit identities, the missing-module RED, both retained
production comparison failures, typing checks and final regression status
belong in FRICTION_ENTRY_RESULTS.json. General stick/slide reversal, nonplanar
transport, event-resolved work, mesh/mode refinement, measured contact/FRF
validation, radiation and blinded perception remain requirements. Do not
promote these synthetic cases into a claim that grip, prestress or a player
produces a heavier or sweeter physical hit.

At SELF, all282 affected Windows tests pass in252.01s after the corrected
cutoff criterion and adjacent-pair test organization. All three modified/new
helper/test files pass NumPy-aware typing, Ruff and structural constraints.
Hosted qualification remains separate and pending.
