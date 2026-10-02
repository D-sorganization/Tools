# Why the optimizer stops the hands — and what would fix it

> ## ⚠️ Correction (2026-08-27, issue [#4785](https://github.com/D-sorganization/Tools/issues/4785))
>
> **Sections 3–6 of the original version of this document were wrong**, and the
> corrected results are folded in below. The cause was a mis-specified club, not
> a limit of the model.
>
> The historical preset lumped **0.500 kg at the tip** of a 1.10 m shaft. A real driver is
> **0.310 kg with its centre of mass 76% down the shaft**. In a point-mass-at-tip
> model the quantity that matters for the uncoupled wrist row is inertia about the wrist.
> Placing 0.310 kg at the tip of a 1.10 m modeled shaft gives an inertia of
> **0.3751 kg·m²** (0.310 * 1.10² = 0.3751, ~1.30× the real 0.288 kg·m²). The
> previously stated 0.605 kg·m² value belongs to the 0.500 kg preset
> (0.500 * 1.10² = 0.605 kg·m², **2.10×**).
>
> Club properties reported by clubfitters follow the convention of measuring inertia
> about the physical butt of the grip. In contrast, the model wrist origin is the
> joint pivot axis (situated down the grip where the hands grasp), and the physical
> club length (butt to sole, 1.143 m) differs from the modeled second link length
> (wrist pivot to modeled head COM, 1.10 m).
>
> With an inertia-matched club (`me = 0.238 kg`) the _same_ model, optimizer and
> objective produce **49.7 m/s clubhead, 7.26 m/s hand speed, club/arm 3.46** —
> inside the heuristic reference intervals, with **no hand-speed floor imposed**.
> The claim that "the measured 6–9 m/s band is unreachable at any price" was an artifact.
>
> What survives unchanged: the impact-optimality theorem (§1), the fact that hub
> reversal is the release mechanism, and the direction of the club-inertia
> result. What changes: the _severity_ was a parameter error, a moving hub is
> **not** required, and the objective comparison **does** discriminate once the
> club is right.

**Epic:** [#4775](https://github.com/D-sorganization/Tools/issues/4775)
**Package:** `src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/`
**Predecessor:** [`SWING_OBJECTIVE_COMPARISON.md`](SWING_OBJECTIVE_COMPARISON.md) (epic [#4766](https://github.com/D-sorganization/Tools/issues/4766))

## 0. Summary

The objective comparison shipped in epic #4766 produces swings that satisfy the
dynamics defect tolerance predicate (`max_defect < 1e-6`) and are unrealistic. The
optimizer drives the arms hard, reverses the hub torque, and brings the hands to a
standstill at impact — 0.36 m/s where heuristic reference intervals place a skilled
golfer at 6–9 m/s. (Note: "feasible" in this document denotes satisfaction of the
collocation dynamics defect threshold predicate, not verification of every constraint
or a certificate of global reachability/infeasibility).

Three things are now established, each with code and a regression test behind it:

1. **The optimizer is right and the model is wrong.** With a point-mass clubhead the
   energy-optimal hand speed at impact is _identically zero_, for every parameter value.
2. **Distributed club inertia cannot fix it.** For a real driver the same coefficient
   goes _negative_ — the optimum wants the hands moving backward.
3. **Neither can actuation limits alone.** They stop the torque reversing, which is
   correct, but then the club never releases and the impact posture fails the defect
   tolerance predicate.

What remains is a structural limit of the two-link fixed-hub model: **releasing the club
and decelerating the hands are the same act**. Closing that gap needs a moving hub.

---

## 1. The impact-optimality theorem

At impact the club is in line with the arms, so both angular rates drive the clubhead
along the same perpendicular:

```
v_head = (L1 + L2) * omega1  +  L2 * phidot
```

Maximising `v_head` for a fixed kinetic energy `0.5 * qdot^T M qdot` is a linear
objective on a quadratic form. The optimum lies along `qdot* ∝ M^-1 c` with
`c = (L1 + L2, L2)`, and the arm component reduces to

```
omega1*  ∝  L1 * [ I2 - m2 * r2 * (L2 - r2) ]
```

| Club model                                                         | Bracket       | Optimal hands at impact |
| ------------------------------------------------------------------ | ------------- | ----------------------- |
| Point mass at tip (`r2 = L2`, `I2 = 0`) — **as shipped**           | **exactly 0** | stopped                 |
| Real driver (`m2 = 0.31`, `r2 = 0.89`, `L2 = 1.143`, `I2 = 0.043`) | **−0.027**    | moving _backward_       |
| COM at 88% of length, or `I2 ≥ 0.2` (≈5× a driver)                 | positive      | moving forward          |

The zero is algebraic, not numerical: with all of segment 2's mass at the tip, the
club's kinetic energy _is_ `0.5 * me * v_head²`, so any arm motion is energy that never
reached the clubhead. Verified to machine precision across a 27-point parameter sweep in
`tests/test_impact_optimality.py`.

### Assumptions and realizability

The impact-optimality theorem is an instantaneous variational calculation evaluated under two assumptions:
1. **Fixed posture**: the club is aligned with the arms (`phi = 0`, impact pose).
2. **Fixed kinetic energy**: the instantaneous kinetic energy budget `0.5 * qdot^T M qdot` is constant.

It does **not** assert or guarantee dynamic realizability — whether there exists a continuous,
torque-bounded downswing trajectory from the top of the backswing that can actually reach this
state under physiological torque, torque-slew, or joint limits without violating constraints.

**No physically realistic club moves the bracket positive.** This is why the epic does
not pursue distributed club inertia, and it is the single most useful thing the analysis
produced: it rules out the intuitive fix before any of it gets built.

Implementation: [`impact_optimality.py`](../../src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/impact_optimality.py).

---

## 2a. The club correction (#4785)

`physics.mass_matrix` treats segment 2 as a point mass at the tip. A real club is not
that, so an equivalence is required. The invariant preserved by `club_equivalence` is
**inertia about the wrist**, because it sets the uncoupled wrist-row mass term M22:
`delta = I_com + m * r²`.

Matching the wrist-row inertia gives an equivalent tip mass `me = delta_real / L2² = 0.238 kg`,
implemented in [`club_equivalence.py`](../../src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/club_equivalence.py).

### Coupling error and mass matrix limits

Matching the wrist-row inertia delta does **not** match the full mass matrix or arm/club coupling.
For an axial center of mass at distance `r`, the real arm/club coupling is `mu_real = m * L1 * r`.
In the equivalent point-mass model with mass `me = delta / L2²` at tip distance `L2`, the modeled coupling is:

```
mu_model = me * L1 * L2 = (L1 / L2) * delta = (L1 / L2) * (I_com + m * r²)
```

After matching delta, the residual coupling error is:

```
mu_model - mu_real = (L1 / L2) * [ I_com - m * r * (L2 - r) ]
```

Notice that the bracket is identically the factor appearing in the impact-optimality theorem (§1).
Furthermore, the base diagonal arm inertia term `M11` differs by `(me - m) * L1²` plus the coupling error
in the `cos(phi)` term. Matching `delta` reproduces `M22`, but leaves coupling and diagonal discrepancies.

### Geometric and measurement distinctions

* **Butt-based club properties**: Clubfitting measurements (such as `DRIVER_SPEC`) report inertia and
  COM relative to the physical butt of the grip.
* **Model wrist origin**: In the double pendulum model, segment 2 rotates about the wrist joint axis,
  which lies several centimeters down the grip from the physical butt where the hands grasp the club.
* **Physical versus modeled length**: Physical driver length (e.g. 1.143 m / 45 in butt-to-sole) differs
  from the modeled second-link length `L2` (e.g. 1.10 m), which is the distance from the wrist joint
  pivot to the modeled head center of mass at the tip.

|                         | Real driver     | 0.310 kg at tip | Old preset (0.500 kg) | Ratio (old / real) |
| ----------------------- | --------------- | --------------- | --------------------- | ------------------ |
| Club mass               | 0.310 kg        | 0.310 kg        | 0.500 kg              | 1.61×              |
| COM from wrist          | 0.867 m         | 1.100 m         | 1.100 m               | 1.27×              |
| **Inertia about wrist** | **0.288 kg·m²** | **0.3751 kg·m²**| **0.605 kg·m²**       | **2.10×**          |
| **Coupling `mu`**       | **0.175 kg·m²** | **0.222 kg·m²** | **0.358 kg·m²**       | **2.05×**          |

Placing the real driver's 0.310 kg at the tip of a 1.10 m modeled shaft yields `delta = 0.3751 kg·m²`
(0.310 * 1.10² = 0.3751), which is 1.30× the real value. The historical preset lumped 0.500 kg at the tip,
producing 0.605 kg·m² (0.500 * 1.10² = 0.605, 2.10×) and severe coupling distortion.

### Effect, holding everything else fixed

| Lumped tip mass            | Clubhead     | Hands        | Club/arm | Braking |
| -------------------------- | ------------ | ------------ | -------- | ------- |
| 0.500 kg (old preset)      | 36.4 m/s     | **0.36 m/s** | **59.4** | 33%     |
| 0.320 kg                   | 45.3 m/s     | 6.01 m/s     | 3.86     | 24%     |
| **0.238 kg (real driver)** | **50.8 m/s** | **7.95 m/s** | **3.18** | 24%     |

All defects ≤ 2e-13. Note that in the corrected mass sweep (0.238 kg), **24% modeled braking remains**:
the optimizer still reverses hub torque for roughly a quarter of the downswing to achieve club release
at impact in this planar fixed-hub geometry.
Independently corroborated by the `Double-Pendulum-Optimization`
research repo, whose model carries distributed club inertia natively and reaches
6.1–7.4 m/s hand speed robustly across arm-inertia, torque-budget and duration sweeps.

## 2. What the literature says a golfer actually does

The two-link golf pendulum is old and well characterised. [Williams
(1967)](https://doi.org/10.1093/qjmam/20.2.247) and [Jorgensen
(1970)](https://doi.org/10.1119/1.1976419) established the model; [Cochran & Stobbs
(1968)](https://archive.org/details/searchforperfect0000coch), _Search for the Perfect
Swing_, is the standard reference treatment. [Pickering & Vickers
(1999)](https://doi.org/10.1007/BF02844532) revisited its assumptions directly.

The six intervals below are **heuristic reference intervals** synthesized to score
planar double-pendulum simulations against plausible skilled golfer kinematics,
rather than exact primary-source reported literature bounds for a single professional-tour
cohort. Exact primary-source locations and measurement mappings for professional
tour players are not supplied for these widened heuristic bands. For example,
[Nesbit (2005)](https://www.jssm.org/jssm-04-499.xml.xml) Table 3 reports grip speed
7.1–8.0 m/s and head speed 43–50 m/s for its described amateur cohort (subjects across
four handicap categories from scratch to high); this is not an exact professional-tour
6–9 m/s / 45–55 m/s interval.

| Observable                      | Heuristic band      | Source                                                                                                    |
| ------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------- |
| Clubhead speed at impact        | 45–55 m/s           | Heuristic band; [Nesbit 2005](https://www.jssm.org/jssm-04-499.xml.xml) Table 3 reports 43–50 m/s for amateur cohort |
| **Hand speed at impact**        | **6–9 m/s**         | Heuristic band; [Nesbit 2005](https://www.jssm.org/jssm-04-499.xml.xml) Table 3 reports 7.1–8.0 m/s grip speed; [Miura 2001](https://doi.org/10.1007/BF02844309) |
| Downswing duration              | 0.23–0.32 s         | Heuristic band; [Jorgensen 1970](https://doi.org/10.1119/1.1976419); [Nesbit 2005](https://www.jssm.org/jssm-04-499.xml.xml) |
| Club / arm rate ratio at impact | 2.5–4               | Heuristic band derived from [Nesbit 2005](https://www.jssm.org/jssm-04-499.xml.xml) segment angular velocities |
| Wrist cock at impact            | −5 to 20°           | Heuristic band; [MacKenzie & Sprigings 2009](https://doi.org/10.1007/s12283-009-0020-9) release timing |
| Half-release point              | 55–80% of downswing | Heuristic band; [Sprigings & Neal 2000](https://doi.org/10.1123/jab.16.4.356); [MacKenzie & Sprigings 2009](https://doi.org/10.1007/s12283-009-0020-9) |

Two results from that literature bear directly on the diagnosis.

**Hands do decelerate — but nothing like this.** [Miura
(2001)](https://doi.org/10.1007/BF02844309) measured the inward hand pull near impact
and named the resulting clubhead gain _parametric acceleration_: shortening the hand
radius transfers speed to the head. Real hand deceleration is a real effect. The model
exaggerates it to a physically impossible degree — and, critically, achieves it by a
mechanism (a fixed radius plus a reversed hub torque) that is not the one Miura measured.

**Arm deceleration is largely passive.** [Sprigings & Neal
(2000)](https://doi.org/10.1123/jab.16.4.356) and [Nesbit & Serrano
(2005)](https://www.jssm.org/jssm-04-520.xml.xml) find the deceleration of the proximal
segment is dominated by the interaction torque from the accelerating club, not by
muscular braking. The shipped model produces it by applying **+222 N·m of active braking
torque for 32% of the downswing**. That is the wrong mechanism, not merely the wrong
magnitude.

### Active lengthening and biomechanical distinctions

Active muscle lengthening occurs when an active muscle is stretched by an external load
(eccentric contraction). In this regime, the muscle performs negative mechanical work on the
skeletal segment, absorbing mechanical energy. The implemented braking branch already permits
active lengthening and energy absorption.

Crucially, **net actuator power, segment deceleration, and individual-muscle activation are distinct**:
1. **Net actuator power** (`P_net = tau * omega`): The net rate of work done across the joint.
   When torque opposes velocity (`P_net < 0`), net actuator power is negative and mechanical
   energy is absorbed from the segment.
2. **Segment deceleration** (`omega * omegadot < 0`): Deceleration of a limb segment can arise
   passively from inertial interaction / coupling torques (such as uncocking acceleration)
   even when net actuator torque is driving or zero. Segment deceleration does *not* imply active
   muscular braking.
3. **Individual-muscle activation**: Agonist and antagonist muscles can co-contract, and
   individual muscle units can actively lengthen while net joint torque remains positive.
   The lumped joint torque model does not resolve individual-muscle activation states.

Reference data and scoring: [`reference_kinematics.py`](../../src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/reference_kinematics.py). Every band carries its source and a resolvable link, and a test enforces that.

---

## 3. Actuation limits: necessary, not sufficient

The shipped `TorqueClamp` is symmetric and velocity-independent, so braking costs exactly
what driving costs. Two pieces of physiology are missing.

**Torque falls with joint speed.** [Hill (1938)](https://doi.org/10.1098/rspb.1938.0050)
established the hyperbolic force–velocity relation; at a joint it becomes

```
tau_max(w) = tau0 * (w_max - w) / (w_max + curvature * w)
```

Golf forward-dynamics models have carried limits of this kind since [Sprigings & Neal
(2000)](https://doi.org/10.1123/jab.16.4.356) and [MacKenzie & Sprigings
(2009)](https://doi.org/10.1007/s12283-009-0020-9).

**Braking uses different, weaker muscles.** Antagonist capacity is modelled as a fraction
of the driving peak, raised by an eccentric gain (permitting active lengthening and energy absorption).

Implementation: [`actuation.py`](../../src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/actuation.py).

### What they actually do

| Model variant             | Hand at impact | Club/arm ratio | Braking | Feasible? (defect < 1e-6) |
| ------------------------- | -------------- | -------------- | ------- | ------------------------- |
| Symmetric clamp (shipped) | 0.36 m/s       | 59.2           | 32%     | yes                       |
| Weak brake only           | 0.08 m/s       | 235.7          | 44%     | yes                       |
| **Hill torque–velocity**  | **6.8 m/s**    | **2.8**        | **0%**  | **no**                    |
| Hill + weak brake         | 7.0 m/s        | 2.8            | 0%      | no                        |

The Hill candidate produces kinematics close to heuristic targets (hands at 6.8 m/s, ratio 2.8,
and zero active braking), but **its order-one dynamics defects (~1.0) prevent presenting its
6.8 m/s output as a feasible or measured swing**. It is dynamically infeasible under this model formulation.
The "Feasible?" column indicates satisfaction of the collocation dynamics defect tolerance predicate
(`max_defect < 1e-6`), not verification of every constraint or a certificate of global reachability/infeasibility.

---

## 4. The structural result

Hub torque does not only turn the arms. Through the off-diagonal mass-matrix term `M12`
it also drives the wrist _open_. In a free rollout at full drive the wrist cock grows
from 100° to **184°** — the club never releases at all; it just lags further.

The only way this model brings the club through to `phi = 0` at impact is to cut, and
then reverse, the hub torque, which does cost some hand speed. That mechanism is real
and survives the correction — an inertia-matched club still spends about 24% of the
downswing with hub torque opposing the arms.

What does **not** survive is the severity. With the coupling at its correct value the
cost is a few m/s of hand speed, not all of it.

`hand_speed_frontier` measured the price with the **mis-specified** club under historical
experiment settings (duration 0.36 s, ±250 N·m):

| Hand-speed floor          | Feasible (defect < 1e-6) | Clubhead speed | Club/arm ratio |
| ------------------------- | ------------------------ | -------------- | -------------- |
| none                      | yes                      | 36.4 m/s       | 59.2           |
| 3 m/s                     | yes                      | 34.1 m/s       | 6.1            |
| 5 m/s                     | marginal                 | 29.6 m/s       | 2.9            |
| **6 m/s (heuristic band)**| **no**                   | —              | —              |

(Note: "Feasible" reports satisfaction of the collocation dynamics defect threshold predicate
`max_defect < 1e-6`; rising defects indicate solver non-convergence under the collocation grid,
not a global certificate of infeasibility.)

With the **corrected** club (evaluated under the revised objective comparison duration of 0.28 s)
the picture is entirely different: the unconstrained optimum already sits at 7.26 m/s, inside the
heuristic band, and floors up to 8 m/s satisfy the defect predicate. Approximately 24% modeled
braking remains in the corrected sweep. The table above characterises the artifact, not the model.
Implementation: [`model_adequacy.py`](../../src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/model_adequacy.py);
both regimes are pinned in `tests/test_model_adequacy.py`.

---

## 5. So which objective is a good golfer optimizing?

Asked properly — solve each objective under identical conditions (duration 0.28 s), reduce to the
observables, score against heuristic reference bands — the answer is that **the question cannot be
settled on this model**.

| Regime        | Deviation spread across objectives | Observables inside band |
| ------------- | ---------------------------------- | ----------------------- |
| Unconstrained | 81.5 → 120.3 (Coriolis worst)      | 1 of 6                  |
| Hands ≥ 3 m/s | 8.66 → 8.71 (**0.6% spread**)      | 1 of 6                  |

In the only near-realistic regime the model can reach, the five objectives land within
0.6% of each other while every one of them sits far outside the heuristic reference bands. The
spread between objectives is an order of magnitude smaller than the gap between all of
them and a real swing.

**The objective is not what makes these swings unrealistic.** `is_discriminating` reports
this directly so a 0.6% ordering never gets quoted as "golfers optimize X".
Implementation: [`objective_realism.py`](../../src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/objective_realism.py).

The one thing that _is_ separable: under the unconstrained model, Coriolis
kinetic-chain transfer is a genuine dissenter — it costs 2.7% of clubhead speed and the
other four objectives reach only 98.4% of its transfer value. That result stands from
epic #4766 and is unaffected by this analysis.

---

## 6. What would actually fix it

In priority order, with the evidence for each.

0. **Correct the club first.** Done (#4785), and it was the dominant term. Everything
   below is now an improvement rather than a prerequisite.
1. **A moving hub — a torso segment.** Still the largest remaining physics gap, and the
   most likely route to fixing the late release. A real golfer's
   [`physics_triple.py`](../../src/pendulum_simulator/src/double_pendulum_golf/physics_triple.py);
   the work is to give it the objective and actuation layers built here.
   [Nesbit (2005)](https://www.jssm.org/jssm-04-499.xml.xml) and [MacKenzie & Sprigings
   (2009)](https://doi.org/10.1007/s12283-009-0020-9) both treat the torso as a driven
   segment for exactly this reason.
2. **A variable hand radius.** [Miura (2001)](https://doi.org/10.1007/BF02844309)'s
   parametric acceleration needs `L1` to shorten through impact. Not expressible with the
   current fixed-length links, and it is the measured mechanism for the hand
   deceleration this model fakes with torque reversal.
3. **Hill actuation, retained.** Already built and tested. It is necessary — it removes
   the impossible braking — and becomes sufficient only once (1) makes the release
   achievable without reversing the hub.
4. **Not** distributed club inertia _as a route to forward hand speed_ — ruled out
   analytically in §1. Note the distinction from §2a: what matters is matching the real
   club's inertia about the wrist, which the corrected preset now does. The §1 result is
   about the unconstrained ideal; §2a is about the constrained optimum actually reached.

### Scientific boundary

Planar two-link model, point-mass arm and clubhead, fixed hub, constant torque limits,
no shaft flex, no ground reaction, no plane change. The proximal link's angular rate is
not an anatomical shoulder or thorax velocity. Every number here describes a two-link
chain under a torque budget; none of it is anatomical attribution or coaching authority.

---

## 7. References

- Cochran, A. & Stobbs, J. (1968). _[Search for the Perfect Swing](https://archive.org/details/searchforperfect0000coch)_.
- Hill, A. V. (1938). [The heat of shortening and the dynamic constants of muscle](https://doi.org/10.1098/rspb.1938.0050). _Proc. R. Soc. Lond. B_ 126, 136–195.
- Jorgensen, T. (1970). [On the dynamics of the swing of a golf club](https://doi.org/10.1119/1.1976419). _Am. J. Phys._ 38(5), 644–651.
- MacKenzie, S. J. & Sprigings, E. J. (2009). [A three-dimensional forward dynamics model of the golf swing](https://doi.org/10.1007/s12283-009-0020-9). _Sports Eng._ 11(4), 165–175.
- Miura, K. (2001). [Parametric acceleration — the effect of inward pull of the golf club at impact stage](https://doi.org/10.1007/BF02844309). _Sports Eng._ 4, 75–86.
- Nesbit, S. M. (2005). [A three dimensional kinematic and kinetic study of the golf swing](https://www.jssm.org/jssm-04-499.xml.xml). _J. Sports Sci. Med._ 4(4), 499–519.
- Nesbit, S. M. & Serrano, M. (2005). [Work and power analysis of the golf swing](https://www.jssm.org/jssm-04-520.xml.xml). _J. Sports Sci. Med._ 4(4), 520–533.
- Pickering, W. M. & Vickers, G. T. (1999). [On the double pendulum model of the golf swing](https://doi.org/10.1007/BF02844532). _Sports Eng._ 2(3), 161–172.
- Sprigings, E. J. & Neal, R. J. (2000). [An insight into the importance of wrist torque in driving the golfball](https://doi.org/10.1123/jab.16.4.356). _J. Appl. Biomech._ 16(4), 356–366.
- Williams, D. (1967). [The dynamics of the golf swing](https://doi.org/10.1093/qjmam/20.2.247). _Q. J. Mech. Appl. Math._ 20(2), 247–264.
