# Wedge Turf-Contact Model

## Purpose and Claim Boundary

This module provides a replaceable, passive first-order contact law for wedge
sole diagnostics. It is intended for software verification, sensitivity
studies, and solver integration. It does not predict divot shape, grass or root
fracture, granular sand flow, injury, or the behavior of a named course.

The supplied `firm_fairway`, `soft_turf`, and `sand_like` profiles are
illustrative engineering starting values. They are not fitted measurements.
Only a profile explicitly marked `calibrated`, with its own provenance and
uncertainty record, permits downstream turf-supported ranking language.

## Frames, Signs, and Units

All quantities use SI units in the declared ground frame. The surface normal
`n` is unit length and points out of the ground. Penetration `delta` is
non-negative. Relative normal velocity is

```text
v_n = (v_contact - v_surface) dot n
delta_dot = -v_n
```

so an approaching contact has `v_n < 0` and `delta_dot > 0`. A wrench is
reported at a declared reference point. For contact point `P` and reference
point `R`,

```text
tau_R = (P - R) cross F
```

The Rate adapter preserves its established ground frame: x toward the target,
y up, and z right of the target.

## Instantaneous Law

The unilateral Kelvin-Voigt normal force is

```text
F_n = max(0, k delta + c delta_dot)
```

and the regularized Coulomb tangential force is

```text
F_t = -mu F_n v_t / sqrt(||v_t||^2 + v_epsilon^2)
```

The regularization makes the law continuous at zero tangential speed. The
instantaneous dissipation rate is `c delta_dot^2 - F_t dot v_t`, which is
non-negative inside the declared model. Stored elastic energy is
`0.5 k delta^2`.

The unilateral tensile branch is clipped. If the reduced model detaches with
spring energy remaining, that energy is reported separately as
`separation_loss_energy_j` and included in total dissipated energy. It is not
allowed to vanish from the balance.

## Wedge Contact Quadrature

The wedge evaluator applies the same law to the nine stable candidates shared
with the swept-clearance model:

- leading edge: heel, center, and toe;
- primary sole: heel, center, and toe;
- trailing sole: heel, center, and toe.

Each point receives one ninth of the profile stiffness and damping. The summed
wrench therefore remains tied to the declared aggregate profile instead of
becoming nine times stiffer merely because the geometry has nine samples.
This is a deterministic quadrature, not a continuous pressure field. The
active named regions, maximum penetration, stored energy, dissipated power,
force, and moment about the head origin are retained.

A candidate above the surface has no force even when moving toward it. A
candidate on or below the surface uses its full rigid-body point velocity,
`v_R + omega cross (P - R)`. Sloped planes are supported through their declared
normal and origin, making surface height spatially varying in the ground frame.

## Reduced Dynamics and Solver Integration

`simulate_reduced_turf_contact` integrates one effective contact mass with a
caller-controlled timestep, maximum duration, and cooperative cancellation
callback. It returns typed no-contact, no-response, separated, out-of-domain,
step-limit, and cancelled states. It is a convergence and sensitivity
diagnostic; it is not the wedge's complete six-degree-of-freedom motion.

`evaluate_wedge_turf_wrench` is the force-coupling seam for a full dynamics
solver. It consumes a head pose and twist and returns the net ground-frame
wrench at the head origin. The Rate retained-run adapter evaluates this wrench
at first geometric ground contact, but deliberately does not replay an already
retained swing under that force. Its limitation string makes this distinction
visible.

`run_turf_convergence_study` executes a declared coarse-to-fine timestep plan.
It reports the finest-pair relative changes in normal impulse, peak
penetration, and dissipated energy and marks convergence only when every run
separates normally and every change is within the requested tolerance.

## Persistence and Validation

Profiles use strict deterministic JSON format
`golf-club.turf-profile/v1`. Unknown fields, wrong format identifiers,
non-finite numbers, invalid bounds, and missing provenance are rejected.
Serialization includes calibration status, source name, parameter basis,
uncertainty, and optional source URI.

The test suite pins:

- the analytic normal force, wrench, stored energy, and dissipation rate;
- passivity, friction direction, the Coulomb bound, and energy balance;
- no-response, frictionless, no-contact, sloped-frame, and stiffness limits;
- rotation equivariance and timestep refinement;
- cancellation and typed outside-domain behavior;
- active wedge-region changes with delivered pitch and ground slope;
- deterministic strict profile persistence; and
- the Rate retained-state adapter and its explicit coupling limitation.

## Remaining Validation Work

Before making real-surface or universal bounce-forgiveness claims, obtain
repeatable force-displacement-velocity data for declared turf preparations,
fit parameters with uncertainty, validate contact patch density and full
rigid-body response against held-out trials, and publish the calibration
record. The current illustrative presets cannot satisfy that evidence burden.
