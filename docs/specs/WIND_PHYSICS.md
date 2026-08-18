# Reproducible Ball-Flight Wind Physics

## Scope

The wind layer supplies air velocity to the existing golf-ball aerodynamic
models. Aerodynamic forces use relative velocity

```text
v_relative = v_ball - v_wind
```

at the integrator's physical time and ball position. Wind affects the actual
drag and lift calculation; it is not a display-only offset.

The canonical flight frame is right-handed: `x` forward along the target
line, `y` left, and `z` up. `base_velocity_mps` is the direction the air moves
*to*. All components and positions are SI.

## Meteorological Adapter

The UI accepts a horizontal speed and a bearing the wind comes *from*, measured
clockwise from the target line:

- 0 degrees is a headwind from the target;
- 90 degrees comes from the player's right and moves toward flight-frame left;
- 180 degrees is a tailwind;
- 270 degrees comes from the player's left.

For speed `s`, from-bearing `b`, and upward component `w_z`, the flight-frame
wind-to vector is

```text
[-s cos(b), s sin(b), w_z]
```

Both interfaces label the from-bearing and show the corresponding wind-to
direction to avoid the common from/to ambiguity.

## Time- and Altitude-Varying Components

A `wind-scenario/v1` record may include:

- a constant three-dimensional base vector;
- linear fractional speed shear per 10 m above ground;
- declared gust events with start time, duration, and peak vector;
- deterministic seeded turbulence intensity and provenance.

Gusts use a squared-sine envelope, so each event is exactly zero at its start
and end and reaches the declared peak at its midpoint. The turbulence function
uses six deterministic harmonics per axis. It is intentionally a reproducible
perturbation for sensitivity and strategy studies; it is not a validated
von Karman/Kaimal spectrum or a forecast for a particular course. The shared
golden fixture pins Python and TypeScript field values to `1e-12` m/s.

The original reusable concepts were audited from UpstreamDrift's
`physics/aerodynamics/_wind.py`: base wind, sinusoidal gusts, turbulence,
altitude gradient, and seeded evaluation. Tools owns the reusable contract and
integrator coupling so downstream applications can import one implementation.

## Paired Comparison

Wind comparisons use identical launch, ball, environment, and flight-model
inputs. One trajectory has no wind; the other has the selected scenario.
Reported deltas are always

```text
selected wind result - no-wind result
```

for carry, lateral landing, apex, flight time, and landing angle. Both paths
are retained and rendered. The two-dimensional canvases and Matplotlib axes use
one metres-to-pixels/unit scale in each view, preventing a trajectory from
appearing steeper or flatter because the horizontal and vertical scales differ.

## Backend Boundaries

The Python literature models and TypeScript Waterloo/Penner model support the
full scenario evaluation implemented here. The current Rust flight API accepts
only one constant environmental wind vector; the facade therefore rejects
shear, gust, or turbulence instead of silently sampling them at launch.

The initial UI exposes the qualified steady horizontal case. Vertical wind,
shear, declared gust schedules, seeded turbulence, realized-history export,
wind-estimate error distributions, and strategy optimization remain explicit
follow-on integrations under issues #4198 and #4199. These limitations must
remain visible until their controls, exports, performance budgets, and
validation evidence are delivered.
