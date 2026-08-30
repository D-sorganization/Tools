# Force-Source Web Lab Design Contract

## Purpose

The React force-source lab compares six double-pendulum optimizations on one
timeline and one declared visual frame. It is an exploratory model workspace,
not a golfer-fitting or coaching surface.

The objectives are Coriolis impulse, Coriolis energy transfer, squared-speed
impulse, squared-speed energy transfer, impact clubhead speed, and signed grip
force impulse along the hand path. Every candidate must make one rightward pass
near the bottom of the arc without an arm, club, or wrist loop.

## User-Controlled Contract

The user may type the starting arm and relative-wrist angles directly. Search
bounds, torque ranges and increments, release-onset range and increment,
candidate budget, integration step, impact-path tolerance, bottom reach, and
held-out perturbations are also numeric inputs. Wrist drive and restrain torque
are hard-limited to 30 N m. Invalid or internally inconsistent inputs fail before
simulation.

`quick`, `thorough`, and `research` use one deterministic low-discrepancy global
sample and respectively one, three, or six local refinement rounds. The
candidate budget is explicit rather than hidden behind the depth name. Winning
programs are rerun under start-pose and torque perturbations; the cards report
the fraction that still qualifies.

## Impact Qualification

The first forward crossing is the only eligible impact. Before that event:

- arm, wrist, and club travel must remain inside the entered bounds;
- arm and club travel must remain below the anti-loop limits;
- clubhead velocity at impact must point rightward;
- vertical path deviation must not exceed the entered tolerance; and
- downward reach must meet the entered fraction of total link length.

This deliberately excludes delayed-release trajectories that circle through the
ball or arrive steeply. The qualification is a model geometry screen, not proof
of human feasibility.

## Comparison Frame and Playback

`Fixed hub` is the default and places every model shoulder at the same SVG point
and uses one scale. The physical hub is fixed in every simulation. The optional
impact-aligned view translates each card's camera so impact tips coincide; it
does not alter dynamics and may make hubs appear to drift. Playback uses
continuous interpolation on `requestAnimationFrame`, rate controls from 0.05×
to 3×, and a 0.25 ms scrubber.

## Hand-Path Impulse

At each moving sample the browser resolves the physical distal grip force along
the instantaneous wrist velocity and integrates the signed projection over
time. A zero-speed sample has no defined tangent and contributes zero. The
canonical Python surface exposes the same signed quantity as
`hand_path_impulse`.

This objective is inspired by Sasho MacKenzie's emphasis on force delivered
along the hand path, but it is intentionally not called a reproduction of that
published metric. MacKenzie-style average force along the path is work divided
by path length. The new score is impulse, `integral(F_parallel dt)`, and does not
include path speed. See MacKenzie and Lavers,
[How Amateur Golfers Deliver Energy to the Driver](https://www.golfsciencejournal.org/api/v1/articles/12640-how-amateur-golfers-deliver-energy-to-the-driver.pdf),
and the repository's rank-aware mapping contract in
`docs/development/pendulum-force-attribution.md`.

## Interpretation of the Registered Comparison

The five historical scenarios in the checked-in artifact produce impact speeds
of roughly 24.5–36.8 m/s; the reproducible research-depth hand-path scenario
reaches 38.0 m/s. Those values are mechanically plausible for a simplified club
model, but they are not a validated human-performance range. The canonical
inertia-matched two-link study reaches 49.7 m/s under a different torque budget
and club representation. Agreement in proximal-to-distal release and sensitivity
to wrist-torque timing is consistent with other double-pendulum studies;
numerical speed equality is not expected unless mass, inertia, initial pose,
torques, duration, coordinates, and impact event all match.

The historical registered comparison also shows that some apparent winners are fragile:
three of five have held-out qualification rates near 52–56%, one reaches 89%,
and one reaches 100%. Boundary hits and qualification rate must accompany any
claim about the best objective.

## Architecture

- `forceSourceTypes.ts` owns the schema and user-visible constraints.
- `forceSourceOptimization.ts` owns validation, deterministic search,
  qualification, scoring, and robustness.
- `forceSourceArtifact.ts` owns fail-closed artifact parsing and updates.
- `forceSourceView.ts` owns interpolation and explicit camera alignment.
- `ForceSourceLab.tsx` coordinates state and commands.
- `ForceSourceResults.tsx` renders animations and supplemental plots.

Physics remains in `physics.ts`; React components do not implement equations of
motion. Version-1 artifacts remain readable because the new hand-path force
series is additive and optional for the five historical scenarios.
