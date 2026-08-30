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
bounds, coefficient increments, polynomial-duration range, torque-slew ceiling,
low-torque wrist-transition band and duration, elite count, candidate budget,
integration step, impact-path tolerance, bottom reach, speed target, and
held-out perturbations are numeric inputs. Wrist torque is hard-limited to
30 N m. Invalid or internally inconsistent inputs fail before simulation.

Each joint is controlled by a degree-6 Bernstein polynomial with seven torque
coefficients. Bernstein form is still a sixth-order polynomial, but its
convex-hull property makes the bound contract strong: when every control point
is inside the torque limits, every value on the continuous curve is also inside
them. Both profiles finish at zero; the wrist profile also starts at zero,
crosses from restraining to driving torque exactly once, remains inside the
selected low-torque band for the minimum transition time, and respects the
selected analytic slew bound.

`quick`, `thorough`, and `research` use deterministic low-discrepancy global
sampling plus physically shaped constant-drive, front-loaded, braking, ramped,
early-release, and late-release seeds. They retain multiple elite starts and
run respectively two, six, or twelve coefficient-refinement rounds. The
candidate and elite budgets are explicit rather than hidden behind the depth
name. Winning programs are rerun under start-pose and whole-profile torque
perturbations; the cards report the fraction that still qualifies.

Every scenario in one comparison carries the same versioned contract ID. The
contract covers the initial state, model parameters, all constraints, candidate
budget, integration step, robustness settings, and search depth. Changing any
of those inputs starts a new comparison instead of retaining stale rows. After
the independent searches finish, every winning candidate is evaluated against
every displayed objective. An artifact is rejected if an objective's own row
loses to another displayed row. This is a displayed-candidate certification,
not a proof of the mathematical global optimum.

Imported coefficients and durations must fall on their declared grids, satisfy
the complete continuity/transition/slew contract, and reproduce every plotted
shoulder and wrist torque sample. Impact diagnostics are checked against the
contract's selected path-angle and bottom-reach thresholds, not hidden defaults.

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
and uses one scale and one registered starting pose. Fixed-hub cards show only
the physical hub, wrist, links, and clubhead; they do not show an impact target
or reference line. The optional impact-aligned view translates each card's
camera so impact tips coincide at a labelled camera-only crosshair; it does not
alter dynamics and may make hubs appear to drift. Playback uses
continuous interpolation on `requestAnimationFrame`, rate controls from 0.05×
to 3×, and a 0.25 ms scrubber.

The energy channels have an explicit interface meaning. Coriolis transfer is
the power drained from the proximal arm, while centrifugal (squared-speed)
transfer is power delivered at the distal wrist coordinate. In the declared
relative-wrist coordinates they satisfy
`P_coriolis_to_distal = 2 * P_centrifugal_to_distal` at every sample. A unit
test pins both the sign and this identity.

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

The checked-in version-3 artifact runs all six objectives under one research
contract: 512 deterministic global/seeded candidates, 12 elite refinement
rounds, 1 ms integration, 0.5 N m wrist-coefficient granularity, and 25 held-out
robustness trials. It now uses the repository-authoritative inertia-matched
driver equivalent (`m2 + mClub = 0.2381186694 kg`) and a selectable symmetric
250 N m hub-torque budget instead of the web preset's stale 0.50 kg lumped club.

The certified clubhead-speed winner reaches about 53.7 m/s (120.2 mph), above
the 52.3 m/s marker corresponding to the PGA TOUR's 116.96 mph 2026 year-to-date
average reported on the official
[Club Head Speed stat](https://www.pgatour.com/stats/detail/02401). Coriolis
impulse selects a distinct approximately 50.9 m/s strategy and centrifugal
impulse selects a markedly different approximately 34.7 m/s strategy; the
energy-transfer, speed, and hand-path rows share the displayed speed winner.
That agreement is reported rather than artificially broken. Boundary hits,
qualification rates, polynomial coefficients, work, slew, transition timing,
and the full cross-objective score/rank matrix accompany the ranking.

## Architecture

- `forceSourceTypes.ts` owns the schema and user-visible constraints.
- `forceSourceOptimization.ts` owns validation, deterministic search,
  qualification, scoring, and robustness.
- `forceSourceArtifact.ts` owns version-3 contract identity, polynomial/plot
  consistency, cross-objective
  dominance validation, fail-closed parsing, and contract-aware updates.
- `forceSourceView.ts` owns interpolation and explicit camera alignment.
- `ForceSourceLab.tsx` coordinates state and commands.
- `ForceSourceResults.tsx` renders animations, all twelve sampled channels,
  the cross-objective/Pareto table, control-strategy diagnostics, and all
  polynomial coefficients.

Physics remains in `physics.ts`; React components do not implement equations of
motion. Older mixed or bang-bang artifacts are intentionally rejected rather
than silently promoted to comparable evidence.
