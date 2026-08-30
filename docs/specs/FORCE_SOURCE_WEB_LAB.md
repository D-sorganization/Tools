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

Every scenario in one comparison carries the same versioned contract ID. The
contract covers the initial state, model parameters, all constraints, candidate
budget, integration step, robustness settings, and search depth. Changing any
of those inputs starts a new comparison instead of retaining stale rows. After
the independent searches finish, every winning candidate is evaluated against
every displayed objective. An artifact is rejected if an objective's own row
loses to another displayed row. This is a displayed-candidate certification,
not a proof of the mathematical global optimum.

Imported candidates must also fall on the torque and onset grids declared by
that contract. Impact diagnostics are checked against the contract's selected
path-angle and bottom-reach thresholds, not hidden default thresholds.

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

The checked-in version-2 artifact runs all six objectives under one research
contract: 512 deterministic global candidates, 1 ms integration, 0.5 N m wrist
granularity, and 25 held-out robustness trials. The certified clubhead-speed
winner reaches about 38.33 m/s, while the hand-path-impulse winner reaches about
38.03 m/s. The Coriolis impulse and the two energy-transfer objectives select
the same displayed candidate as clubhead speed; centrifugal release impulse is
slower at about 30.93 m/s. These values are mechanically plausible for this
simplified club model, but they are not a validated human-performance range.
The canonical inertia-matched two-link study reaches 49.7 m/s under a different
torque budget and club representation. Boundary hits and qualification rates
must accompany any ranking claim.

## Architecture

- `forceSourceTypes.ts` owns the schema and user-visible constraints.
- `forceSourceOptimization.ts` owns validation, deterministic search,
  qualification, scoring, and robustness.
- `forceSourceArtifact.ts` owns version-2 contract identity, cross-objective
  dominance validation, fail-closed parsing, and contract-aware updates.
- `forceSourceView.ts` owns interpolation and explicit camera alignment.
- `ForceSourceLab.tsx` coordinates state and commands.
- `ForceSourceResults.tsx` renders animations and supplemental plots.

Physics remains in `physics.ts`; React components do not implement equations of
motion. Version-1 mixed-contract artifacts are intentionally rejected rather
than silently promoted to comparable evidence.
