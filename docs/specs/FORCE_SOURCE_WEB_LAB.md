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
integration step, impact-path tolerance, bottom reach, speed band, positive
actuator-work cap, squared-torque-effort cap, minimum robust qualification, and
held-out perturbations are numeric inputs. Wrist torque is hard-limited to
30 N m. Invalid or internally inconsistent inputs fail before simulation.

The comparison basis is explicit:

- `equal_speed` requires every nominal winner to lie between the selected TOUR
  target and target plus the selected band width, under common positive-work
  and squared-effort caps;
- `equal_effort` applies the same input caps but leaves clubhead speed as an
  observed outcome; and
- `common_bounds` applies the torque, slew, timing, and motion limits without
  equalizing realized work.

Clubhead speed is a feasibility filter only in `equal_speed`; it is not added
to a component objective's score. The component score therefore cannot buy a
better rank merely by hiding a clubhead-speed reward. Conversely, a common
peak-torque envelope is not described as equal effort: realized work, braking,
torque impulse, squared effort, RMS torque, and peak power remain visible.

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
perturbations. Final selection considers both nominal objective elites and
high-headroom candidates, and rejects any winner below the selected held-out
qualification rate. Headroom covers speed, work, activation, path angle, and
bottom reach; it does not alter the component score.

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
shoulder and wrist torque sample. Plotted actuator power and cumulative work
must reproduce torque times angular velocity and trapezoidal integration.
Derived effort totals and stable control-profile IDs are also recomputed on
import. Impact diagnostics are checked against the contract's selected
path-angle and bottom-reach thresholds, not hidden defaults.

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

## Interpretation of the Registered Comparisons

The checked-in version-4 artifacts each run all six objectives under an
independent research contract: 2,048 deterministic global/seeded candidates,
16 nominal elite starts, 12 coefficient-refinement rounds, 1 ms integration,
0.5 N m wrist-coefficient granularity, and 25 held-out robustness trials. They
use the repository-authoritative inertia-matched driver equivalent
(`m2 + mClub = 0.2381186694 kg`) and a selectable symmetric 250 N m hub-torque
budget.

The default equal-speed study requires 52.30–53.05 m/s, no more than 525 J of
positive actuator work, no more than 7,500 N²m²s of squared torque effort, and
at least 60% held-out qualification. Its three distinct robust control programs
reach 52.87–53.05 m/s. Coriolis and centrifugal impulse share one program;
Coriolis energy, centrifugal energy, and hand-path impulse share a second; the
speed row selects a third. These identities are explicit profile IDs rather
than six visually duplicated claims.

The equal-effort capacity study uses the same input caps and robustness floor
without a speed filter. It exposes the output difference the equal-speed study
necessarily hides: the robust centrifugal-impulse winner reaches about
34.7 m/s, the Coriolis-impulse winner about 51.1 m/s, and the energy/speed/hand-
path program about 53.5 m/s. This is evidence about the declared optimization
and model, not proof that one named inertial term independently causes human
clubhead speed. The two energy objectives must remain identical because their
declared powers satisfy the exact 2:1 identity.

The 52.3 m/s lower bound corresponds to the PGA TOUR's 116.96 mph 2026
year-to-date average reported on the official
[Club Head Speed stat](https://www.pgatour.com/stats/detail/02401). Boundary
hits, robustness, profile identity, positive/net/negative work, peak/RMS
activation, power, slew, transition timing, coefficients, and the complete
cross-objective matrix accompany every comparison.

## Architecture

- `forceSourceTypes.ts` owns the schema and user-visible constraints.
- `forceSourceOptimization.ts` owns validation, deterministic search,
  qualification, scoring, and robustness.
- `forceSourceArtifact.ts` owns version-4 contract identity, polynomial/plot
  consistency, cross-objective
  dominance validation, fail-closed parsing, and contract-aware updates.
- `forceSourceView.ts` owns interpolation and explicit camera alignment.
- `ForceSourceLab.tsx` coordinates state and commands.
- `ForceSourceResults.tsx` renders animations, all seventeen sampled channels,
  the cross-objective/Pareto table, control-strategy diagnostics, and all
  polynomial coefficients.

Physics remains in `physics.ts`; React components do not implement equations of
motion. Older mixed or bang-bang artifacts are intentionally rejected rather
than silently promoted to comparable evidence.
