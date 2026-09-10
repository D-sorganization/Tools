# Impact Solver Handoff

## Active Work

- Event child #5151 is PR#5152 and passes386 Windows/Linux controls at tree4e1b19810; shared chart/geometry ports, independent piecewise event/work reference and strict SI/budget/history guards are retained. See NORMAL_EVENT_RESULTS.json; protected review, friction/modes and physical/acoustic qualification remain.
- Parent #5073 remains active. Numerical foundation child #5145 is under protected review in PR #5146 at published 1be394e90, based on merged shaft main 2c9a8d6c.
- Private geometry, normal/objective tangential work, full-tensor body response and instantaneous normal shaft/ball coupling are implemented. Head inertia is retained once; existing public APIs and legacy T2 equations are unchanged.
- The archived foundation passes 323 Windows and 323 Linux coverage controls. The isolated pre-push typing repair passes 46 affected controls and mypy across all 13 changed source files. Normal publication hooks and repository-wide Ruff checks pass.
- Exact source/JUnit hashes and retained failures are in `docs/development/impact-acoustics/SPATIAL_CONTACT_RESULTS.json`; derivations and remaining scope are in `SPATIAL_CONTACT_KINEMATICS.md` and `PROGRESS.md`.
- These are synthetic numerical controls. PR #5149 now supplies fixed-step normal contact; full flexible-contact and physical/acoustic qualification remain open.

## Next Steps

1. Resolve numerical foundation review; preserve canonical import paths.
2. Qualify the new private normal trajectory and its five-channel work ledger (12 controls pass; combined 371 Windows/Linux tests pass), then resolve contact events and integrate objective tangential history.
3. Couple full ball/head/shaft states and face/hosel modes; verify time, mesh, modes and events independently.
4. Compare matched-state detached/unloaded/preloaded cases with intervention energy and ringdown.
5. Qualify held-out force/spin/face-map observations before physical claims.

## Constraints

- Body twists are linear-first in material axes. The sphere's pose origin is its center.
- Contact force and reaction share one declared point; do not silently use different tangential lever arms.
- Point migration is not material surface velocity. Do not reuse a fixed-point load tangent as a complete contact derivative.
- Resolved friction/gear effect and the approximate legacy gear correction are mutually exclusive.
- Numerical agreement is not physical or acoustic validation. Parent epics remain open.

## Integrated Port Qualification

Provider 0cd6dce22 integration passes all 176 impact and legacy coupling
controls on Windows (51.01 s) and Linux coverage (62.71 s), without warnings
or skips. Exact source tree 218006534 and JUnit digests are retained in
SPATIAL_CONTACT_RESULTS.json. No coupled spatial trajectory or physical/
acoustic qualification is inferred. Further consumer CI repair is in progress
in the separate shaft/provider worktrees; this contact source is preserved.

## Rigid-Body Response Qualification

The private full-tensor free-body response starts with missing-module RED.
Its 18 initial controls and 57 existing moving-shaft controls pass together
(75 tests). Six additional relative-inertia domain cases expose two failures
for tiny nonphysical tensors; scaling the existing realizability check by
the tensor magnitude makes all 24 body controls pass. No tensor is projected
or regularized. NumPy-aware mypy passes all four changed production files.
The canonical shaft mass solve is extracted unchanged and shared, and its
linear-first spatial inertia accepts read-only physical fields without
inventing a club-component role for the ball. Observer momentum derivatives,
full-tensor Euler acceleration, eccentric COM, mechanical power, material-axis
relabeling and singular/unresolved mass refusal have independent controls.
Exact staged tree 45ccf7726 is archived for expanded Linux coverage. This is
an instantaneous response; spatial coupling and trajectories remain open.

## Coupled Normal-Response Qualification

The normal-only shaft/ball composition now recomputes the contact point,
normal force and common-point reaction for every supplied stage state. It
retains the shaft's existing head inertia, loads and moving grips. It adds no
second head mass and no approximate gear-effect correction. Independent
three-mass controls cover compression, clearance and unilateral cutoff;
nonplanar offset contact matches the total-energy derivative with moving
grips and remains invariant under observer rotation/translation. Central
normal contact creates no spurious spin of a centered isotropic ball.
The incomplete-state RED is corrected by validating the node count before
indexing. All 12 composition and 24 body tests pass together; five production
files pass NumPy-aware mypy. API RED exposed a changed annotation, so the old
club entry-point signature is preserved over a private shared inertia kernel.
Every pre-existing golf_club and swing_sim API record is unchanged; new
modules have no public exports. The earlier body/shaft archive passes all
311 Linux coverage tests. Expanded composition qualification passes all 323 Windows tests (43.21 s) and 323 Linux coverage tests (60.28 s), with no failures or skips, at archived source tree ec863402f. Source/JUnit digests are in SPATIAL_CONTACT_RESULTS.json.

Full spatial trajectories, discrete tangential-history coupling, face/hosel
modes, independent time/mesh/mode/event convergence and measured impact and
acoustic qualification remain required. The coefficient values are synthetic.
