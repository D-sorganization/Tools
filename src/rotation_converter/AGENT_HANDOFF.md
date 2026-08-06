# AGENT_HANDOFF — rotation_converter

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-04

## Where This Tool Is Headed

Comprehensive rotation and rigid-body transform converter (quaternions,
Euler angles, rotation matrices, axis-angle, SE(3), twists, screw axes,
frame-aware transforms) with PyQt6 desktop + web implementations, built on
`modern_robotics` conventions. See `src/rotation_converter/README.md`.

No dedicated feature epic is currently open against this tool in its own
right, but it has become a **shared kinematics dependency for rate_of_closure**:
epic #4103 Phase 4 (Screw Axis) computes the instantaneous screw axis of the
moving clubhead "via the shared kinematics used by the rotation
converter / screw-axis visualizer," and PR #4119's app-integration section
explicitly adds a "screw-axis overlay (rotation_converter adapter)." Treat
public functions in `_mr_kinematics.py`, `twist_screw.py`, and
`screw_visualization.py` as having a downstream consumer inside this same
repo now, not just UpstreamDrift.

## Recent Activity (grounding — `git log --oneline -15 -- src/rotation_converter`)

Recent history is accessibility/consolidation maintenance: Palette
focus-visible states (#3967), a batched agent/dependabot consolidation
(#3912), DbC/contract cleanup for the plugin manager and robotics module
(#3825), `modern_robotics` O-safe contract hardening (#3817), a 20-PR
fleet-relief consolidation (#3806), and earlier import-canonicalization and
cross-repo contract fixes. No open PRs currently target this tool directly
other than #4119's screw-axis adapter usage.

## Must-Read Architecture Pointers

1. `src/rotation_converter/README.md` — feature/surface inventory (PyQt6
   implemented, web implemented, 20 Python files, 4 test files).
2. `src/rotation_converter/_mr_kinematics.py`, `_mr_dynamics.py`,
   `_mr_rotation_matrices.py` — `modern_robotics`-convention kinematics core;
   the "shared kinematics" epic #4103 Phase 4 reuses.
3. `src/rotation_converter/twist_screw.py`, `screw_visualization.py` — screw
   axis representation and rendering; the direct dependency surface for
   rate_of_closure's ISA overlay.
4. `src/rotation_converter/_contracts.py` — DbC contract layer (recently
   hardened for O-safety, #8817/#3825 lineage).
5. `src/rotation_converter/modern_robotics_pkg/` — vendored/adapted
   `modern_robotics` package boundary.

## Gate Commands (this tool)

```bash
python3 -m pytest src/rotation_converter/tests -n auto --timeout=60
python3 -m ruff check src/rotation_converter
python3 -m mypy src/rotation_converter
cd src/rotation_converter/web && npm run test && npx tsc --noEmit
```

## Do-Not List

- Do not change the public signatures of `_mr_kinematics.py` /
  `twist_screw.py` screw-axis functions without checking `src/rate_of_closure`
  for the ISA-overlay adapter introduced by #4119 — this is now an
  in-repo contract, not just an UpstreamDrift one.
- Do not bypass `_contracts.py`'s DbC validation when adding new conversion
  entry points; the O-safe contract hardening in #3817/#3825 was deliberate.
- Do not restore the generic "Legacy boundary" shortcuts removed in #3825 —
  they were flagged as deferred P2 cleanup and closed out.

## Roadmap (ordered)

1. No dedicated feature epic open; treat as a stable kinematics provider.
   Prioritize API stability for the #4103 screw-axis adapter now that #4119
   depends on it.
2. Once #4119 merges, verify the screw-axis adapter usage doesn't need
   upstream changes here (check for any TODOs left in the adapter code).
3. Continue routine accessibility/hardening maintenance (Palette/Bolt/Sentinel
   sweeps) as they arrive — no structural changes expected otherwise.
