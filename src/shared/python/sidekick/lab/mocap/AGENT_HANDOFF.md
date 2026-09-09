# Markerless Mocap Handoff

Last updated: 2026-09-08

## Authority

Tools owns the MIT vendor-neutral markerless-mocap contracts and reference algorithms. UpstreamDrift owns orchestration and UX. AffineDrift owns evidence, validation publication, and sanitized visualization. Tools_Private is not part of the open runtime.

## Active issues

- Epic #4706: vendor-neutral acquisition, calibration, reconstruction, and C3D exchange.
- #4708 / TOOLS-M0: authority ADR and acceptance program (merged in #4734).
- #4710 / TOOLS-M1: canonical mocap schemas (merged in #4734).
- #4713 / TOOLS-M2: camera acquisition protocol (merged in #5056).
- #4718 / TOOLS-M3: synchronization and recording (merged in #5059).
- #4714 / TOOLS-M4: intrinsic calibration (merged in #5064).
- #4721 / TOOLS-M5: extrinsic calibration (merged in #5066).
- #4715 / TOOLS-M6: pose backend adapters (merged in #5076).
- #4724 / TOOLS-M7: association and N-view reconstruction (merged in #5111).
- #4726 / TOOLS-M8: temporal reconstruction and biomechanical mapping (active PR).
- #4716 / TOOLS-M9 remains queued on this delivery.

## Current branch

- Branch: `feat/4726-mocap-temporal-mapping`
- Base: `origin/main` at `f9a5d736e`
- Worktree: `C:\Users\diete\Repositories\Tools`
- Pull request: Pending creation

## Delivered in this slice

- Subepic #4726 (TOOLS-M8): Temporal reconstruction and biomechanical mapping.
- `sidekick.lab.mocap.temporal` defines:
  - `GapPolicy`: policy enum (`REJECT`, `LINEAR_INTERPOLATE`, `HOLD_PREVIOUS`, `DROP`).
  - `ButterworthFilter`: zero-phase forward-backward IIR Butterworth low-pass filter.
  - `SavitzkyGolayFilter`: polynomial least-squares convolution filter for smoothing and derivative estimation.
  - `fill_trajectory_gaps`: gap handling with length-bounded interpolation and explicit `GapPolicy`.
  - `compute_kinematic_derivatives`: numerical velocity and acceleration derivation with covariance propagation.
  - `apply_segment_length_constraint`: bone-length invariance enforcement preserving anatomical rigidity.
  - `apply_joint_angle_constraint`: physiological range-of-motion bounding for hinge and spherical joints.
  - `landmarks_to_delivery_trajectory`: canonical adapter mapping filtered landmark streams to `DeliveryTrajectory` (`swing_sim.delivery_trajectory/1`).
- Contract test suites in `tests/shared/python/sidekick/lab/mocap/test_temporal_contracts.py`.

## Required gates

```powershell
python -m pytest tests/shared/python/sidekick/lab/mocap tests/architecture/test_mocap_authority_program.py -q
python -m pytest tests/test_sidekick_public_api_stability.py -q
python -m ruff format --check src/shared/python/sidekick/lab/mocap
python -m ruff check src/shared/python/sidekick/lab/mocap
python -m mypy src/shared/python/sidekick/lab/mocap
```

Consumer coordination: UpstreamDrift #9069 owns schema, acquisition, sync, calibration, adapter, and reconstruction adoption;
Gasification_Model #4751 owns exact-Tools-SHA impact qualification.

## Do not

- Do not add a vendor SDK, model weight, FreeMoCap, or SkellyCam dependency to the MIT core.
- Do not call model-derived single-camera depth triangulated 3-D.
- Do not collapse device, trigger, host-monotonic, and UTC clocks.
- Do not introduce ambiguous transform direction or duplicate UpstreamDrift schemas.
- Do not extend C3D here until the existing reader is characterized under #4716.
- Do not change protected workflow/runner policy to force completion.
