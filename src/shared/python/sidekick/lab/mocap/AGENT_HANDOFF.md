# Markerless Mocap Handoff

Last updated: 2026-09-07

## Authority

Tools owns the MIT vendor-neutral markerless-mocap contracts and reference algorithms. UpstreamDrift owns orchestration and UX. AffineDrift owns evidence, validation publication, and sanitized visualization. Tools_Private is not part of the open runtime.

## Active issues

- Epic #4706: vendor-neutral acquisition, calibration, reconstruction, and C3D exchange.
- #4708 / TOOLS-M0: authority ADR and acceptance program (merged in #4734).
- #4710 / TOOLS-M1: canonical mocap schemas (merged in #4734).
- #4713 / TOOLS-M2: camera acquisition protocol (merged in #5056).
- #4718 / TOOLS-M3: synchronization and recording (merged in #5059).
- #4714 / TOOLS-M4: intrinsic calibration (current branch).
- #4715, #4716, and #4721 have dependency-stacked slices; none is merged or release authority.

## Current branch

- Branch: `feat/4714-mocap-intrinsic-calibration`
- Base: `origin/main` at `8c81c0bd7`
- Worktree: `C:\Users\diete\Repositories\Tools`
- Pull request: Pending creation

## Delivered in this slice

- Subepic #4714 (TOOLS-M4): Intrinsic calibration.
- `sidekick.lab.mocap.calibration` defines:
  - `DistortionModel`, `DistortionCoefficients`: Brown-Conrady, rational, and Kannala-Brandt models.
  - `PinholeIntrinsics`, `FisheyeIntrinsics`: forward projection and inverse unprojection ray mapping.
  - `CalibrationTarget`, `CalibrationObservation`: provenance and feature mapping for calibration frames.
  - `ReprojectionResidual`, `IntrinsicCalibrationResult`: residual accounting and covariance.
  - `CalibrationQuality`, `CalibrationDegeneracyKind`: qualification floors and degeneracy detection.
  - Integration with `CameraIdentity.stable_key`.
- Unit and contract test suites in `tests/shared/python/sidekick/lab/mocap/`.

## Required gates

```powershell
python -m pytest tests/shared/python/sidekick/lab/mocap tests/architecture/test_mocap_authority_program.py -q
python -m pytest tests/test_sidekick_public_api_stability.py -q
python -m ruff format --check <changed-python-files>
python -m ruff check <changed-python-files>
python -m mypy <changed-python-files>
```

Consumer coordination: UpstreamDrift #9069 owns schema, acquisition, sync, and calibration adoption;
Gasification_Model #4751 owns exact-Tools-SHA impact qualification.

## Do not

- Do not add a vendor SDK, model weight, FreeMoCap, or SkellyCam dependency to the MIT core.
- Do not call model-derived single-camera depth triangulated 3-D.
- Do not collapse device, trigger, host-monotonic, and UTC clocks.
- Do not introduce ambiguous transform direction or duplicate UpstreamDrift schemas.
- Do not extend C3D here until the existing reader is characterized under #4716.
- Do not change protected workflow/runner policy to force completion.
