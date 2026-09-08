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
- #4714 / TOOLS-M4: intrinsic calibration (merged in #5064).
- #4721 / TOOLS-M5: extrinsic calibration (merged in #5066).
- #4715 / TOOLS-M6: pose backend adapters (current branch).
- #4716 and #4724 have dependency-stacked slices; none is merged or release authority.

## Current branch

- Branch: `feat/4715-mocap-pose-adapters`
- Base: `origin/main` at `f583d26fc`
- Worktree: `C:\Users\diete\Repositories\Tools`
- Pull request: Pending creation

## Delivered in this slice

- Subepic #4715 (TOOLS-M6): Pose backend adapters.
- `sidekick.lab.mocap.adapters` defines:
  - `ProviderLicenseManifest`, `LicenseRecord`: separate fail-closed authority across 5 asset categories.
  - `LicenseCategory`, `PermittedUse`, `ApprovalStatus`, `LicenseEvaluationResult`: typed licensing outcomes.
  - `KeypointMapping`, `SkeletonConverter`: map backend detections to canonical `PixelObservation` records.
  - Built-in canonical skeletons: `mediapipe-pose-33-v1` and `coco-17-v1`.
  - `PoseBackendProtocol`: abstract base protocol for pose inference providers.
  - `MediaPipePoseAdapter`: fail-closed wrapper returning `UNAVAILABLE_BACKEND` when uninstalled.
  - `ExternalServicePoseAdapter`: process-separated external mocap service adapter.
  - `SyntheticPoseAdapter`: deterministic zero-dependency test provider.
- Unit and contract test suites in `tests/shared/python/sidekick/lab/mocap/`.

## Required gates

```powershell
python -m pytest tests/shared/python/sidekick/lab/mocap tests/architecture/test_mocap_authority_program.py -q
python -m pytest tests/test_sidekick_public_api_stability.py -q
python -m ruff format --check <changed-python-files>
python -m ruff check <changed-python-files>
python -m mypy <changed-python-files>
```

Consumer coordination: UpstreamDrift #9069 owns schema, acquisition, sync, calibration, and adapter adoption;
Gasification_Model #4751 owns exact-Tools-SHA impact qualification.

## Do not

- Do not add a vendor SDK, model weight, FreeMoCap, or SkellyCam dependency to the MIT core.
- Do not call model-derived single-camera depth triangulated 3-D.
- Do not collapse device, trigger, host-monotonic, and UTC clocks.
- Do not introduce ambiguous transform direction or duplicate UpstreamDrift schemas.
- Do not extend C3D here until the existing reader is characterized under #4716.
- Do not change protected workflow/runner policy to force completion.
