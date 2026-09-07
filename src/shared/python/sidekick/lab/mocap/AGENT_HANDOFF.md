# Markerless Mocap Handoff

Last updated: 2026-09-07

## Authority

Tools owns the MIT vendor-neutral markerless-mocap contracts and reference algorithms. UpstreamDrift owns orchestration and UX. AffineDrift owns evidence, validation publication, and sanitized visualization. Tools_Private is not part of the open runtime.

## Active issues

- Epic #4706: vendor-neutral acquisition, calibration, reconstruction, and C3D exchange.
- #4708 / TOOLS-M0: authority ADR and acceptance program (merged in #4734).
- #4710 / TOOLS-M1: canonical mocap schemas (merged in #4734).
- #4713 / TOOLS-M2: camera acquisition protocol (current branch).
- #4714, #4715, #4716, #4718, and #4721 have dependency-stacked slices; none is merged or release authority.

## Current branch

- Branch: `feat/4713-camera-acquisition-protocol`
- Base: `origin/main` at `c9fccd4ac`
- Worktree: `C:\Users\diete\Repositories\Tools`
- Pull request: Pending creation

## Delivered in this slice

- Subepic #4713 (TOOLS-M2): Camera Acquisition Protocol.
- `sidekick.lab.mocap.acquisition` defines:
  - `SourceState`, `DropPolicy`, `FramePacket`, `FrameSource`, `CaptureGroup`.
  - Synthetic and prerecorded reference drivers (`SyntheticFrameSource`, `PrerecordedFrameSource`).
  - Strict bounded queue backpressure, fail-closed handling, and frame drop policies.
- Unit and property contract tests in `tests/shared/python/sidekick/lab/mocap/test_acquisition_contracts.py`.

## Required gates

```powershell
python -m pytest tests/shared/python/sidekick/lab/mocap tests/architecture/test_mocap_authority_program.py -q
python -m pytest tests/test_sidekick_public_api_stability.py -q
python -m ruff format --check <changed-python-files>
python -m ruff check <changed-python-files>
python -m mypy <changed-python-files>
```

Consumer coordination: UpstreamDrift #9069 owns schema and acquisition adoption;
Gasification_Model #4751 owns exact-Tools-SHA impact qualification.

## Do not

- Do not add a vendor SDK, model weight, FreeMoCap, or SkellyCam dependency to the MIT core.
- Do not call model-derived single-camera depth triangulated 3-D.
- Do not collapse device, trigger, host-monotonic, and UTC clocks.
- Do not introduce ambiguous transform direction or duplicate UpstreamDrift schemas.
- Do not extend C3D here until the existing reader is characterized under #4716.
- Do not change protected workflow/runner policy to force completion.
