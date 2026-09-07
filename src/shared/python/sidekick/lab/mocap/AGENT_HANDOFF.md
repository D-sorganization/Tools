# Markerless Mocap Handoff

Last updated: 2026-09-07

## Authority

Tools owns the MIT vendor-neutral markerless-mocap contracts and reference algorithms. UpstreamDrift owns orchestration and UX. AffineDrift owns evidence, validation publication, and sanitized visualization. Tools_Private is not part of the open runtime.

## Active issues

- Epic #4706: vendor-neutral acquisition, calibration, reconstruction, and C3D exchange.
- #4708 / TOOLS-M0: authority ADR and acceptance program (merged in #4734).
- #4710 / TOOLS-M1: canonical mocap schemas (merged in #4734).
- #4713 / TOOLS-M2: camera acquisition protocol (merged in #5056).
- #4718 / TOOLS-M3: synchronization and recording (current branch).
- #4714, #4715, #4716, and #4721 have dependency-stacked slices; none is merged or release authority.

## Current branch

- Branch: `feat/4718-mocap-synchronization-and-recording`
- Base: `origin/main` at `c7eca91c9`
- Worktree: `C:\Users\diete\Repositories\Tools`
- Pull request: Pending creation

## Delivered in this slice

- Subepic #4718 (TOOLS-M3): Synchronization and recording.
- `sidekick.lab.mocap.sync` defines:
  - `SyncQuality`, `SyncAnomalyType`, `SyncAnomaly`, `ClockSkewEstimate`, `SyncMonitor`.
  - Inter-camera clock skew, drift, and jitter bounds with fail-closed rejection.
  - Frame anomaly detection (dropped, duplicate, out-of-order sequence numbers).
- `sidekick.lab.mocap.recording` defines:
  - `FrameIndexEntry`, `RecordingIntegrityReport`, `RecordingWriter`, `RecordingReader`.
  - Append-only chunked frame streaming with CRC32 integrity verification.
  - Crash-safe atomic manifest updates using temporary file replacement.
  - Strict enforcement of `RecordingPolicy` (rejection of raw image storage under `no_store`).
- Unit and contract test suites in `tests/shared/python/sidekick/lab/mocap/`.

## Required gates

```powershell
python -m pytest tests/shared/python/sidekick/lab/mocap tests/architecture/test_mocap_authority_program.py -q
python -m pytest tests/test_sidekick_public_api_stability.py -q
python -m ruff format --check <changed-python-files>
python -m ruff check <changed-python-files>
python -m mypy <changed-python-files>
```

Consumer coordination: UpstreamDrift #9069 owns schema, acquisition, and sync adoption;
Gasification_Model #4751 owns exact-Tools-SHA impact qualification.

## Do not

- Do not add a vendor SDK, model weight, FreeMoCap, or SkellyCam dependency to the MIT core.
- Do not call model-derived single-camera depth triangulated 3-D.
- Do not collapse device, trigger, host-monotonic, and UTC clocks.
- Do not introduce ambiguous transform direction or duplicate UpstreamDrift schemas.
- Do not extend C3D here until the existing reader is characterized under #4716.
- Do not change protected workflow/runner policy to force completion.
