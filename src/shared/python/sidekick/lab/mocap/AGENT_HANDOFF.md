# Markerless Mocap Handoff

Last updated: 2026-09-09

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
- #4726 / TOOLS-M8: temporal reconstruction and biomechanical mapping (PR #5118).
- #4716 / TOOLS-M9: C3D biomechanical data exchange (merged in #5119).
- #4727 / TOOLS-M10: reference CLI and service (active branch).

## Current branch

- Branch: `feat/4727-mocap-cli-service`
- Base: `feat/4716-mocap-c3d-exchange`
- Worktree: `C:\Users\diete\Repositories\Tools`
- Pull request: Pending creation

## Delivered in this slice

- Subepic #4727 (TOOLS-M10): Reference CLI and Service.
- `sidekick.lab.mocap.service`:
  - `MocapService`: headless orchestration protocol for discover, capture, calibrate, reconstruct, and export workflows.
  - `MocapServiceStatus`, `MocapHealthReport`, `MocapCapabilitiesReport`: structured health, capabilities, and lifecycle states.
  - `cancel`: graceful task cancellation without resource leaks.
  - `no-store` policy: enforces privacy rules preventing raw frame/video persistence in ephemeral sessions.
- `sidekick.lab.mocap.cli`:
  - Headless CLI subcommands: `discover`, `capture`, `calibrate`, `reconstruct`, and `export`.
  - Structured `--json` and human-readable output formatting.
  - Deterministic exit codes (0 for success, 1 for errors, 2 for arg errors, 130 for SIGINT).
- Contract test suite: `tests/shared/python/sidekick/lab/mocap/test_cli_service_contracts.py` (12 passed).

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
- Do not change protected workflow/runner policy to force completion.
