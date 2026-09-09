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
- #4716 / TOOLS-M9: C3D biomechanical data exchange (active branch).

## Current branch

- Branch: `feat/4716-mocap-c3d-exchange`
- Base: `feat/4726-mocap-temporal-mapping`
- Worktree: `C:\Users\diete\Repositories\Tools`
- Pull request: Pending creation

## Delivered in this slice

- Subepic #4716 (TOOLS-M9): C3D Biomechanical Data Exchange.
- `sidekick.lab.mocap.c3d` defines:
  - `C3DHeader`: binary header parameter schemas with fail-closed bounds checking.
  - `C3DPointChannel`: 3D point trajectory channel with residual and camera masks.
  - `C3DAnalogChannel`: 1D analog channel stream with scaling and offset metadata.
  - `C3DForcePlatform`: force platform geometry and type 2 channel mapping.
  - `C3DContainer`: in-memory container for points, analogs, events, and parameters.
  - `serialize_c3d_header` & `write_c3d_file`: deterministic C3D binary writer with golden round-trip serialization.
  - `parse_c3d_header` & `validate_c3d_header_magic`: header parser with magic byte validation.
  - `unit_scale_factor`: metric and imperial length scaling factors.
  - `compute_center_of_pressure`: ground reaction force and moment COP derivation with contact thresholding.
- Contract test suites in `tests/shared/python/sidekick/lab/mocap/test_c3d_contracts.py`.

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
