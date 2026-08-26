# Markerless Mocap Handoff

Last updated: 2026-08-25

## Authority

Tools owns the MIT vendor-neutral markerless-mocap contracts and reference algorithms. UpstreamDrift owns orchestration and UX. AffineDrift owns evidence, validation publication, and sanitized visualization. Tools_Private is not part of the open runtime.

## Active issues

- Epic #4706: vendor-neutral acquisition, calibration, reconstruction, and C3D exchange.
- #4708 / TOOLS-M0: authority ADR and acceptance program.
- #4710 / TOOLS-M1: canonical mocap schemas.
- Next: #4713 acquisition protocol and #4714 intrinsic calibration may start after #4710 merges.

## Current branch

- Branch: `feat/4708-mocap-authority-schemas`
- Base: `origin/main` at `e76a7a214`
- Worktree: `C:\Users\diete\Repositories\Tools-worktrees\4708-mocap-authority-schemas`

## Delivered in this slice

- ADR-007 records cross-repository, coordinate/time, evidence, C3D, privacy, and licensing authority.
- The acceptance program defines unit through physical/release gates.
- `sidekick.lab.mocap` establishes frozen, DbC-validated identity, capability, clock, frame, coordinate, transform, skeleton, 2-D/3-D observation, provenance, policy, and session records.
- Strict `mocap-session/1.0.0` JSON Schema, golden fixture, canonical serializer, and fail-closed loader are under test.

## Required gates

```powershell
python -m pytest tests/shared/python/sidekick/lab/mocap tests/architecture/test_mocap_authority_program.py -q
python -m pytest tests/test_sidekick_public_api_stability.py -q
python -m ruff format --check <changed-python-files>
python -m ruff check <changed-python-files>
python -m mypy <changed-python-files>
```

Observed before the documentation-only rebase at `2c86e9168`: 26 focused tests passed; the nine mocap API modules
exactly match their hand-edited baseline entries; Ruff format/check and mypy
passed. The full suite reached 11,012 passed, 73 failed, and 8 errors before an
xdist worker-replacement internal error. Failures were distributed across
pre-existing subsystems; the repository-wide API test includes known non-mocap
drift. Do not regenerate the full baseline. Full exact-HEAD export and protected
CI remain mandatory before merge claims.

Consumer coordination: UpstreamDrift #9069 owns schema adoption;
Gasification_Model #4751 owns exact-Tools-SHA impact qualification.

## Do not

- Do not add a vendor SDK, model weight, FreeMoCap, or SkellyCam dependency to the MIT core.
- Do not call model-derived single-camera depth triangulated 3-D.
- Do not collapse device, trigger, host-monotonic, and UTC clocks.
- Do not introduce ambiguous transform direction or duplicate UpstreamDrift schemas.
- Do not extend C3D here until the existing reader is characterized under #4716.
- Do not change protected workflow/runner policy to force completion.
