# Markerless Mocap Handoff

Last updated: 2026-08-25

## Authority

Tools owns the MIT vendor-neutral markerless-mocap contracts and reference algorithms. UpstreamDrift owns orchestration and UX. AffineDrift owns evidence, validation publication, and sanitized visualization. Tools_Private is not part of the open runtime.

## Active issues

- Epic #4706: vendor-neutral acquisition, calibration, reconstruction, and C3D exchange.
- #4708 / TOOLS-M0: authority ADR and acceptance program.
- #4710 / TOOLS-M1: canonical mocap schemas. M0/M1 are under protected review in PR #4734.
- #4713, #4714, #4715, #4716, #4718, and #4721 have locally verified dependency-stacked slices; none is merged or release authority.

## Current branch

- Branch: `feat/4708-mocap-authority-schemas`
- Original base: `origin/main` at `e76a7a214`
- Latest merged base: `origin/main` at `31d28b0a0a0435cd47d05bedc61a1357d670a8d8`
- Worktree: `C:\Users\diete\Repositories\Tools-worktrees\4708-mocap-authority-schemas`
- Pull request: #4734

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

Observed before merging current `origin/main`: 26 focused tests passed; the nine mocap API modules
exactly match their hand-edited baseline entries; Ruff format/check and mypy
passed. After merging `31d28b0a0a0435cd47d05bedc61a1357d670a8d8`,
78 of 79 focused mocap plus incoming-main tests passed; the single failure is the
repository-wide API baseline's known non-mocap drift across existing API,
calculator, units, and shell surfaces. Do not regenerate the full baseline.
Full exact-HEAD export and protected CI remain mandatory before merge claims.

Consumer coordination: UpstreamDrift #9069 owns schema adoption;
Gasification_Model #4751 owns exact-Tools-SHA impact qualification.

## Do not

- Do not add a vendor SDK, model weight, FreeMoCap, or SkellyCam dependency to the MIT core.
- Do not call model-derived single-camera depth triangulated 3-D.
- Do not collapse device, trigger, host-monotonic, and UTC clocks.
- Do not introduce ambiguous transform direction or duplicate UpstreamDrift schemas.
- Do not extend C3D here until the existing reader is characterized under #4716.
- Do not change protected workflow/runner policy to force completion.
