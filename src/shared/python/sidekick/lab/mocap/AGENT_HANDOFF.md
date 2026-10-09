# Markerless Mocap Handoff

Last updated: 2026-10-08

## Authority

Tools owns the MIT vendor-neutral markerless-mocap contracts and reference algorithms. UpstreamDrift owns orchestration and UX. AffineDrift owns evidence, validation publication, and sanitized visualization. Tools_Private is not part of the open runtime.

## Feedback-Control Replay Contract — #5461

- T01 introduces `ExperimentReplayBundle`, wire version `experiment-replay/1.0.0`, and `schemas/mocap/experiment-replay-v1.schema.json`.
- Identity/state types and drive/replay enums live in `experiment_contracts.py`; input, time-grid, and execution-policy validation lives in `experiment_execution.py`; bundle validation and digests live in `experiment_replay.py`; strict JSON lives in `replay_serialization.py`; public exports are in `__init__.py`.
- Inputs distinguish command, actuator torque/force, generalized effort, muscle excitation/activation, and external load. Named complete initial state supports native manifold dimensions without assuming equal position and velocity dimensions.
- Integrity digests bind model identity, state schema, complete state values, input history/time grid, and execution policy. Source-model and optional loaded-native-model digests are distinct.
- Required capability and availability are explicit; the transport has no qualification status. The consumer owns physics qualification and verifies the declared execution against its engine.
- Synthetic test: `tests/shared/python/sidekick/lab/mocap/test_experiment_replay.py`. Implementation notes: `docs/development/feedback_controls/T01-CONTRACT.md`.

## Feedback-Control Evidence Interchange — #5462

- T02 adds `ComparisonEvidenceReceipt`, wire version `comparison-evidence/1.0.0`, and `schemas/mocap/comparison-evidence-v1.schema.json`.
- Rows retain the T01 bundle when present, explicit package/variant/coarse-drive identity, required/support/availability values, implementation evidence for dynamics/contact/integrator/actuator/restart, and opaque digest-bound artifact references.
- F01's coarse drive mode is `torque` or `muscle_excitation`; T01's fine input kind remains inside the replay bundle. F01 torque accepts actuator command/torque/force or generalized effort; muscle excitation accepts only excitation. Other pairings are rejected.
- The caller supplies required model/engine row IDs and exact replay modes. Missing rows and evidence kinds remain visible; a different replay mode cannot satisfy the required mode, and non-identity comparisons cannot mix drive or replay modes.
- Structural checks do not execute a model, score measurements, decide muscle equivalence, or declare scientific qualification. UpstreamDrift F01 owns admission and qualification; native-consumer acceptance follows runnable F06/F07 and campaign evidence.
- Private source paths and observation payloads must remain in authorized private storage. Public fixtures use synthetic identities and opaque references only.

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
