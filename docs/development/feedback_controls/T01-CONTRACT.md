# T01 replay contract implementation

Tools #5461 adds `sidekick.lab.mocap.ExperimentReplayBundle`, a strict,
versioned interchange record for immutable model/provider identity, complete
named model-native initial state, ordered inputs, the simulation-relative time
grid, and the exact replay policy. Its wire version is
`experiment-replay/1.0.0`; the JSON Schema is
`schemas/mocap/experiment-replay-v1.schema.json`.

`ActuationInputKind` distinguishes actuator commands, actuator torque, actuator
force, generalized effort, muscle excitation, muscle activation, and external
loads. Torque, force, and normalized muscle channels have separate unit checks;
muscle excitation/activation also require a muscle-activation state in the
model-authored state schema. Position and velocity are named state components,
with native representation strings and independent dimensions, so the
contract does not assume `nq == nv`. State values must match the complete,
ordered state schema. The v1 input clock is explicitly `simulation_relative`
seconds from zero; actuator-force and external-load channels require an
explicit `frame_id`.

Each history records the exact declared injection boundary. A held actuator
command is not evidence that a state-dependent downstream force or torque was
held constant; use `actuator_torque` or `actuator_force` when those efforts are
the values being replayed.

The bundle binds model identity, state-schema identity, initial-state bytes,
input history, time grid, and execution policy with SHA-256 digests, exposed as
`model_identity_sha256`, `state_schema_sha256`,
`capability_declarations_sha256`, `input_channel_schema_sha256`,
`initial_state_sha256`, `applied_input_sha256`, `time_grid_sha256`, and
`execution_policy_sha256` (`policy_sha256` on the
bundle). The channel-schema digest binds ordered target, unit, coordinate, and
frame mappings. The source-model digest and optional
`loaded_native_model_sha256` remain separate. The
policy records solver/version, integration/step policy, native initialization
policy, time-only input player, replay mode, contact identity, and external-load
identity. Observation access, state feedback, and state resets are rejected.
The time-only player may be implemented by a controller object; its name or
class does not determine whether replay is closed-loop.

Required status, declared support, and runtime availability are independent
fields, and their ordered declarations are integrity-bound. The bundle reports
required capabilities that are unsupported or unavailable as blockers but has
no qualification or certification status. Native-own-contact replay, externally forced replay,
and shared rigid-body emulation are distinct modes. A transport digest proves
payload integrity, not that an engine executed the declared policy or produced
physically qualified behavior; UpstreamDrift owns those physics and gate checks.

The synthetic contract tests are independent of private mocap recordings and
cover serialization, mapping order, required muscle state, units, policy
tampering, time-grid/state-schema digest exposure, and strict JSON Schema
validation.
