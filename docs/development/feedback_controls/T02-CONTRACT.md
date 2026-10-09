# T02 Early Evidence Interchange Contract

Tools #5462 adds `ComparisonEvidenceReceipt` in
`sidekick.lab.mocap.experiment_evidence`, with wire version
`comparison-evidence/1.0.0` and JSON Schema
`schemas/mocap/comparison-evidence-v1.schema.json`. The receipt names an
authoritative comparison level, the expected model/engine rows, the coarse F01
drive mode (`torque` or `muscle_excitation`), the exact replay mode required
for each row, and which implementation/artifact evidence
must be present. It embeds the T01 replay bundle when one exists and retains
required rows with unknown or unavailable support rather than dropping them.
Each row names both `package_id` and `variant_id`, matching F01's package
inventory key; its `drive_mode` is the coarse F01 drive class.

Rows carry separate `required`, `support`, and `availability` values. Replay
drive mode remains separate from T01's finer `input_kind`: F01 `torque` admits
actuator commands, actuator torques, actuator forces, and generalized efforts;
F01 `muscle_excitation` admits only muscle excitation. Muscle activation,
external loads, and incompatible drive/input pairs are rejected as replay rows
for this version. These wire values preserve both classifications without
silently relabeling excitation or activation as torque. A row cannot satisfy a
different required replay mode; comparisons above identity cannot mix
drive classes or native-own-contact, externally-forced, and
shared-rigid-body-emulation modes.
Same-input comparisons also require identical applied-input and time-grid
digests, channel-schema, execution-policy, and state-schema digests. These are
structural admission checks, not numerical or physical acceptance.

Dynamics, contact, integrator, actuator, and restart implementation identities
are carried independently with optional digest-bound evidence references.
Private evidence references are opaque tokens only; the public payload carries
no local path or source observation bytes. Unknown and unavailable evidence
stays explicit with a reason. The receipt reports missing required rows and
references but contains no qualification status or pass/fail physics result.
UpstreamDrift F01 remains the authority for model capability, evidence
admission and scientific qualification.

This is the early schema-readiness milestone. It does not implement native
consumer integration or claim that a replay executed. Downstream acceptance
must wait for runnable UpstreamDrift F06/F07 consumers and their campaign
evidence. Synthetic public tests validate wire stability, row completeness,
replay-mode separation, evidence-status preservation, and opaque-reference
privacy without using private mocap recordings.
