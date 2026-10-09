# Shared Controlled-Matching Contract Design

## Scope and Authority

Governing epic: https://github.com/D-sorganization/Tools/issues/5460. Domain design and primary-source review: [UpstreamDrift #11784](https://github.com/D-sorganization/UpstreamDrift/issues/11784), documentation packet `docs/development/feedback_controls/DESIGN.md`. Reference that authority rather than duplicate control equations here. The Tools transport contract is implemented under T01; it does not qualify a solver or model.

## Public Boundaries

Extend existing `sidekick.lab.mocap` and mocap schemas for sessions, identities, units/frames/clocks, calibration and observations. General control/replay contracts carry actuator-space identity, torque versus excitation semantics, full initial state, ordered mappings, interval/interpolation policy, model/provider/contact/solver provenance, capability and qualification evidence. UpstreamDrift owns golf tasks, optimization, muscle physiology, acceptance and engine-specific implementation. Shared downstream contracts must remain backward compatible and consume pinned provider revisions.

## Input and Capability Invariants

Reject non-finite/unsorted times, unknown units/frames, mixed clock domains, incompatible actuator dimensions/order, missing required muscle states, unsupported drive modes and stale provenance. Torque ZOH matches the #11605 interval policy; muscle excitation uses an explicitly versioned policy and activation/tendon dynamics. Contacts/reaction forces are not actuator inputs. Reserve/root drives are declared channels. Different muscle definitions permit only bounded biomechanical equivalence, not false excitation equality.

## Evidence and Privacy

Retain separate statuses for IK, feedback tracking, torque replay, excitation replay, externally forced replay and autonomous model-contact replay. T02's versioned evidence receipt preserves replay modes, required/support/availability axes, missing rows and digest-bound opaque references. Structural readiness never upgrades a result to qualified; UpstreamDrift owns scientific admission. Use independently generated synthetic fixtures in public tests. Private paths, original identities/source hashes and identifying videos remain in authorized private storage; release-safe summaries require the existing review boundary.

## Resources and Preview Interchange

Represent source/provider-complete cache keys, job/resume identity, cancellation, disk/memory/worker budgets, cold/warm timings, host/license identity and preview metadata. Preview root is a configuration value, with this program using `%USERPROFILE%\Desktop\Motion_Matching_Previews`. No hard-coded runtime endpoint or host assumption is required. Tailscale execution requires availability/access/capacity checks and output-hash verification. Cleanup eligibility requires task ownership, verified merge, clean tree and preserved evidence; source data and user/unmerged work are excluded.

## Testing, Documentation and Migration

TDD contract tests use invalid permutations/timebases, missing muscle states, saturation provenance and false-qualified fixtures; stable public API, schema migration and pinned UpstreamDrift integration are required. DbC uses descriptive errors; LoD routes through small facades; DRY keeps one schema/conversion authority. Check other consumers when shared surfaces change. Update governed manuals/registries when implementation changes public interchange, and use change fragments plus the PR handoff. The implementation plan and turnover record dependencies, first bounded task and known limits.

## Review Clarifications

Adopt the existing canonical model/variant/capability registry and stable engine/variant/drive-mode keys; F01 freezes baseline/policy and F09 executes ongoing conformance. #11605/#11607 retain replay authority. Tools validates generic evidence interchange; UpstreamDrift decides physics and qualification. Replay APIs accept a frozen bundle and a time-only input player with the native plant; observation/state-feedback callbacks and measured-state resets are forbidden, regardless of player class name. Bind the model-native complete initial state, ordered input history and time grid, and executed integration/initialization/contact policy. F07 supplies the complete pinned muscle-model/state/contact/forward-API handoff to F08. Optional NMPC needs a report/disposition and does not block acceptance of a qualified simpler controller. Public preview manifests omit local paths; resolve configured artifacts through MOTION_MATCHING_PREVIEW_ROOT, and preserve shared/user-owned directories and videos.
