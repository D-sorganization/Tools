# Implementation Plan

Governing epic: https://github.com/D-sorganization/Tools/issues/5460

Implementation and qualification are separate. T01 is implementing the transport contract; it does not certify physics. IDs and dependencies resolve to real issue URLs; existing MOSAIC and parity work remain authoritative.

The detailed milestone DAG, including early schema readiness and downstream acceptance, is recorded in [milestone_dependencies.json](milestone_dependencies.json) and follows [ASTRA_REVIEW.md](ASTRA_REVIEW.md).

| Task | Deliverable | Dependencies |
| --- | --- | --- |
| [T01](https://github.com/D-sorganization/Tools/issues/5461) | Versioned full-state/input-boundary replay contract with executed-policy and payload integrity | Implementing; coordinate UpstreamDrift [F01](https://github.com/D-sorganization/UpstreamDrift/issues/11785) and #11607 |
| [T02](https://github.com/D-sorganization/Tools/issues/5462) | Exchange parity and qualification evidence without false equivalence | Early `comparison-evidence/1.0.0` schema readiness after [T01](https://github.com/D-sorganization/Tools/issues/5461) + UpstreamDrift [F01](https://github.com/D-sorganization/UpstreamDrift/issues/11785); later native-consumer acceptance after [F06](https://github.com/D-sorganization/UpstreamDrift/issues/11790) and [F07](https://github.com/D-sorganization/UpstreamDrift/issues/11791). See [T02-CONTRACT.md](T02-CONTRACT.md). |
| [T03](https://github.com/D-sorganization/Tools/issues/5463) | Provide Bounded Experiment Resources and Privacy-Aware Preview Manifests | [T01](https://github.com/D-sorganization/Tools/issues/5461), [T02](https://github.com/D-sorganization/Tools/issues/5462), and the frozen UpstreamDrift [F01](https://github.com/D-sorganization/UpstreamDrift/issues/11785) resource/preview contract; F09 is downstream acceptance, not an implementation prerequisite |

## Engineering and Evidence Contract

- TDD: record a meaningful failing behavioral test, minimal green implementation and refactor; use independent analytic/synthetic truth and native integration where physics is claimed.
- DbC: validate finite states, units/frames/clocks, manifold and actuator dimensions, bounds, capability and provenance. Use existing public facades (LoD) and authoritative shared kernels/contracts (DRY).
- Update design calculations/manual QMD and registries when applicable, SPEC/change fragment and canonical turnover in the implementation PR. Include exact commands, provider versions, failures and receipt evidence.
- Preserve existing gates; new tolerances must be ratified from noise/baseline/convergence evidence before fitting. No weakened acceptance to rescue a result.
- Respect private-data boundaries. All models/engines remain represented in the capability/parity matrix, with unavailable or unqualified states explicit. No simulation result is implied by this issue.
- Replay bundles name the actual injection boundary (command, actuator force/torque, generalized effort, excitation/activation, or external load), carry the complete model-native initial state and exact ordered input/time payload, and hash the executed integration/initialization/contact policy. ZOH command semantics do not imply held physical actuator effort.
- The replay player may be a native time-only prescribed-input controller. Independent replay forbids observation/state-feedback callbacks and measured-state resets; object/class naming alone does not determine whether replay is open-loop.
- Apply the binding clarifications in [ASTRA_REVIEW.md](ASTRA_REVIEW.md), especially the distinction among native-own-contact, externally forced, and shared-rigid-body-emulation evidence.

## Review Refinements

**T02:** Separate two milestones to keep the protocol acyclic. The T02 early schema-readiness implementation is on a feature branch stacked on T01 and aligned to the frozen F01 vocabulary; it can unblock private D02 evidence splitting once prerequisites merge. It validates generic interchange structure and evidence references only. T02 remains open for downstream native-consumer acceptance after F06/F07, which exercises actual engine use. Evidence status is data: Tools does not certify physical equivalence, define model capabilities or decide scientific qualification. UpstreamDrift owns those decisions and versioned gate authority. Private D04 supplies evidence to F10; it does not depend on F10.

**T03:** Public preview manifests omit local filesystem paths and resolve artifacts through a configured root (MOTION_MATCHING_PREVIEW_ROOT). Worktree/cache cleanup never authorizes deletion of videos or source captures based on age or merge alone; preserve shared/user-owned directories.

## Planning Delivery

[T00: Publish This Planning Packet](https://github.com/D-sorganization/Tools/issues/5464) is the documentation-only delivery child. Its PR may close that child; all implementation/qualification children and the parent epic remain open.
