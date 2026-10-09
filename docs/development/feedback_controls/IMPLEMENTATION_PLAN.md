# Implementation Plan

Governing epic: https://github.com/D-sorganization/Tools/issues/5460

All items are planned and unqualified. IDs and dependencies resolve to real issue URLs; existing MOSAIC and parity work remain authoritative.

| Task | Deliverable | Dependencies |
| --- | --- | --- |
| [T01](https://github.com/D-sorganization/Tools/issues/5461) | Version Control/Replay Contracts for Torque and Muscle Actuation | Ready; coordinate UpstreamDrift [F01](https://github.com/D-sorganization/UpstreamDrift/issues/11785) and #11607 |
| [T02](https://github.com/D-sorganization/Tools/issues/5462) | Exchange Parity and Qualification Evidence Without False Equivalence | [T01](https://github.com/D-sorganization/Tools/issues/5461); UpstreamDrift [F01](https://github.com/D-sorganization/UpstreamDrift/issues/11785)/[F06](https://github.com/D-sorganization/UpstreamDrift/issues/11790)/[F07](https://github.com/D-sorganization/UpstreamDrift/issues/11791) |
| [T03](https://github.com/D-sorganization/Tools/issues/5463) | Provide Bounded Experiment Resources and Privacy-Aware Preview Manifests | [T01](https://github.com/D-sorganization/Tools/issues/5461), [T02](https://github.com/D-sorganization/Tools/issues/5462); UpstreamDrift [F09](https://github.com/D-sorganization/UpstreamDrift/issues/11793) |

## Engineering and Evidence Contract

- TDD: record a meaningful failing behavioral test, minimal green implementation and refactor; use independent analytic/synthetic truth and native integration where physics is claimed.
- DbC: validate finite states, units/frames/clocks, manifold and actuator dimensions, bounds, capability and provenance. Use existing public facades (LoD) and authoritative shared kernels/contracts (DRY).
- Update design calculations/manual QMD and registries when applicable, SPEC/change fragment and canonical turnover in the implementation PR. Include exact commands, provider versions, failures and receipt evidence.
- Preserve existing gates; new tolerances must be ratified from noise/baseline/convergence evidence before fitting. No weakened acceptance to rescue a result.
- Respect private-data boundaries. All models/engines remain represented in the capability/parity matrix, with unavailable or unqualified states explicit. No simulation result is implied by this issue.

## Review Refinements

**T02:** Tools validates generic interchange structure, mappings and evidence references only. Evidence status is data: Tools does not certify physical equivalence, define model capabilities or decide scientific qualification. UpstreamDrift owns those decisions and versioned gate authority.

**T03:** Public preview manifests omit local filesystem paths and resolve artifacts through a configured root (MOTION_MATCHING_PREVIEW_ROOT). Worktree/cache cleanup never authorizes deletion of videos or source captures based on age or merge alone; preserve shared/user-owned directories.

## Planning Delivery

[T00: Publish This Planning Packet](https://github.com/D-sorganization/Tools/issues/5464) is the documentation-only delivery child. Its PR may close that child; all implementation/qualification children and the parent epic remain open.
