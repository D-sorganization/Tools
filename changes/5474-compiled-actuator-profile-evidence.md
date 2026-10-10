---
issue: 5474
summary: "Require matching versioned actuator-profile references for mixed command evidence"
dl_state: "in_review"
next_step: "Resolve and verify profile bytes against native execution in UpstreamDrift #11955"
branch: "feat/5474-actuator-profile-reference-admission"
---

T02 keeps `comparison-evidence/1.0.0` and existing ACTUATOR evidence/artifact fields. A muscle-drive row with T01 `ACTUATOR_COMMAND` now requires a matching opaque profile implementation reference and ACTUATOR artifact digest. This structural receipt does not resolve the bytes or certify native model semantics; UpstreamDrift #11955 owns that check.
