# F-TL-036

**Unauthenticated POST /api/estop has no rate limit or caller record**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | security | medium     | S      | proposed |

**Locations:** `src/p1am_control_system/backend/main.py:790`, `src/p1am_control_system/backend/auth_config.py:33`

## Finding

- Severity: Low - Category: security
- Confidence: medium
- Location: src/p1am_control_system/backend/main.py:790-800; src/p1am_control_system/backend/auth_config.py:33-36
- Evidence: `@app.post("/api/estop")` with no `dependencies=`; auth_config docstring: "E-stop _activation_ … is intentionally left unauthenticated so a panic stop is always reachable".
- Failure scenario: anything on the HMI network can trip the heater relay in a loop and prevent restart; defensible, but should carry caller IP in the audit event, a per-source cooldown, and a documented network-segmentation assumption.
- Proposed change: keep unauthenticated; log `request.client.host`; 1/s per-source limit; note in SECURITY.md.
- Effort: S

## Draft Pull Request

**Title:** `feat(p1am): record caller IP and rate-limit the unauthenticated e-stop`

### Why This Matters (The Pitch)

The unauthenticated e-stop is a documented safety choice, and it is also a lever anyone on the HMI network can pull in a loop. Keeping it open while logging request.client.host, adding a one-per-second per-source cooldown and writing the network-segmentation assumption into SECURITY.md preserves the panic stop and adds an audit trail.

### Proposed PR Body

## Problem

`src/p1am_control_system/backend/main.py:790-800` exposes `@app.post("/api/estop")` with no `dependencies=`; `auth_config.py:33-36` documents that e-stop activation is intentionally unauthenticated.

## Change

- [ ] Keep the endpoint unauthenticated
- [ ] Log `request.client.host` in the audit event
- [ ] Add a 1/s per-source rate limit
- [ ] Document the network-segmentation assumption in SECURITY.md

## Verification

Test that the audit event includes the source address and that a second request within 1 s from one source is rate-limited.

## Risk and Rollback

A rate limit on a safety endpoint must never block the first stop; the limit applies only to repeats per source. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-036

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
