# STATUS.md — Tools

> Updated by Project Steward. Reflects state as of **2026-09-23**.
> This is a current-state snapshot; history lives in git and SPEC.md §12.

## What Moved (since 2026-09-10)

| Area | What shipped | PR |
| ---- | ------------ | -- |
| Release | Bumped to **v1.20.0** | #5283 |
| rate_of_closure | Workspace view compositor (impact, swing, flight) | #5264 |
| rate_of_closure | Native file commands and workspace persistence | #5258 |
| rate_of_closure | Renderer source provenance restored | #5304 |
| rate_of_closure | Stale groundRegionalExecution.ts.orig removed | #5295 |
| rate_of_closure | Content-based visual baseline gate and re-baseline | #5269 |
| Flight | Reconciled Rust and Penner lift models, documented wind | #5265 |
| Variation | Sobol indices, Spearman significance, landing ellipse normality | #5266 |
| Wedge UI | Delivery metrics cards, linear waterfall, 3D visualization | #5268 |
| Conventions | Side-by-side comparison workspace and full coverage matrix | #5267 |
| Impact / grip | Grip FRF qualification requires operating-strain evidence | #5298 |
| Impact / grip | Constrained NNLS for measured-grip identification | #5300 |
| Impact / grip | Grip fixture provenance and qualification boundary | #5302 |
| Docs / planning | Deferred physical validation campaigns preserved | #5306 |
| Architecture | Historian licensing and impact-interval tab rulings ratified | #5289 |
| Accessibility | Keyboard accessibility for file import | #5280 |
| Performance | Five Bolt single-pass loop optimizations | #5279–#5294 |
| Dependencies | ruff 0.16.8, matplotlib 3.11.2, playwright 1.63.0, pypdf 6.19.0 | #5248–#5252 |

## What Is Stuck

| Item | Epic / Issue | Blocker |
| ---- | ------------ | ------- |
| **P0 security: fork-PR policy** | #4464 | No PR; requires explicit Board decision on fork approval policy for public repo with self-hosted runners |
| TOOLS-D9 completion | #4730 (#4707) | Final open subepic of engineering design manual; no active PR |
| Impact acoustics — physical evidence | #5068 | Traceable apparatus, calibration, operating strain, and participant data are unavailable; deferred to Board resource allocation |
| Ball landing / surface physical validation | #4267 | Same physical evidence gap; deferred in planning catalog |
| Renderer provenance (#5303) | #5303 | Dev-machine disk full (Errno 28); no PR yet |
| Cross-repo consumer lane (`RUNNER_CHECK_TOKEN`) | #5305 | Organization-level credential approval pending; hosted consumer suite blocked |
| Mocap reconstruction | #5111 | PR open; awaiting CI green and review |
| Qt worker losses (#5114) | #5114 | Partial cleanup on separate branch; root cause unexplained (3 workers lost after 2,880 passes) |

## What Is Next (Ordered)

1. **Merge TOOLS-D9 (#4730)** — close the final `[DOC-TOOLS]` epic #4707.
2. **Board decision on #4464 (P0)** — fork-PR policy for public repo with self-hosted runners; no code fix is possible without a policy ruling.
3. **Land mocap reconstruction PR #5111** — TOOLS-M0/M1 markerless mocap authority.
4. **Resolve disk capacity and merge #5303** — renderer provenance unblocks UpstreamDrift impact lane.
5. **Org approval for RUNNER_CHECK_TOKEN (#5305)** — re-enables private consumer contract lane in CI.
6. **Camera cluster epic #4571** — closes parent #4103 after camera delivery.

## CI Health (2026-09-23)

| Workflow | Status |
| -------- | ------ |
| CI Standard | in_progress (no failures seen on recent `main` commits) |
| Detect Secrets | success |
| Merge Hold Guard | success |
| Benchmark Suite | in_progress |
| Docs Governance | in_progress |
| Rate of Closure Web Distribution | in_progress |
| Rate Web Playwright Trusted | in_progress |
| Release Automation | queued |

No workflow failures on `main` at time of this update.

## Open PRs (2026-09-23)

| PR | Title | Notes |
| -- | ----- | ----- |
| #5310 | Palette: Add aria-live region for conversion results | spec-exempt label |
| #5308 | chore(docs): archive stale session artifacts | docs-only |

Both PRs are low-risk; neither blocks any active epic.

## Decisions Needed (Board / Dieter)

1. **P0 — Fork-PR security (#4464).** The public repo uses self-hosted runners
   with a weak fork-PR approval policy. Any non-new GitHub account can trigger
   workflow execution on personal hardware. Requires an explicit policy ruling:
   either restrict fork PR approvals, make the repo private, or accept the risk.
   No agent can make this change without a Board decision.

2. **Physical validation resource allocation (#5068, #4729, #4267).** Three
   deferred validation plans are in `docs/development/planning/`. Each needs
   Board approval plus verified equipment and participant access before any
   physical or acoustic claim can be made. Software work continues independently.

3. **TOOLS-D9 (#4730) prioritization.** This is the sole remaining subepic to
   close `[DOC-TOOLS]` (#4707). If the Board wants the design manual epic closed
   in v1.21, a delivery agent should be assigned now.
