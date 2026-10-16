# F-TL-024

**AGENTS.md documents 10 Jules workflows and a nightly schedule; 3 exist; architecture link broken**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | docs     | high       | S      | proposed |

**Locations:** `AGENTS.md:394`, `AGENTS.md:417`, `AGENTS.md:428`, `AGENTS.md:505`, `AGENTS.md:516`, `requirements.txt:20`

## Finding

- Severity: Medium - Category: docs
- Confidence: high
- Location: AGENTS.md:394, :398-411, :417, :428, :438, :450, :461, :472, :483, :494, :505, :516; requirements.txt:20
- Evidence: `**Reference:** [JULES_ARCHITECTURE.md](JULES_ARCHITECTURE.md)` (file lives at docs/architecture/JULES_ARCHITECTURE.md); `**Workflow:** .github/workflows/Jules-Control-Tower.yml` … `Jules-PR-Compiler.yml`, `Jules-Auto-Rebase.yml` — none exist; only `Jules-Auto-Assign-Issues.yml`, `Jules-Diff-Verifier.yml`, `Jules-Issue-Mention-Handler.yml` do. requirements.txt:20 cites `Jules-PR-AutoFix.yml`, also absent.
- Failure scenario: an agent reading the "authoritative guide for AI agents" plans around a Sentinel security sweep at 02:30 PST and an Auto-Rebase at 05:00 that never run.
- Why a newer engineer misses this: 130 lines of confident prose with cron expressions; nothing cross-checks doc references to workflow files.
- Proposed change: delete the section or replace with a 3-row table of real workflows; add a 15-line test that every `.github/workflows/*.yml` named in AGENTS.md/CLAUDE.md/requirements.txt exists.
- Verification: the test.
- Effort: S

## Draft Pull Request

**Title:** `docs(agents): replace fictional Jules workflows with the real three`

### Why This Matters (The Pitch)

The authoritative guide for AI agents describes ten overnight Jules workflows with cron times, and three exist. An agent reading AGENTS.md will plan around a Sentinel security sweep at 02:30 PST and an Auto-Rebase at 05:00 that never run, and the architecture link points at a file that lives elsewhere. requirements.txt cites a Jules-PR-AutoFix workflow that is also absent. About 130 lines of confident prose misleading every agent session is a bad trade, and the fix is cheap: replace the section with a three-row table of the real workflows, and add a fifteen-line test that every workflow file named in AGENTS.md, CLAUDE.md or requirements.txt exists. Agents get true context and the next drift gets caught in CI. It is mostly deletion, which is the cheapest kind of fix to review and the one most likely to be correct.

### Proposed PR Body

## Problem

`AGENTS.md:394, :398-411, :417-516` references workflows such as `Jules-Control-Tower.yml`, `Jules-PR-Compiler.yml`, `Jules-Auto-Rebase.yml` that do not exist; only `Jules-Auto-Assign-Issues.yml`, `Jules-Diff-Verifier.yml` and `Jules-Issue-Mention-Handler.yml` do. `JULES_ARCHITECTURE.md` lives at `docs/architecture/`. `requirements.txt:20` cites `Jules-PR-AutoFix.yml`.

## Change

- [ ] Delete the section or replace it with a 3-row table of real workflows
- [ ] Fix the `JULES_ARCHITECTURE.md` link
- [ ] Fix the `Jules-PR-AutoFix.yml` reference in `requirements.txt:20`
- [ ] Add a test that every `.github/workflows/*.yml` named in AGENTS.md, CLAUDE.md and requirements.txt exists

## Verification

The new existence test passes.

## Risk and Rollback

Documentation only plus one test; a mis-named real workflow would fail CI, which is the point. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-024

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
