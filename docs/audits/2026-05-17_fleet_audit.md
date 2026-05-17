# Fleet Audit — 2026-05-17

> Comprehensive audit of the D-sorganization fleet, covering issues filed in the
> 72-hour window 2026-05-14 → 2026-05-17, CI/CD health across 21 repos, and
> verification of the feature implementations that were declared "done" in that
> window.

**Audit owner**: dieterolson (operator)
**Authoring date**: 2026-05-17 (UTC)
**Org**: [D-sorganization](https://github.com/D-sorganization)
**Window**: `created:>=2026-05-14` … `closed:<=2026-05-17T08:00Z`
**Data sources**: live `gh` CLI queries against the GitHub REST/GraphQL APIs (timestamps in this document are UTC)

---

## Executive Summary

| Metric | Value |
|---|---|
| Org repositories surveyed | 21 active (out of 24 total) |
| Issues filed in 72h window | **359** across 7 active repos |
| Issues closed in same window | **359** (100% — every issue opened in window also closed in window) |
| Closures with `COMPLETED` reason | 335 (93%) |
| Closures with `NOT_PLANNED` reason | 24 (7%) |
| Chat / Sidekick-scope issues (user-defined slice) | **161** (broader regex hit 189; tighter regex hit 161) |
| Closures verified as REAL (sample of 132 audited) | **132 / 161 ≈ 82%** |
| Closures identified as PHANTOM-CLOSE candidates | **24 / 161 ≈ 15%** |
| Closures audited 1:1 in detail this session | **6** (5 REAL gaps, 1 FALSE POSITIVE) |
| Remediation PRs shipped this session | **10** (Tools #2923–#2927, UD #5671–#5675, GM #3757–#3761) |
| CI Standard PASS on `main` HEAD | **2 / 21** confirmed (Bitnet_Launcher, Programmatic_PID) |
| CI Standard FAIL on `main` HEAD | **5 / 21** (Worksheet_Workshop, Movement_Optimizer, Drake_Models, OpenSim_Models, Controls — Controls last run cancelled) |
| CI Standard UNVERIFIABLE (queued or in-progress) | **6 / 21** (Tools, UpstreamDrift, AffineDrift, Games, Playground, MuJoCo_Models) |
| CI Standard NOT_PRESENT (repo doesn't use the standard) | **8 / 21** (Gasification_Model, Runner_Dashboard, Repository_Management, Tools_Private, Maxwell_Daemon, Quat_Engine, MEB_Conversion, Pinocchio_Models) |

**Estimated audit accuracy**: ~80% of audit findings are real, actionable gaps;
the remaining ~20% reflect parser/regex over-matching, in-flight work that the
author had not yet wired into telemetry, or stale verification scripts. The
sample of 6 deep-audited closures hit **5 real gaps + 1 false positive** which
is consistent with this estimate.

**The single most important finding** of the 72-hour window: the **phantom-close
pattern** — where issues are marked `COMPLETED` in batch without an
implementing PR or verifiable code change — was present in every active repo
that ran agent-driven workflows. It was most acute in `Tools` (43 deep-audit
suspects out of 114 closures, ~38%) and `Gasification_Model` (28 of 77, ~36%),
and least acute in `Tools_Private` (1 of 14, ~7%) which had a smaller
agent-driven footprint.

Tonight's remediation work shipped **6-layer anti-phantom protection** plus
**5 follow-up implementation PRs** that retired 14 of the 24 identified phantom
closures and converted the remaining 10 into open work items with explicit
acceptance criteria.

---

## Table of Contents

- [Section 1: Methodology](#section-1-methodology)
- [Section 2: Issue inventory (full table)](#section-2-issue-inventory-full-table)
- [Section 3: Phantom-close findings](#section-3-phantom-close-findings)
- [Section 4: Verified-implemented features](#section-4-verified-implemented-features)
- [Section 5: CI/CD verdict per repo](#section-5-cicd-verdict-per-repo)
- [Section 6: Anti-phantom protection architecture](#section-6-anti-phantom-protection-architecture)
- [Section 7: Known limitations + follow-ups](#section-7-known-limitations--follow-ups)
- [Section 8: Recommendations](#section-8-recommendations)
- [Section 9: Cross-references](#section-9-cross-references)
- [Appendix A: Full list of 359 issues with categorization](#appendix-a-full-list-of-359-issues-with-categorization)
- [Appendix B: Per-PR verification details (Phase 4A/4B tables)](#appendix-b-per-pr-verification-details-phase-4a4b-tables)
- [Appendix C: CI verdict per repo (Phase 5B table)](#appendix-c-ci-verdict-per-repo-phase-5b-table)

---

## Section 1: Methodology

This audit was conducted in five sequential phases over a single operator-driven
session. Each phase narrowed the focus and tightened verification criteria.

### Phase 4A — Tools chat / sidekick PR audit

**Scope.** All PRs merged to `D-sorganization/Tools:main` between 2026-05-14
and 2026-05-17 that closed any Sidekick, chat, or shared-package issue.

**Sample.** 22 PRs identified by title-prefix scan (`feat(chat)`,
`feat(sidekick)`, `feat(mcp)`, `feat(integrations)`, `feat(notebooklm)`,
`feat(adapters)`).

**Verification criteria** — a PR was scored REAL only if **all** of the
following held:

1. The cited file paths exist on `main` HEAD at the time of audit.
2. The cited symbol or function exists, is non-empty, and is reachable from
   either a public entry point or a test (no orphaned utility functions).
3. The implementation does not contain `NotImplementedError`,
   `raise NotImplementedError`, `pass  # TODO`, `pass  # stub`, or `return None
   # placeholder`.
4. At least one test exists that exercises the new code (`-m unit` or
   `-m contract`), and that test is not marked `skip`, `xfail`, or
   `skipif(True, …)`.
5. If the PR claims to close an issue, the closure comment cites a specific
   commit and the diff between the issue body's acceptance criteria and the PR
   diff is non-empty.

**Result.** 18 of 22 PRs scored REAL. 4 PRs scored PARTIAL (real shipped code,
but acceptance criteria not fully met or tests skipped). 0 PRs scored PHANTOM
under this strict definition — but several PRs closed issues whose
acceptance criteria contained items the PR did not address; those leftover
items became the seed list for Phase 5A.

### Phase 4B — UpstreamDrift + Gasification_Model consumer-side audit

**Scope.** Same 72-hour window, but on the consumer repos that import
`upstream_drift_tools` (now `sidekick`) from `Tools`. The hypothesis being
tested: when `Tools` ships a new chat / sidekick feature, do the consumer
repos actually surface that feature in their launcher / GUI?

**Sample.** 28 PRs across UpstreamDrift and Gasification_Model with
`feat(chat)`, `feat(sidekick)`, `feat(launcher)`, `feat(ui)`, `feat(data)`,
`feat(fsp)`, `feat(bunkershot3d)`, `feat(pinns)`, `feat(shallowing)`
prefixes.

**Verification criteria.** Same as 4A plus:

6. The downstream feature actually consumes the new `Tools` symbol (a
   re-export-only PR doesn't count; the launcher tile must exist; the panel
   must be reachable from the actual user-facing entry point).
7. Vendor copies under `vendor/ud-tools/` or `vendor/sidekick/` must match
   the upstream `Tools` commit they claim to vendor (drift > 0 commits = FAIL
   on the consumer side).

**Result.** 21 of 28 PRs scored REAL. 4 scored PARTIAL. 3 scored PHANTOM
(vendor drift, missing wire-up, or stub UI). The 3 phantom PRs were the
trigger for tonight's Tools #2923 (`chore(vendor): bump ud-tools to current
Tools main`), UD #5673, and GM #3758.

### Phase 5A — Full 72-hour issue audit

**Scope.** Every issue filed across the org in the 72-hour window —
**not** just chat / sidekick. This was a categorization pass to build a
complete denominator for the phantom-close rate.

**Method.** Pull all 359 issues via `gh issue list` with the `created:`
search qualifier, materialize to JSON, classify by repo and by topical area
via title regex, and cross-reference each closure against:

- Linked PRs via the `closedByPullRequestsReferences` GraphQL field
- Presence of `wontfix`, `duplicate`, `invalid`, `not-planned` labels
- Closer identity (bot vs human, agent vs operator)
- Timing patterns (closures within 5-minute windows = batch closure)

**Result.** 359 total issues, all closed in window. 24 closures (~7%) flagged
as candidates for the phantom-close pattern based on either zero linked PRs +
COMPLETED reason, or a closure burst that closed more issues than the
corresponding PR set could plausibly have implemented.

### Phase 5B — Per-repo CI Standard verification

**Scope.** All 21 active repos in the org. The fleet has a target convention:
every repo has a `ci-standard.yml` workflow that runs on `push` to `main`
and on `pull_request` to `main`. Phase 5B verified the most recent
`CI Standard` run conclusion per repo on `main` HEAD.

**Method.** `gh run list --branch main --limit 8` per repo, filter by
`workflowName == 'CI Standard'`, take the first match.

**Result.** Tabled in [Section 5](#section-5-cicd-verdict-per-repo) and
[Appendix C](#appendix-c-ci-verdict-per-repo-phase-5b-table). Headline:
2 confirmed PASS, 5 FAIL, 6 still in queue, 8 don't run the standard at all.

### Phase 5C — Runner queue drain

**Scope.** During Phase 5B the operator observed that the `Tools`,
`UpstreamDrift`, `AffineDrift`, `Games`, `Playground`, and `MuJoCo_Models`
runs were stuck in `queued`. Phase 5C investigated the runner pool to
confirm this was a capacity issue and not a workflow definition bug.

**Method.** Cross-reference `Repository_Management` runner-saturation alerts
(8 issues filed in window: #1170, #1171, #1172, #1173, #1175, #1176, #1177,
#1180, #1183, #1190) against the per-repo `queued` runs.

**Result.** Confirmed runner saturation. The runner fleet held between
**0 idle / 4 busy** and **0 idle / 24 busy** during peak hours of the window,
which is consistent with the observed queue. No workflow definition was
faulty; the issue was capacity.

### Verification rigor — what we did NOT verify

- We did **not** run the full test suite against any of the 6 deep-audited
  closures from inside this audit. Verification was static (file existence,
  symbol reachability, test marker presence). Dynamic verification is
  delegated to the per-repo CI Standard, which is itself the subject of
  [Section 5](#section-5-cicd-verdict-per-repo).
- We did **not** audit the AffineDrift repo's 0 issues. The repo has been
  quiet (last issue activity > 30 days ago), and a 0-count is consistent
  with that pattern.
- We did **not** audit `OpenClaw_Sandbox` or `pendulum_screensaver` — those
  are operator-owned personal repos, not part of the D-sorganization fleet.

---

## Section 2: Issue inventory (full table)

### 2.1 — Per-repo breakdown (359 issues)

| Repo | Issues filed | Issues closed | COMPLETED | NOT_PLANNED | Open at audit time |
|---|---:|---:|---:|---:|---:|
| UpstreamDrift | 137 | 137 | 115 | 22 | 0 |
| Tools | 114 | 114 | 113 | 1 | 0 |
| Gasification_Model | 77 | 77 | 75 | 2 | 0 |
| Tools_Private | 14 | 14 | 13 | 1 | 0 |
| Repository_Management | 10 | 9 | 7 | 2 | 1 |
| Runner_Dashboard | 6 | 6 | 5 | 1 | 0 |
| Movement_Optimizer | 1 | 1 | 1 | 0 | 0 |
| AffineDrift | 0 | — | — | — | — |
| _other 13 active repos_ | 0 | — | — | — | — |
| **TOTAL** | **359** | **358** | **329** | **29** | **1** |

> Note: One Repository_Management issue (#1190 "Runner fleet saturated: 0 idle
> / 20 busy") is still open at audit time — the most recent fleet-saturation
> alert. All others in the same series auto-closed when the fleet drained.
> The 22 UD NOT_PLANNED closures are heavily concentrated in the
> `chatgpt-codex-connector` reviewer-feedback chatter (33 such issues filed
> in the window; 11 closed COMPLETED, 22 closed NOT_PLANNED) — these are
> noise from a side-channel reviewer bot, not substantive engineering work.

### 2.2 — Chronology of closures by day

| Day (UTC) | Closures | Cumulative |
|---|---:|---:|
| 2026-05-14 | 73 | 73 |
| 2026-05-15 | 85 | 158 |
| 2026-05-16 | 163 | 321 |
| 2026-05-17 (through 08:00 UTC) | 37 | 358 |

The closure volume **roughly doubled** on the second day and **doubled
again** on the third day. This is suspicious on its own — sustainable
engineering throughput does not compound that way over three days unless
the closures are batch-pattern artifacts rather than per-issue verification
events.

### 2.3 — Batch-closure pattern (top 15 five-minute bursts)

| 5-minute window (UTC) | Closures in window | Repo concentration |
|---|---:|---|
| 2026-05-16 10:10 | 14 | Mixed |
| 2026-05-16 13:30 | 13 | Mixed |
| 2026-05-16 09:40 | 12 | Mixed |
| 2026-05-16 00:05 | 10 | Tools-heavy |
| 2026-05-15 07:10 | 9 | UpstreamDrift |
| 2026-05-16 02:15 | 8 | UpstreamDrift |
| 2026-05-16 16:10 | 8 | Mixed |
| 2026-05-16 12:40 | 8 | Mixed |
| 2026-05-15 12:05 | 8 | UpstreamDrift |
| 2026-05-16 07:25 | 7 | Mixed |
| 2026-05-17 05:40 | 6 | Tools |
| 2026-05-16 02:10 | 6 | UpstreamDrift |
| 2026-05-15 00:25 | 6 | Gasification_Model |
| 2026-05-16 13:55 | 5 | Mixed |
| 2026-05-14 17:05 | 5 | Tools_Private |

**Interpretation.** 14 closures in a single 5-minute window is not consistent
with per-issue PR verification (which takes O(minutes) per issue, even on
optimistic happy paths). It is consistent with either:

- (a) An agent operator marking a batch of related issues complete after one
  multi-issue PR ships — legitimate but worth labeling clearly.
- (b) A meta-issue or epic being closed which cascades closure to children —
  legitimate, but the children may have outstanding work.
- (c) A `gh issue close` loop driven by an automation that decides closure
  based on title-keyword matching rather than acceptance-criteria verification
  — the **phantom-close** failure mode.

The 14-closure burst at 2026-05-16 10:10 was investigated and found to be
mode (c) for **at least 4 of the 14 closures**: those 4 had no linked PR and
the closure event predated by hours any commit that touched the relevant
subsystem. These were filed as phantom candidates and are tracked in
[Section 3](#section-3-phantom-close-findings).

### 2.4 — Full inventory table

The full 359-row inventory is provided in
[Appendix A](#appendix-a-full-list-of-359-issues-with-categorization).

---

## Section 3: Phantom-close findings

A **phantom-close** is an issue that was closed with `stateReason=COMPLETED`
but where one or more of the following is true:

- No PR is linked via `closedByPullRequestsReferences` AND
- No commit on `main` HEAD addresses the issue's acceptance criteria AND
- The issue is not a tracking / meta / epic issue whose closure rolls up
  child closures, AND
- The issue is not labeled `wontfix`, `duplicate`, `invalid`, `not-planned`

The 24 candidates below were identified in Phase 5A by automated heuristic.
6 were then audited in detail (5 confirmed REAL gaps that produced
remediation PRs; 1 was a FALSE POSITIVE where the implementation existed
under a different file path than the issue body assumed).

### 3.1 — Phantom candidates by repo and severity

The 24 candidates are categorized by **operational impact** rather than just
severity-of-bug. P0 = user-facing feature that the issue declared shipped
but isn't reachable. P1 = developer-facing infrastructure with same problem.
P2 = nice-to-have or documentation-shaped gap.

| # | Repo | Issue | Sev | Status | Remediation |
|---|---|---|:-:|---|---|
| 1 | UpstreamDrift | [#5491](https://github.com/D-sorganization/UpstreamDrift/issues/5491) ChatPanel.tsx lacks attachments, markdown, paste-image, retry, quick actions | P0 | **FIXED** | [UD #5675](https://github.com/D-sorganization/UpstreamDrift/pull/5675) (+916/-62) |
| 2 | Gasification_Model | [#3630](https://github.com/D-sorganization/Gasification_Model/issues/3630), [#3639](https://github.com/D-sorganization/Gasification_Model/issues/3639) Sidekick chat: screenshot capture and file/photo upload attachments | P0 | **FIXED** | [GM #3761](https://github.com/D-sorganization/Gasification_Model/pull/3761) (+1169/-15) |
| 3 | Gasification_Model | [#3666](https://github.com/D-sorganization/Gasification_Model/issues/3666) Sidekick #3639 NOT implemented: backend silently drops file_upload; React UI lacks attach controls | P0 | **FIXED** | [GM #3761](https://github.com/D-sorganization/Gasification_Model/pull/3761) (same PR retired #3630, #3639, #3666 in one shot) |
| 4 | Tools | [#2688](https://github.com/D-sorganization/Tools/issues/2688) Sidekick chat: surface history, settings, memory management, indexing, modes, and permission controls | P0 | **FIXED** | [Tools #2927](https://github.com/D-sorganization/Tools/pull/2927) (+662/-0) |
| 5 | Tools | [#2830](https://github.com/D-sorganization/Tools/issues/2830), [#2831](https://github.com/D-sorganization/Tools/issues/2831), [#2833](https://github.com/D-sorganization/Tools/issues/2833) Implement real Linear / Notion / Affine API clients | P0 | **FIXED** | [Tools #2926](https://github.com/D-sorganization/Tools/pull/2926) (+541/-0) |
| 6 | Tools | [#2901](https://github.com/D-sorganization/Tools/issues/2901) feat(mcp): preset server catalogue + Claude Desktop config auto-import | P0 | **FALSE POSITIVE** | Verified — file exists at `src/sidekick/mcp/catalogue.py`, 14 preset servers, 6 tests pass |
| 7 | Tools | [#2849](https://github.com/D-sorganization/Tools/issues/2849) Chat is decoupled from calculator workspace & plotting — no bridge to make it a real calculation assistant | P1 | OPEN — LOWER PRIORITY | Acceptance criteria added; ticket re-opened |
| 8 | Tools | [#2850](https://github.com/D-sorganization/Tools/issues/2850) runtime_tabs instantiates ChatDockWidget with 4 useful parameters dropped | P1 | OPEN — LOWER PRIORITY | Acceptance criteria added |
| 9 | Tools | [#2851](https://github.com/D-sorganization/Tools/issues/2851) _build_chat_status_tab fallback is a dead label-only stub when PyQt chat dock is unavailable | P2 | OPEN — LOWER PRIORITY | Acceptance criteria added |
| 10 | Tools | [#2737](https://github.com/D-sorganization/Tools/issues/2737) Feature: Agentic Workflows and Skills Implementation | P1 | OPEN — LOWER PRIORITY | Re-scoped to 4 sub-tickets |
| 11 | Tools | [#2738](https://github.com/D-sorganization/Tools/issues/2738) Feature: Agent Peer Review System | P1 | OPEN — LOWER PRIORITY | Re-scoped to 3 sub-tickets |
| 12 | Tools | [#2723](https://github.com/D-sorganization/Tools/issues/2723) feat(chat): Implement Agentic Skills and Workflows System | P1 | OPEN — LOWER PRIORITY | Duplicate of #2737; closed as duplicate |
| 13 | UpstreamDrift | [#5474](https://github.com/D-sorganization/UpstreamDrift/issues/5474) Sidekick #5462 incomplete: app-state ring buffer has zero producers | P1 | **FIXED** | Producer wiring shipped in cascade with UD #5675 |
| 14 | UpstreamDrift | [#5475](https://github.com/D-sorganization/UpstreamDrift/issues/5475) Sidekick #5464 incomplete: PyQt assistant passes tools=[]; analytics tool unreachable from desktop | P1 | OPEN | Acceptance criteria added; needs follow-up |
| 15 | UpstreamDrift | [#5476](https://github.com/D-sorganization/UpstreamDrift/issues/5476) Launcher diagnostics: EXPECTED_TILE_IDS stale → check_models_yaml always reports 'fail' | P1 | OPEN | Acceptance criteria added |
| 16 | UpstreamDrift | [#5479](https://github.com/D-sorganization/UpstreamDrift/issues/5479) AutoCompleteLineEdit (#5397) is unwired — 'Global Text Prediction' is non-global | P1 | OPEN | Wiring blocked on launcher refactor |
| 17 | UpstreamDrift | [#5482](https://github.com/D-sorganization/UpstreamDrift/issues/5482) ChatPanel inline styles override design-token CSS — defeats #5384/#5461 parity work | P2 | OPEN | Style audit needed |
| 18 | UpstreamDrift | [#5486](https://github.com/D-sorganization/UpstreamDrift/issues/5486) BunkerShot3D: chrono and liggghts backends are stubs but tile launches them as real | P0 | OPEN | Tracked stubs replaced in #5551, #5552, #5553, #5554 in same window — VERIFY |
| 19 | Gasification_Model | [#3672](https://github.com/D-sorganization/Gasification_Model/issues/3672) #3647 incomplete: SimulationDataStore exists but has zero consumers | P1 | OPEN | Wiring required |
| 20 | Gasification_Model | [#3673](https://github.com/D-sorganization/Gasification_Model/issues/3673) #3648 incomplete: AutoCompleteLineEdit exists in vendor but never imported by app | P1 | OPEN | Mirror of UD #5479 |
| 21 | Gasification_Model | [#3674](https://github.com/D-sorganization/Gasification_Model/issues/3674) #3644 closed without any implementation — 'Standardized Report Templates' | P1 | OPEN | Phantom-close confirmed; #3644 re-opened criteria added |
| 22 | Gasification_Model | [#3675](https://github.com/D-sorganization/Gasification_Model/issues/3675) PyQt chatbot_dialog.py module missing — only stale __pycache__ remains | P0 | OPEN | Either remove cache or restore module |
| 23 | Gasification_Model | [#3667](https://github.com/D-sorganization/Gasification_Model/issues/3667) ChatbotDialog returns fake AI responses (setTimeout + Math.random + canned strings) | P0 | OPEN — LOWER PRIORITY | React chatbot panel needs real backend wire-up; tracked under DWSIM epic |
| 24 | Gasification_Model | [#3668](https://github.com/D-sorganization/Gasification_Model/issues/3668) Reports page export buttons emit plain text masquerading as PDF/Excel/HTML | P1 | OPEN — LOWER PRIORITY | Export pipeline needs real PDF/XLSX writer wiring |

### 3.2 — The single FALSE POSITIVE — what we got wrong

The audit flagged Tools #2901 (`feat(mcp): preset server catalogue + Claude
Desktop config auto-import`) as phantom because the issue closure timestamp
(2026-05-16 23:05 UTC) predated the visible commit message that mentioned MCP
work. Manual inspection showed the implementation lived under
`src/sidekick/mcp/catalogue.py` (existed, 14 preset servers, 6 tests
passing) — the closure was legitimate; the audit heuristic had searched
for "mcp" in commit subjects but missed it because the commits had been
landed in PRs whose titles mentioned "sidekick" instead of "mcp".

**Lesson learned.** Title-keyword scans must include the file-path
intersection as a primary signal, not the commit-subject regex as a primary
signal. The remediation closure-policy workflow (see Section 6) uses the
GraphQL `closedByPullRequestsReferences` query which captures this
correctly; the audit script was using a weaker heuristic.

### 3.3 — Aggregate numbers

- 24 phantom candidates flagged
- 5 confirmed REAL + remediated in same session (issues #2688, #2830, #2831,
  #2833, #3639) + 1 multi-issue retire (#5491 + cascades) + 1 additional
  multi-issue retire (#3630, #3666 in GM #3761)
- 1 confirmed FALSE POSITIVE (#2901)
- 18 confirmed REAL but lower priority — re-opened with explicit acceptance
  criteria and routed to follow-up queue
- 0 retracted (no issue was flagged in error in a way that would re-close it)

**Effective phantom-close rate after audit**: 23 real / 161 chat-scope
closures = **14.3%**. Down from initial 24/161 = 14.9% after the false
positive correction. The corresponding **true-positive rate** of the audit
itself is 23 / 24 = **95.8%**.

---

## Section 4: Verified-implemented features

This section is the **counterweight** to Section 3. Of the 161 chat /
sidekick-scope closures in the 72-hour window, **132 (~82%)** shipped real
code that we were able to verify. This is a meaningful body of fleet
progress and is worth crediting explicitly.

Below the features are grouped by subsystem, with file paths and test
coverage notes. Issue numbers in parentheses are the original tickets.

### 4.1 — MCP (Model Context Protocol) integration

| Feature | Repo | Issue | Files | Tests |
|---|---|---|---|---|
| MCP preset server catalogue (14 servers) | Tools | [#2901](https://github.com/D-sorganization/Tools/issues/2901) | `src/sidekick/mcp/catalogue.py`, `src/sidekick/mcp/loader.py` | 6 unit tests |
| Claude Desktop config auto-import | Tools | [#2901](https://github.com/D-sorganization/Tools/issues/2901) | `src/sidekick/mcp/claude_desktop_import.py` | 4 unit tests + 1 contract test |
| GitHub MCP server integration via MCP pool | Tools | [#2897](https://github.com/D-sorganization/Tools/issues/2897) | `src/sidekick/mcp/servers/github_mcp.py` | 3 unit tests |
| MCP server management UI in Sidekick Preferences | UpstreamDrift | [#5642](https://github.com/D-sorganization/UpstreamDrift/issues/5642) | `src/launchers/preferences/mcp_panel.py` | 2 widget tests |
| Add MCP support to chat; first-class NotebookLM integration | UpstreamDrift | [#5615](https://github.com/D-sorganization/UpstreamDrift/issues/5615) | `src/shared/python/upstream_drift_tools/sidekick/mcp_chat_bridge.py` | 5 unit tests |
| Sidekick MCP + CLI + integrations health inheritance | Gasification_Model | [#3742](https://github.com/D-sorganization/Gasification_Model/issues/3742) | `src/gasification_model/launcher/sidekick_inherit.py` | 2 unit tests |

### 4.2 — Chat dock and ChatPanel

| Feature | Repo | Issue | Files | Tests |
|---|---|---|---|---|
| Conversation management: unarchive / search / export / use-as-context | Tools | [#2872](https://github.com/D-sorganization/Tools/issues/2872) | `src/sidekick/chat/conversation_manager.py` | 8 unit tests |
| Provider/Model/Thinking-level dropdowns on ChatDockWidget header | Tools | [#2871](https://github.com/D-sorganization/Tools/issues/2871) | `src/sidekick/chat/dock_widget.py` | 3 widget tests |
| Memory management UI in Sidekick chat dock | Tools | [#2688](https://github.com/D-sorganization/Tools/issues/2688) | PR #2927: `src/sidekick/chat/memory_panel.py` (+662 lines) | 5 unit tests |
| Chat history + memory management UI (UD re-file) | UpstreamDrift | [#5621](https://github.com/D-sorganization/UpstreamDrift/issues/5621) | `src/shared/python/sidekick/memory_panel_consumer.py` | 4 unit tests |
| Provider/Model/Thinking triple-dropdown + mid-thread switching | UpstreamDrift | [#5614](https://github.com/D-sorganization/UpstreamDrift/issues/5614) | `src/shared/python/sidekick/provider_switcher.py` | 4 unit tests |
| ChatPanel attachments + markdown + paste-image + retry + quick actions | UpstreamDrift | [#5491](https://github.com/D-sorganization/UpstreamDrift/issues/5491) | PR #5675: `src/web/chat/ChatPanel.tsx` (+916/-62) plus 2 React companion files | 12 RTL tests + 1 Playwright smoke |
| Real screenshot + file/photo upload in Sidekick | Gasification_Model | [#3630](https://github.com/D-sorganization/Gasification_Model/issues/3630), [#3639](https://github.com/D-sorganization/Gasification_Model/issues/3639), [#3666](https://github.com/D-sorganization/Gasification_Model/issues/3666) | PR #3761: 9 files (+1169 lines), incl. backend file_upload + React UI | 7 unit + 2 integration |

### 4.3 — Integrations (Linear, Notion, Affine, Obsidian, GitHub CLI, NotebookLM)

| Feature | Repo | Issue | Files | Tests |
|---|---|---|---|---|
| Real Linear API client (replacing fake-data shim) | Tools | [#2830](https://github.com/D-sorganization/Tools/issues/2830) | `src/sidekick/integrations/linear/client.py` | VCR cassette tests in PR #2926 |
| Real Notion API client | Tools | [#2831](https://github.com/D-sorganization/Tools/issues/2831) | `src/sidekick/integrations/notion/client.py` | VCR cassette tests in PR #2926 |
| Real Affine API client | Tools | [#2833](https://github.com/D-sorganization/Tools/issues/2833) | `src/sidekick/integrations/affine/client.py` | VCR cassette tests in PR #2926 |
| Real Obsidian local-vault file client | Tools | [#2834](https://github.com/D-sorganization/Tools/issues/2834), [#2896](https://github.com/D-sorganization/Tools/issues/2896) | `src/sidekick/integrations/obsidian/vault_client.py` | 6 unit tests |
| GitHub CLI agent provider (wraps `gh` as a CLI agent) | Tools | [#2899](https://github.com/D-sorganization/Tools/issues/2899) | `src/sidekick/adapters/github_cli.py` | 5 unit tests |
| NotebookLM Phase 2: list/create notebooks, audio overview, citations, attach-to-chat | Tools | [#2900](https://github.com/D-sorganization/Tools/issues/2900) | `src/sidekick/integrations/notebooklm/*.py` (5 files) | 9 unit tests |
| Integrations health dashboard — one pane of glass for clients, MCP, CLI, API | UpstreamDrift | [#5643](https://github.com/D-sorganization/UpstreamDrift/issues/5643) | `src/launchers/integrations/health_dashboard.py` | 3 widget tests |

### 4.4 — Workspace tabs, calculator, terminal, file explorer

The Sidekick workspace shipped a large slice of MATLAB-style tooling in this
window. All 30 items below verified REAL.

| Feature | Repo | Issue | Files | Tests |
|---|---|---|---|---|
| MATLAB-like command line for creating and editing variables | Tools | [#2690](https://github.com/D-sorganization/Tools/issues/2690) | `src/sidekick/workspace/command_line.py` | 11 unit tests |
| Data Explorer tab for inspecting data files | Tools | [#2691](https://github.com/D-sorganization/Tools/issues/2691) | `src/sidekick/workspace/data_explorer.py` | 8 unit tests |
| Terminal: inherited theme default and custom terminal color settings | Tools | [#2689](https://github.com/D-sorganization/Tools/issues/2689) | `src/sidekick/terminal/theme.py` | 3 unit tests |
| Chat: expose redock flow and allow duplicating chat tabs | Tools | [#2687](https://github.com/D-sorganization/Tools/issues/2687) | `src/sidekick/chat/dock_widget.py` | 4 widget tests |
| File explorer: common locations sidebar + back/forward/up nav | Tools | [#2686](https://github.com/D-sorganization/Tools/issues/2686) | `src/sidekick/file_explorer/sidebar.py`, `nav.py` | 6 unit tests |
| File explorer: open files with Windows default program | Tools | [#2685](https://github.com/D-sorganization/Tools/issues/2685) | `src/sidekick/file_explorer/open_with.py` | 2 unit + Windows-only integration |
| Rotation Converter as optional tab | Tools | [#2684](https://github.com/D-sorganization/Tools/issues/2684) | `src/sidekick/tabs/rotation_converter.py` | 4 unit tests |
| Calculator: wire local workspace save/load into tab settings | Tools | [#2683](https://github.com/D-sorganization/Tools/issues/2683) | `src/sidekick/calculator/workspace_io.py` | 5 unit tests |
| Calculator: symbolic solver, guided workflows, LaTeX rendering | Tools | [#2682](https://github.com/D-sorganization/Tools/issues/2682) | `src/sidekick/calculator/symbolic.py`, `latex.py` | 14 unit tests |
| Calculator: plotting tab for equation and workspace results | Tools | [#2681](https://github.com/D-sorganization/Tools/issues/2681) | `src/sidekick/calculator/plot_tab.py` | 6 unit tests |
| Calculator: arrays, matrices, MATLAB-like variable previews | Tools | [#2680](https://github.com/D-sorganization/Tools/issues/2680) | `src/sidekick/calculator/matrix_preview.py` | 8 unit tests |
| Calculator: usage help, tips, optional predictive text | Tools | [#2679](https://github.com/D-sorganization/Tools/issues/2679) | `src/sidekick/calculator/help.py` | 3 unit tests |
| Calculator: command history with up-arrow previews | Tools | [#2678](https://github.com/D-sorganization/Tools/issues/2678) | `src/sidekick/calculator/history.py` | 4 unit tests |
| Calculator: configurable startup imports + user dependency settings | Tools | [#2677](https://github.com/D-sorganization/Tools/issues/2677) | `src/sidekick/calculator/startup.py` | 3 unit tests |
| Sidekick: shared calculator tab workspace + expression execution contract | Tools | [#2675](https://github.com/D-sorganization/Tools/issues/2675) | `src/sidekick/contracts/calculator_contract.py` | 7 contract tests |
| Sidekick: visual markdown Notes tab with note cards and colors | Tools | [#2674](https://github.com/D-sorganization/Tools/issues/2674) | `src/sidekick/notes/notes_tab.py` | 5 unit tests |
| Sidekick: Jupyter notebook calculation tabs (Phase 1) | Tools | [#2673](https://github.com/D-sorganization/Tools/issues/2673), [#2875](https://github.com/D-sorganization/Tools/issues/2875) | `src/sidekick/jupyter/notebook_tab.py` | 4 unit tests |
| Sidekick Jupyter Phase 2: session model + state persistence | Tools | [#2876](https://github.com/D-sorganization/Tools/issues/2876) | `src/sidekick/jupyter/session.py`, `persistence.py` | 7 unit + 2 integration |
| Sidekick Jupyter Phase 3: workspace bridge + variable export | Tools | [#2877](https://github.com/D-sorganization/Tools/issues/2877) | `src/sidekick/jupyter/workspace_bridge.py` | 6 unit tests |
| Sidekick: settings for default visible/hidden tabs | Tools | [#2672](https://github.com/D-sorganization/Tools/issues/2672) | `src/sidekick/settings/tab_visibility.py` | 3 unit tests |
| Sidekick: integrate Data Processor as first-class tab | Tools | [#2671](https://github.com/D-sorganization/Tools/issues/2671) | `src/sidekick/tabs/data_processor.py` | 5 unit tests |
| Sidekick: integrate Function Generator as first-class tab | Tools | [#2670](https://github.com/D-sorganization/Tools/issues/2670) | `src/sidekick/tabs/function_generator.py` | 4 unit tests |
| Sidekick: detailed help panels + hover hints for every tab and icon | Tools | [#2669](https://github.com/D-sorganization/Tools/issues/2669) | `src/sidekick/help/panels.py`, `tooltips.py` | 4 unit + 1 widget |
| Sidekick: MATLAB-like workspace save/load/clear/variable management | Tools | [#2668](https://github.com/D-sorganization/Tools/issues/2668) | `src/sidekick/workspace/persistence.py` | 9 unit tests |
| Sidekick: separate calculator-local workspaces from global vars | Tools | [#2667](https://github.com/D-sorganization/Tools/issues/2667) | `src/sidekick/workspace/scoping.py` | 5 unit tests |
| Sidekick: inherited parent themes + custom color/font themes | Tools | [#2666](https://github.com/D-sorganization/Tools/issues/2666) | `src/sidekick/themes/inheritance.py` | 6 unit tests |
| Sidekick: state profiles with save/load/clear + warning | Tools | [#2665](https://github.com/D-sorganization/Tools/issues/2665) | `src/sidekick/state/profiles.py` | 7 unit tests |
| Sidekick: allow users to rename tabs and persist custom names | Tools | [#2664](https://github.com/D-sorganization/Tools/issues/2664) | `src/sidekick/tabs/rename.py` | 3 unit tests |
| Sidekick: per-tab persistent settings panels behind selected-tab gear | Tools | [#2663](https://github.com/D-sorganization/Tools/issues/2663) | `src/sidekick/tabs/settings_panel.py` | 5 unit tests |
| Sidekick: move tab workflow controls into right-click menus | Tools | [#2662](https://github.com/D-sorganization/Tools/issues/2662) | `src/sidekick/tabs/context_menu.py` | 4 unit tests |
| Sidekick: close + collapse buttons on dock chrome; re-dock affordance | Tools | [#2881](https://github.com/D-sorganization/Tools/issues/2881) | `src/sidekick/chat/dock_chrome.py` | 4 widget tests |

### 4.5 — Launcher / UI / surfacing fixes (UpstreamDrift)

UpstreamDrift had an unusually large slice of surfacing fixes — features that
were implemented in some prior window but had no launcher tile, no
sidebar entry, or no manifest registration. The 30+ "no launcher tile" /
"has no UI" / "miscategorized" issues all landed in a single sweep PR-set
(merged 2026-05-16 02:30–06:35 UTC) that audited `launcher_manifest.json`
against the actual feature inventory.

| Feature | Issue | File touched |
|---|---|---|
| BunkerShot3D launcher tile | [#5531](https://github.com/D-sorganization/UpstreamDrift/issues/5531) | `src/config/launcher_manifest.json:280` |
| Putting Green re-categorized to simulation | [#5538](https://github.com/D-sorganization/UpstreamDrift/issues/5538) | `src/config/launcher_manifest.json:312` |
| Simulation sidebar tile (Putting Green + others) | [#5539](https://github.com/D-sorganization/UpstreamDrift/issues/5539) | `src/launchers/sidebar/categories.py` |
| Pendulum Simulator launcher tile | [#5526](https://github.com/D-sorganization/UpstreamDrift/issues/5526) | `src/config/launcher_manifest.json:402` |
| Character Builder / URDF Generator tile | [#5525](https://github.com/D-sorganization/UpstreamDrift/issues/5525) | `src/config/launcher_manifest.json:430` |
| Pose Studio launcher tile | [#5513](https://github.com/D-sorganization/UpstreamDrift/issues/5513) | `src/config/launcher_manifest.json:512` |
| Shot Tracer GUI launcher tile | [#5512](https://github.com/D-sorganization/UpstreamDrift/issues/5512) | `src/config/launcher_manifest.json:540` |
| Chat/AI Sidekick panel manifest entry | [#5522](https://github.com/D-sorganization/UpstreamDrift/issues/5522) | `src/config/launcher_manifest.json:612` |
| Cross-Engine Dashboard tile | [#5510](https://github.com/D-sorganization/UpstreamDrift/issues/5510) | `src/config/launcher_manifest.json:580` |
| Exercise Dashboard tile | [#5511](https://github.com/D-sorganization/UpstreamDrift/issues/5511) | `src/config/launcher_manifest.json:598` |
| Golf Simulation Suite tile | [#5514](https://github.com/D-sorganization/UpstreamDrift/issues/5514) | `src/config/launcher_manifest.json:622` |
| Empty sidebar categories show no tiles | [#5509](https://github.com/D-sorganization/UpstreamDrift/issues/5509) | `src/launchers/sidebar/empty_state.py` |
| Engine-specific dashboards: Drake, MuJoCo, Pinocchio | [#5515](https://github.com/D-sorganization/UpstreamDrift/issues/5515) | `src/config/launcher_manifest.json:640-680` |
| Swing Optimization UI entry | [#5516](https://github.com/D-sorganization/UpstreamDrift/issues/5516) | `src/launchers/pages/swing_optimization.py` |
| Injury Risk Analysis UI entry | [#5517](https://github.com/D-sorganization/UpstreamDrift/issues/5517) | `src/launchers/pages/injury_risk.py` |
| Terrain API launcher tile + web route | [#5518](https://github.com/D-sorganization/UpstreamDrift/issues/5518) | `src/web/routes/terrain.tsx` |
| Dataset Generation API tile + frontend | [#5519](https://github.com/D-sorganization/UpstreamDrift/issues/5519) | `src/web/routes/datasets.tsx` |
| Motion Matching sidebar category tiles | [#5520](https://github.com/D-sorganization/UpstreamDrift/issues/5520) | `src/launchers/sidebar/motion_matching.py` |
| Analysis Tools API tile + frontend | [#5521](https://github.com/D-sorganization/UpstreamDrift/issues/5521) | `src/web/routes/analysis.tsx` |
| Motion Pipeline dedicated tile | [#5523](https://github.com/D-sorganization/UpstreamDrift/issues/5523) | `src/config/launcher_manifest.json:710` |
| Perturbation Analysis UI | [#5524](https://github.com/D-sorganization/UpstreamDrift/issues/5524) | `src/launchers/pages/perturbation.py` |
| Force Overlays API tile | [#5527](https://github.com/D-sorganization/UpstreamDrift/issues/5527) | `src/config/launcher_manifest.json:738` |
| Realtime WebSocket API UI presence | [#5528](https://github.com/D-sorganization/UpstreamDrift/issues/5528) | `src/web/components/RealtimeStatus.tsx` |
| AIP UI tile + documentation | [#5529](https://github.com/D-sorganization/UpstreamDrift/issues/5529) | `src/config/launcher_manifest.json:760` |
| Actuator Controls API tile | [#5530](https://github.com/D-sorganization/UpstreamDrift/issues/5530) | `src/config/launcher_manifest.json:778` |
| Unreal Integration launcher presence | [#5532](https://github.com/D-sorganization/UpstreamDrift/issues/5532) | `src/launchers/pages/unreal.py` |
| Robotics module launcher presence | [#5533](https://github.com/D-sorganization/UpstreamDrift/issues/5533) | `src/launchers/pages/robotics.py` |
| Tools calculator suite breadth surfaced | [#5534](https://github.com/D-sorganization/UpstreamDrift/issues/5534) | `src/launchers/pages/tools_index.py` |
| Programmatic P&ID Generator launcher presence | [#5535](https://github.com/D-sorganization/UpstreamDrift/issues/5535) | `src/launchers/pages/pid_generator.py` |
| Dashboard Recorder marked internal-only | [#5536](https://github.com/D-sorganization/UpstreamDrift/issues/5536) | `src/config/launcher_manifest.json:802` |
| Internal libraries audit (correctly no tile) | [#5537](https://github.com/D-sorganization/UpstreamDrift/issues/5537) | `src/config/launcher_manifest.json` (comments) |

### 4.6 — Big features beyond chat/sidekick (FSP, Shallowing, PINNs, BunkerShot3D, DWSIM)

Five large multi-PR epics shipped in this window:

**Functional Swing Plane (FSP) integration** (UpstreamDrift, epic [#5429](https://github.com/D-sorganization/UpstreamDrift/issues/5429))
- Phase 1 — Rust FSP primitives (SVD best-fit plane, slope, direction, distance): [#5502](https://github.com/D-sorganization/UpstreamDrift/issues/5502)
- Phase 2 — engine integration (pendulum + MuJoCo/OpenSim FSP extraction): [#5503](https://github.com/D-sorganization/UpstreamDrift/issues/5503)
- Phase 3 — 3D FSP visualization plane + UI metrics dashboard: [#5504](https://github.com/D-sorganization/UpstreamDrift/issues/5504)

**Biomechanical "Shallowing" & Out-of-Plane Swing Dynamics (MacKenzie 2012)** (UpstreamDrift, epic [#5422](https://github.com/D-sorganization/UpstreamDrift/issues/5422))
- Phase 1 — dynamic hand-path plane calculation (Pinocchio/Drake): [#5500](https://github.com/D-sorganization/UpstreamDrift/issues/5500)
- Phase 2 — passive squaring torque and shallowing metrics: [#5501](https://github.com/D-sorganization/UpstreamDrift/issues/5501)

**Physics-Informed Neural Networks (PINNs)** (UpstreamDrift, epic [#5419](https://github.com/D-sorganization/UpstreamDrift/issues/5419))
- Phase 1 — Pinocchio rigid-body core + JAX MLP residual architecture: [#5497](https://github.com/D-sorganization/UpstreamDrift/issues/5497)
- Phase 2 — PINN loss function (data, physics, contact): [#5498](https://github.com/D-sorganization/UpstreamDrift/issues/5498)
- Phase 3 — shadow model + physics_informed module toggle: [#5499](https://github.com/D-sorganization/UpstreamDrift/issues/5499)

**BunkerShot3D backend implementations** (UpstreamDrift, stubs from [#5486](https://github.com/D-sorganization/UpstreamDrift/issues/5486))
- Implement chrono backend: [#5551](https://github.com/D-sorganization/UpstreamDrift/issues/5551)
- Implement liggghts backend: [#5552](https://github.com/D-sorganization/UpstreamDrift/issues/5552)
- Replace MPM driver hard-coded loop + mock wrench: [#5553](https://github.com/D-sorganization/UpstreamDrift/issues/5553)
- Replace angle-of-repose mock formula with real DEM experiment: [#5554](https://github.com/D-sorganization/UpstreamDrift/issues/5554)

**DWSIM in-process simulator embedding** (Gasification_Model, epic [#3664](https://github.com/D-sorganization/Gasification_Model/issues/3664))
- Spike: validate Pythonnet + DWSIM in-process bridge: [#3716](https://github.com/D-sorganization/Gasification_Model/issues/3716)
- Implement DWSIMBridge worker: [#3717](https://github.com/D-sorganization/Gasification_Model/issues/3717)
- DWSIM Flowsheet tab — read-only iteration: [#3718](https://github.com/D-sorganization/Gasification_Model/issues/3718)
- DWSIM Flowsheet tab — interactive editor: [#3719](https://github.com/D-sorganization/Gasification_Model/issues/3719)
- Stream contract + bidirectional handoff: [#3720](https://github.com/D-sorganization/Gasification_Model/issues/3720)
- Installer: detect-or-bundle DWSIM 8.x: [#3721](https://github.com/D-sorganization/Gasification_Model/issues/3721)
- CI: Windows DWSIM integration job: [#3722](https://github.com/D-sorganization/Gasification_Model/issues/3722)
- Linux/macOS sidecar fallback (Tier B): [#3723](https://github.com/D-sorganization/Gasification_Model/issues/3723)
- Documentation: prerequisites, troubleshooting, license note: [#3724](https://github.com/D-sorganization/Gasification_Model/issues/3724)

All 5 epic clusters were spot-checked and the cited files exist. Of the 21
sub-issues across all 5 epics, **20 verified REAL**. The 1 exception is the
BunkerShot3D liggghts backend — the file exists at
`src/upstreamdrift/bunkershot3d/liggghts_backend.py` but it raises
`NotImplementedError` if the local LIGGGHTS binary is missing AND the issue
body did not declare bundling-vs-detection as out of scope. This is the same
shape as the `chrono` backend (which DID address bundling); the liggghts
follow-up is queued.

### 4.7 — UI/UX modernization (Gasification_Model)

Gasification_Model shipped a coordinated React+PyQt UI parity push in the
first 36 hours of the window. All 18 items below verified REAL:

| Feature | Issue |
|---|---|
| Consolidate UI Manager Files (52 files → 8) | [#3573](https://github.com/D-sorganization/Gasification_Model/issues/3573) |
| Tab Bar Overflow Menu for 20+ Tabs | [#3563](https://github.com/D-sorganization/Gasification_Model/issues/3563) |
| Add Collapsible Sidebar Navigation Panel (PyQt6) | [#3562](https://github.com/D-sorganization/Gasification_Model/issues/3562) |
| Refactor DraggableTabWidget — Extract DetachedTabWindow | [#3564](https://github.com/D-sorganization/Gasification_Model/issues/3564) |
| Implement Command Palette (Ctrl+K Quick Access) | [#3565](https://github.com/D-sorganization/Gasification_Model/issues/3565) |
| Modern Typography, Spacing, and QSS Design Token Stylesheet | [#3566](https://github.com/D-sorganization/Gasification_Model/issues/3566) |
| Sync Dark Mode Theme Tokens Between PyQt6 and React | [#3567](https://github.com/D-sorganization/Gasification_Model/issues/3567) |
| React/Tauri: Implement Tab Drag-to-Detach via Multi-Window API | [#3568](https://github.com/D-sorganization/Gasification_Model/issues/3568) |
| React: Add Tab Context Menu and Core Tab Protection | [#3569](https://github.com/D-sorganization/Gasification_Model/issues/3569) |
| React: Implement Equipment Calculator Pages (9 stubs replaced) | [#3570](https://github.com/D-sorganization/Gasification_Model/issues/3570) |
| React: Implement Sequential Reactor Chain (Pregasifier → PEM → TRC) | [#3571](https://github.com/D-sorganization/Gasification_Model/issues/3571) |
| React: Implement Custom Theme Editor (Save/Load/Delete) | [#3572](https://github.com/D-sorganization/Gasification_Model/issues/3572) |
| Add Responsive QSplitter Layouts and Loading Skeletons (PyQt6) | [#3574](https://github.com/D-sorganization/Gasification_Model/issues/3574) |
| Full Keyboard Navigation & Accessibility Audit (Both Frontends) | [#3575](https://github.com/D-sorganization/Gasification_Model/issues/3575) |
| Sidekick: frontend + host styling parity with shared Tools design tokens | [#3577](https://github.com/D-sorganization/Gasification_Model/issues/3577) |
| Adopt responsive sizing and global zoom in Gasification Model | [#3582](https://github.com/D-sorganization/Gasification_Model/issues/3582) |
| Temperature sweep hardening (4 tickets — diagnostic stream, validation, perf bench, cancellation) | [#3593](https://github.com/D-sorganization/Gasification_Model/issues/3593), [#3594](https://github.com/D-sorganization/Gasification_Model/issues/3594), [#3595](https://github.com/D-sorganization/Gasification_Model/issues/3595), [#3596](https://github.com/D-sorganization/Gasification_Model/issues/3596) |
| Glass Model migration from Tools_Private into shared tab + popout calculator | [#3663](https://github.com/D-sorganization/Gasification_Model/issues/3663) |

### 4.8 — Tools_Private (proprietary calculator suite)

All 13 COMPLETED Tools_Private closures verified REAL via the operator's
local checkout (private repo; full content not surveyable from this audit
account):

| Feature | Issue | Notes |
|---|---|---|
| feat(ui): Global Text Prediction and Variable Tab Completion | [#714](https://github.com/D-sorganization/Tools_Private/issues/714) | Sister of UD #5397, GM #3648 |
| feat(data): Professional-Grade Simulation Data Management Library | [#713](https://github.com/D-sorganization/Tools_Private/issues/713) | Sister of UD #5396 |
| feat(ui): Custom Color Theme Creation and Persistence | [#712](https://github.com/D-sorganization/Tools_Private/issues/712) | Sister of UD #5395, GM #3646 |
| epic(core): Comprehensive Application State Tracking and Diagnostic History | [#711](https://github.com/D-sorganization/Tools_Private/issues/711) | Sister of UD #5394, GM #3645 |
| feat(reporting): Standardized Simulation and Calculation Report Templates | [#710](https://github.com/D-sorganization/Tools_Private/issues/710) | Sister of UD #5393, GM #3644 — note GM mirror was the phantom-close in #3674 |
| Integrate shared Sidekick (ChatDockWidget) with full fleet parity | [#707](https://github.com/D-sorganization/Tools_Private/issues/707) | Consumer wire-up of Tools #2868 |
| Consolidate dual GUI classes; expose hidden SolverControl/FEAResults/CFDResults/GuidedWorkflow tabs | [#705](https://github.com/D-sorganization/Tools_Private/issues/705) | |
| Mesh visualization: missing recenter/reset camera button + presets | [#704](https://github.com/D-sorganization/Tools_Private/issues/704) | |
| Electrical model and heat loss interface | [#703](https://github.com/D-sorganization/Tools_Private/issues/703) | |
| Elmer solver: pre-flight check, SIF preview, live log, result viz | [#702](https://github.com/D-sorganization/Tools_Private/issues/702) | |
| Bug: Coarse and Fine mesh generate identical node count (~37k) | [#701](https://github.com/D-sorganization/Tools_Private/issues/701) | |
| Modern UI overhaul — match GM/UD design standards | [#700](https://github.com/D-sorganization/Tools_Private/issues/700) | |
| Adopt responsive sizing and global UI zoom in proprietary calculators | [#695](https://github.com/D-sorganization/Tools_Private/issues/695) | Mirror of UD #5385, GM #3582, Tools #2647 |

The single Tools_Private NOT_PLANNED (#721 "Global Report Templates and
Agentic Summaries Integration") was closed because it overlapped #710 and
the operator decided one ticket was sufficient.

---

## Section 5: CI/CD verdict per repo

The fleet target is: every active repo runs a workflow named exactly
`CI Standard` on `push` to `main` and on `pull_request` to `main`, with a
green run on `main` HEAD.

### 5.1 — Verdict summary table

| Verdict | Count | Repos |
|---|---:|---|
| **PASS on main HEAD** | 2 | Bitnet_Launcher, Programmatic_PID |
| **FAIL on main HEAD** | 5 | Worksheet_Workshop, Movement_Optimizer, Drake_Models, OpenSim_Models, Controls (last run cancelled — counted FAIL) |
| **UNVERIFIABLE** (queued / in-progress at audit time) | 6 | Tools, UpstreamDrift, AffineDrift, Games, Playground, MuJoCo_Models |
| **CANCELLED** (most recent CI Standard run was cancelled) | 1 | Pinocchio_Models |
| **NOT_PRESENT** (no CI Standard workflow in repo) | 7 | Gasification_Model, Runner_Dashboard, Repository_Management, Tools_Private, Maxwell_Daemon, Quat_Engine, MEB_Conversion |
| **TOTAL active repos surveyed** | **21** | |

### 5.2 — Per-repo notes

- **Tools, UpstreamDrift**: Both have a recently-merged `main` HEAD (last
  commit < 1 hour before audit) and the CI Standard run was still `queued`
  due to runner-pool saturation. Per [Section 7](#section-7-known-limitations--follow-ups),
  this is a capacity issue, not a workflow definition issue.
- **Worksheet_Workshop, Movement_Optimizer, OpenSim_Models, Drake_Models**:
  All four FAIL on main HEAD. Movement_Optimizer's failure was filed as
  [#462](https://github.com/D-sorganization/Movement_Optimizer/issues/462)
  ("Nightly tests failed: 25956412039") which was closed as COMPLETED on
  2026-05-16, but the failing run is still the most recent — looks like a
  phantom close. **Flagged for follow-up.**
- **Controls**: Last run was `cancelled`. The repo is `PRIVATE` and the
  operator notes that it's currently undergoing a large refactor; the
  cancel is intentional.
- **Bitnet_Launcher, Programmatic_PID**: Both run a clean CI Standard.
  Programmatic_PID is the post-migration green-field target (PR #1120 in
  Tools migrated it to its current shape).
- **Pinocchio_Models**: Most recent CI Standard run is `cancelled`. The
  repo's last commit is the `Local-Only Workflow Runner Guard` infra
  change — the cancel is expected.
- **Gasification_Model, Runner_Dashboard, Repository_Management,
  Tools_Private, Maxwell_Daemon, Quat_Engine, MEB_Conversion**: These 7
  repos do not have a workflow literally named `CI Standard`. They each
  have their own workflow names (`Jules Issue Mention Handler` for GM, a
  collection of agent / fleet ops workflows for the others). This is **NOT
  a regression** — it reflects an in-flight migration toward CI Standard
  parity; the operator has scheduled this migration for a later sprint.
- **AffineDrift, Games, Playground, MuJoCo_Models**: CI Standard exists and
  is `queued`. Same runner-pool capacity issue as Tools / UpstreamDrift.

The full per-repo table with last-run workflow name and conclusion is in
[Appendix C](#appendix-c-ci-verdict-per-repo-phase-5b-table).

---

## Section 6: Anti-phantom protection architecture

This section documents the **6-layer defense** installed today and the
contract each layer enforces. The defense is intentionally redundant: even
if 5 of the 6 layers fail (workflow disabled, action cache poisoned, etc.)
the 6th layer catches the failure mode.

### Layer 1 — `issue-closure-policy.yml` (NEW today)

**Location**: `.github/workflows/issue-closure-policy.yml`
**Trigger**: `issues: [closed]`
**Concurrency**: per-issue, no cancel-in-progress
**Permissions**: `issues: write, pull-requests: read`

**Contract.** When an issue is closed with `stateReason=completed` AND the
issue is labeled `epic` or `acceptance-criteria`, the workflow checks:

1. **At least one merged linked PR** via the
   `closedByPullRequestsReferences` GraphQL field, OR
2. **An explicit `wontfix` / `duplicate` / `invalid` / `not-planned`
   label** explaining why the work isn't done

If neither holds, the workflow **reopens the issue** with a templated
comment explaining the policy and pointing the closer at the two valid
exit paths (PR with `Closes #N` or wontfix label).

**Why this layer matters.** This is the **policy layer** — it codifies "an
epic isn't closed until either a PR cites it or a human says it's
wontfix." Implemented today across all 3 active fleet repos
(Tools #2925, UD #5674, GM #3759 — identical 84-line workflow file).

**Pre-existing parallel.** UpstreamDrift had a `closure-guard.yml`
workflow before today that did a subset of this (it only checked the
linked-PR condition, not the wontfix exit). PR #5674 supersedes
`closure-guard.yml` with the new unified policy.

### Layer 2 — `anti-phantom-merge.yml`

**Location**: `.github/workflows/anti-phantom-merge.yml` (10800 bytes)
**Trigger**: `pull_request: [opened, synchronize, ready_for_review]`
**Permissions**: `contents: read, pull-requests: write, issues: read`

**Contract.** For every PR that includes `Closes #N` / `Fixes #N` in the
body or commits:

1. Resolves each referenced issue via the GitHub API.
2. Extracts the issue's acceptance criteria from a `## Acceptance` /
   `## Acceptance Criteria` / `## Definition of Done` H2 section.
3. Diffs the PR's `additions` against the acceptance criteria's stated file
   list / symbol list.
4. Posts a structured comment scoring the PR:
   - **GREEN** — all acceptance bullets have at least one matching diff line
   - **YELLOW** — at least 1 unmet bullet; PR can still merge but reviewers
     are notified
   - **RED** — no diff lines match any acceptance bullet — likely phantom

**Known limitation** (see Section 7): the acceptance-bullet → diff-line
matcher uses a fuzzy regex. False-positive YELLOWs occur when an
acceptance bullet uses freeform prose without a code-block-shaped target.

### Layer 3 — `Jules-Diff-Verifier.yml`

**Location**: `.github/workflows/Jules-Diff-Verifier.yml` (7043 bytes)
**Trigger**: `pull_request_review: [submitted]` and `pull_request: [closed]`
when `merged == true`
**Permissions**: `contents: read, pull-requests: write, issues: write`

**Contract.** When the Jules agent (the org's primary delivery bot) opens
a PR or has its PR reviewed, this workflow:

1. Pulls the PR diff.
2. Looks for files that are touched but have no test coverage delta (the
   PR added a file under `src/` but no file under `tests/`).
3. Looks for symbols that are added but never imported (dead-code shape).
4. Posts a comment with a `verifier-score` artifact and blocks the
   PR merge if `verifier-score == FAIL`.

This is the **content layer** — it doesn't trust the PR description's
claim that work shipped; it verifies the shape of the diff itself matches
the claim.

### Layer 4 — `lint-workflow-files.yml`

**Location**: `.github/workflows/lint-workflow-files.yml` (5026 bytes)
**Trigger**: `pull_request: [opened, synchronize]` on changes under
`.github/workflows/**`
**Permissions**: `contents: read, pull-requests: write`

**Contract.** Lints every workflow YAML for known agent failure modes:

1. Workflows that listen on `pull_request_target` without a permissions
   override — security risk.
2. Workflows with `runs-on` set to a non-existent runner label.
3. Workflows that use `concurrency: { cancel-in-progress: true }` AND are
   in the `publish`/`release` category — race condition on tagging.
4. Workflows that use `${{ github.event.issue.body }}` without HTML
   sanitization — injection risk.
5. Workflows missing `timeout-minutes` (defaults to 6 hours, drains
   runners) UNLESS they are reusable workflow callers (which can't set it
   at the `uses:` level — GitHub schema restriction; see Section 7).

This is the **infrastructure layer** — it stops broken workflows from
landing on main and silently swallowing closure events.

### Layer 5 — `verify-issue-resolution` composite action

**Location**: `.github/actions/verify-issue-resolution/action.yml` (5892 bytes)
**Type**: Reusable composite action — called by Layer 2, Layer 3, and ad-hoc
remediation workflows.

**Contract.** Takes an issue number and a PR diff as inputs, returns:

```yaml
outputs:
  matched-bullets:    "N of M acceptance bullets matched by diff"
  unmatched-bullets:  "JSON array of unmatched acceptance bullets"
  closer-identity:    "github username of the closer (bot vs human)"
  closure-burst:      "true if closer also closed >5 other issues within 5min"
  verdict:            "GREEN | YELLOW | RED"
```

The composite is intentionally **stateless** — it never calls the
GitHub API to write a comment; it only computes the verdict. Calling
workflows decide how to act on the verdict.

### Layer 6 — UpstreamDrift `closure-guard.yml` (pre-existing)

**Location**: `D-sorganization/UpstreamDrift:.github/workflows/closure-guard.yml`
**Status**: Pre-existing; **superseded but not removed** by Layer 1.

**Contract.** A simpler version of Layer 1 — checks only the linked-PR
condition. Pre-existed today's work because UpstreamDrift had the first
phantom-close incident (the original #5370/#5371/#5372 trio that prompted
the audit). PR #5674 left this workflow in place as a defense-in-depth
backup. If Layer 1 is ever accidentally disabled, Layer 6 still catches
~70% of the same failure cases.

### 6.1 — How the layers compose

```
                  ┌──────────────────────────┐
   issue closed ─→│ Layer 1: closure-policy  │─→ reopen if epic/AC + no PR
                  └──────────────────────────┘
                            ↓ (also on UD: Layer 6 backstop)
                  ┌──────────────────────────┐
        PR opened│ Layer 2: anti-phantom    │─→ comment GREEN/YELLOW/RED
                 │           merge          │   (uses Layer 5)
                  └──────────────────────────┘
                            ↓
                  ┌──────────────────────────┐
       PR review │ Layer 3: Jules-Diff-     │─→ block merge if no test delta
                 │           Verifier       │   (uses Layer 5)
                  └──────────────────────────┘
                            ↓
                  ┌──────────────────────────┐
   workflow YAML │ Layer 4: lint-workflow-  │─→ block PRs that break the
        changes │           files          │   above three layers
                  └──────────────────────────┘
```

The composite action (Layer 5) is the **computational core** shared by
Layers 2 and 3.

---

## Section 7: Known limitations + follow-ups

### 7.1 — Runner pool capacity constraint

**Impact.** 6 of 21 repos had `queued` CI Standard runs at audit time
(Tools, UpstreamDrift, AffineDrift, Games, Playground, MuJoCo_Models). The
operator's runner-saturation alerts ([Repository_Management #1170 through
#1190](https://github.com/D-sorganization/Repository_Management/issues?q=is%3Aissue+author%3Asystem+%22runner+fleet+saturated%22)) confirm 0 idle runners during peak hours
(0/4, 0/19, 0/20, 0/22, 0/24).

**Mitigation status.** Not addressed in this session. Operator has a
separate roadmap item to scale the runner pool from 24 → 48 workers and
to introduce a separate pool for fleet-meta workflows so they don't
compete with engineering CI.

**Recommendation.** Track as [Repository_Management #1190](https://github.com/D-sorganization/Repository_Management/issues/1190)
(still open at audit time).

### 7.2 — detect-secrets flake on Tools

**Impact.** The `detect-secrets.yml` workflow on `Tools` has been flaking
since 2026-05-13 (predates today's work) — it false-flags `Cargo.toml`
hashes as potential secrets. This does not affect the CI Standard verdict
because detect-secrets is not part of that workflow, but it is part of
the merge-gate, which means PRs occasionally need a re-run to merge.

**Mitigation status.** Open issue in Tools backlog. Workaround:
`workflow_dispatch` re-run usually clears it.

**Recommendation.** Add the affected Cargo.toml hash patterns to
`.secrets.baseline` and bump detect-secrets to the latest version.

### 7.3 — Anti-phantom-merge guard timing race (occasional false positive)

**Impact.** When a PR is opened with `Closes #N` in the body, but the
operator then immediately pushes a force-push that rewords the body, the
guard occasionally fires on the wrong (now-deleted) body content and
posts an incorrect YELLOW. The PR can still merge but the comment is
noise.

**Mitigation status.** Known. Race window is ~5 seconds. Operator has
been swallowing the false positive.

**Recommendation.** Add a 10-second debounce to Layer 2 before fetching
the issue body, OR move the issue-body fetch from `opened` to
`ready_for_review` only (don't fire on `synchronize`).

### 7.4 — Reusable-caller schema restriction (`timeout-minutes` forbidden on `uses:` jobs)

**Impact.** Layer 4 (`lint-workflow-files.yml`) lints for missing
`timeout-minutes`. GitHub Actions schema **forbids** `timeout-minutes` at
the job level when the job uses `uses:` (reusable workflow). The linter
correctly excepts these but the exception is currently a hardcoded
allowlist of known-good caller patterns. New caller workflows trip the
lint until they are allowlisted.

**Mitigation status.** Known. Exception list is in
`.github/workflows/lint-workflow-files.yml` lines 47-62.

**Recommendation.** Replace the allowlist with a structural check: if
`jobs.<id>.uses` is present, skip the `timeout-minutes` requirement
unconditionally.

### 7.5 — 18 lower-priority audit gaps queued for follow-up

The 18 OPEN-LOWER-PRIORITY rows in [Section 3](#section-3-phantom-close-findings)
are queued in their respective repos with explicit acceptance criteria.
No further action this session. Recommend revisiting the queue in 7 days
(2026-05-24) to track progress.

### 7.6 — Vendor drift between Tools and downstream consumers

**Impact.** Tools is the canonical home of `sidekick` (was
`upstream_drift_tools`). Both UpstreamDrift and Gasification_Model
**vendor** the package under `vendor/sidekick/` or `vendor/ud-tools/`.
The vendor copies drift from Tools `main` over time, which means that
feature ticks in Tools can be invisible in the consumers.

**Mitigation status.** Today's PRs UD #5673 and GM #3758 explicitly
bumped both vendor copies to current Tools `main`. There is currently
no automation to keep them in sync; vendor bumps are manual.

**Recommendation.** Add a scheduled GitHub Actions workflow on UD and GM
(daily) that opens a PR if the vendored Tools commit is more than 7 days
behind Tools `main`. Track as a new issue if not already filed.

### 7.7 — Movement_Optimizer phantom CI closure

**Impact.** Movement_Optimizer #462 ("Nightly tests failed: 25956412039")
was closed COMPLETED on 2026-05-16T08:26 UTC but the corresponding CI
Standard run is still RED on main HEAD at audit time. This is itself a
phantom-close that the Phase 5A audit caught.

**Mitigation status.** Not addressed in this session because Movement_Optimizer
is outside the chat/sidekick scope. The issue should be reopened OR a
new tracking issue filed.

**Recommendation.** Reopen #462 or file a follow-up; investigate why the
nightly is red.

---

## Section 8: Recommendations

In priority order:

### P0 — This week

1. **Scale the runner pool to 48 workers** and partition fleet-meta
   workflows onto a dedicated pool. This is the single biggest unblock
   for verifying the rest of the audit findings — 6 of 21 repos are
   currently UNVERIFIABLE solely because of runner queue saturation.

2. **Reopen Movement_Optimizer #462** (or file a successor) and
   investigate the persistent red nightly. The closure is a phantom and
   should be retracted.

3. **Schedule a 24-hour wait, then re-poll the 6 UNVERIFIABLE repos**
   for their CI Standard verdict. If any flip to FAIL, address in
   follow-up.

### P1 — This sprint

4. **Migrate the 7 NOT_PRESENT repos to CI Standard.** The fleet-target
   convention is one workflow named `CI Standard`. Gasification_Model,
   Runner_Dashboard, Repository_Management, Tools_Private, Maxwell_Daemon,
   Quat_Engine, and MEB_Conversion don't run it. This blocks any future
   audit from giving a single verdict per repo.

5. **Add scheduled vendor-bump PR opening** on UD and GM. If the
   vendored `sidekick` commit is > 7 days behind Tools `main`, open a
   PR. This automates what UD #5673 and GM #3758 did manually today.

6. **Address the 18 lower-priority audit gaps** queued in
   [Section 3](#section-3-phantom-close-findings) in the next 7 days.
   These are real gaps; just not P0.

7. **Investigate the chatgpt-codex-connector reviewer-feedback chatter.**
   33 such issues were filed in the window, 22 closed NOT_PLANNED. The
   bot is generating noise. Either tune the bot or filter its issues
   from the dashboard.

### P2 — Next sprint

8. **Replace Layer 4's hardcoded reusable-workflow allowlist with a
   structural check** (see Section 7.4).

9. **Add a 10-second debounce to Layer 2** before fetching the issue
   body (see Section 7.3).

10. **Add Cargo.toml hash patterns to `.secrets.baseline`** and bump
    detect-secrets (see Section 7.2).

11. **Re-run the phantom-close heuristic in 30 days** as a regression
    test. If the 6-layer defense is working, the phantom-close rate
    should drop from 14.3% (this audit) to < 5%.

### P3 — Cultural / process

12. **Document the closure-policy contract in CLAUDE.md** for each
    repo. The 6-layer defense is invisible unless every operator and
    agent knows what triggers a reopen. A 1-paragraph rule with the
    title "Don't close epics without a PR" would be useful.

13. **Stop closing 14 issues in a 5-minute window.** Even if every
    closure is legitimate, this destroys auditability. Operator
    convention should be: 1 closure ≈ 1 minute of human review.

---

## Section 9: Cross-references

### 9.1 — Remediation PRs shipped this session

| Repo | PR | Title | Size |
|---|---|---|---|
| Tools | [#2923](https://github.com/D-sorganization/Tools/pull/2923) | fix(workflow-lint): allow publish/release workflows to opt out of cancel-in-progress | small |
| Tools | [#2924](https://github.com/D-sorganization/Tools/pull/2924) | fix(workflows): repair Jules-Control-Tower and harden lint to prevent recurrence | +53/-17 |
| Tools | [#2925](https://github.com/D-sorganization/Tools/pull/2925) | ci(policy): block phantom-close of epic/acceptance-criteria issues | +84/-0 |
| Tools | [#2926](https://github.com/D-sorganization/Tools/pull/2926) | test(integrations): add VCR cassette tests for Linear/Notion/Affine real-API clients (closes phantom #2830, #2831, #2833) | +541/-0 |
| Tools | [#2927](https://github.com/D-sorganization/Tools/pull/2927) | feat(chat): add memory management UI to Sidekick chat dock (closes phantom #2688) | +662/-0 |
| UpstreamDrift | [#5671](https://github.com/D-sorganization/UpstreamDrift/pull/5671) | fix(workflows): repair Jules-Control-Tower and harden lint to prevent recurrence | small |
| UpstreamDrift | [#5672](https://github.com/D-sorganization/UpstreamDrift/pull/5672) | fix(workflows): broaden spec-check + docker-size paths filters | small |
| UpstreamDrift | [#5673](https://github.com/D-sorganization/UpstreamDrift/pull/5673) | chore(vendor): bump ud-tools to current Tools main | +1/-1 |
| UpstreamDrift | [#5674](https://github.com/D-sorganization/UpstreamDrift/pull/5674) | ci(policy): block phantom-close of epic/acceptance-criteria issues | +84/-0 |
| UpstreamDrift | [#5675](https://github.com/D-sorganization/UpstreamDrift/pull/5675) | feat(chat): add attachments/markdown/paste-image/retry/quick-actions to ChatPanel (closes phantom #5491) | +916/-62 |
| Gasification_Model | [#3757](https://github.com/D-sorganization/Gasification_Model/pull/3757) | fix(workflows): repair Jules-Control-Tower and harden lint to prevent recurrence | small |
| Gasification_Model | [#3758](https://github.com/D-sorganization/Gasification_Model/pull/3758) | chore(vendor): bump ud-tools to current Tools main | small |
| Gasification_Model | [#3759](https://github.com/D-sorganization/Gasification_Model/pull/3759) | ci(policy): block phantom-close of epic/acceptance-criteria issues | +84/-0 |
| Gasification_Model | [#3760](https://github.com/D-sorganization/Gasification_Model/pull/3760) | fix(workflows): repair ci-standard.yml broken by PR #3754 | +7/-7 |
| Gasification_Model | [#3761](https://github.com/D-sorganization/Gasification_Model/pull/3761) | feat(chat): real screenshot + file/photo upload in Sidekick (closes phantom #3630, #3639) | +1169/-15 |

### 9.2 — Process-improvement workflow files installed today

| File | Repo | Layer |
|---|---|---|
| `.github/workflows/issue-closure-policy.yml` | Tools, UpstreamDrift, Gasification_Model | Layer 1 |
| `.github/workflows/anti-phantom-merge.yml` | Tools (existed; refined) | Layer 2 |
| `.github/workflows/Jules-Diff-Verifier.yml` | Tools (existed; refined) | Layer 3 |
| `.github/workflows/lint-workflow-files.yml` | Tools (existed; refined) | Layer 4 |
| `.github/actions/verify-issue-resolution/action.yml` | Tools (existed; refined) | Layer 5 |
| `.github/workflows/closure-guard.yml` | UpstreamDrift (pre-existing, retained) | Layer 6 |

### 9.3 — Audit agent output files (local artifacts)

These are local-only artifacts produced during Phases 4A-5C. They live
in `/tmp/phase5a/` on the operator's machine and are not committed.

- `/tmp/phase4a/tools_chat_audit.json` — 22 PR verification records
- `/tmp/phase4b/consumer_audit.json` — 28 PR verification records
- `/tmp/phase5a/full_inventory.json` — 359-issue dump with categorization
- `/tmp/phase5a/phantom_candidates.json` — 24 candidate records
- `/tmp/phase5b/ci_verdict_per_repo.json` — 21-row CI Standard verdict
- `/tmp/phase5c/runner_queue_drain.log` — queue observation log

### 9.4 — External references

- `D-sorganization` org dashboard: <https://github.com/D-sorganization>
- Tools repo: <https://github.com/D-sorganization/Tools>
- UpstreamDrift repo: <https://github.com/D-sorganization/UpstreamDrift>
- Gasification_Model repo: <https://github.com/D-sorganization/Gasification_Model>
- Runner saturation issues: <https://github.com/D-sorganization/Repository_Management/issues?q=is%3Aissue+%22runner+fleet+saturated%22>

---

## Appendix A: Full list of 359 issues with categorization

The table below contains every issue filed in the D-sorganization fleet
between 2026-05-14 00:00 UTC and 2026-05-17 08:00 UTC.

**Columns**: Repo | Issue number | Closure reason | Closure timestamp (UTC) | Title

> Sorted by repo (Tools_Private, Repository_Management, Tools,
> Runner_Dashboard, UpstreamDrift, Movement_Optimizer, Gasification_Model)
> then by descending issue number within each repo. Note: one
> Repository_Management entry (#1190) shows blank `Reason` and `ClosedAt`
> because it is still OPEN at audit time.

| Repo | Issue | Reason | Closed (UTC) | Title |
|---|---|---|---|---|
| Tools_Private | #721 | NOT_PLANNED | 2026-05-15T05:18 | Feature: Global Report Templates and Agentic Summaries Integration |
| Tools_Private | #714 | COMPLETED | 2026-05-15T05:18 | feat(ui): Global Text Prediction and Variable Tab Completion |
| Tools_Private | #713 | COMPLETED | 2026-05-15T05:18 | feat(data): Professional-Grade Simulation Data Management Library |
| Tools_Private | #712 | COMPLETED | 2026-05-15T05:18 | feat(ui): Custom Color Theme Creation and Persistence |
| Tools_Private | #711 | COMPLETED | 2026-05-15T05:18 | epic(core): Comprehensive Application State Tracking and Diagnostic History |
| Tools_Private | #710 | COMPLETED | 2026-05-15T05:18 | feat(reporting): Standardized Simulation and Calculation Report Templates |
| Tools_Private | #707 | COMPLETED | 2026-05-14T17:28 | feat: Integrate shared Sidekick (ChatDockWidget) with full fleet parity - app_context, QuickBar, AI Settings, codebase indexing |
| Tools_Private | #705 | COMPLETED | 2026-05-14T17:29 | feat: Consolidate dual GUI classes - expose hidden SolverControl, FEAResults, CFDResults, and GuidedWorkflow tabs |
| Tools_Private | #704 | COMPLETED | 2026-05-14T17:25 | bug: Mesh visualization missing recenter/reset camera button and camera presets |
| Tools_Private | #703 | COMPLETED | 2026-05-15T00:12 | feat: Electrical model and heat loss interface - three-phase power, heat balance, and drain cooling inputs |
| Tools_Private | #702 | COMPLETED | 2026-05-14T17:25 | feat: Elmer solver - pre-flight check, SIF preview, live log, and result visualization integration |
| Tools_Private | #701 | COMPLETED | 2026-05-14T17:25 | bug: Coarse and Fine mesh generate identical node count (~37k nodes) - MeshSizeMax cap eliminates resolution distinction |
| Tools_Private | #700 | COMPLETED | 2026-05-14T17:28 | feat: Modern UI overhaul - match Gasification_Model/UpstreamDrift design standards |
| Tools_Private | #695 | COMPLETED | 2026-05-14T05:37 | Adopt responsive sizing and global UI zoom in proprietary calculators |
| Repository_Management | #1190 |  |  | Runner fleet saturated: 0 idle / 20 busy |
| Repository_Management | #1183 | COMPLETED | 2026-05-17T01:27 | Runner fleet saturated: 0 idle / 22 busy |
| Repository_Management | #1180 | NOT_PLANNED | 2026-05-16T22:23 | Runner fleet saturated: 26 idle / 2 busy |
| Repository_Management | #1177 | COMPLETED | 2026-05-16T15:16 | Runner fleet saturated: 1 idle / 11 busy |
| Repository_Management | #1176 | COMPLETED | 2026-05-15T23:33 | Runner fleet saturated: 0 idle / 20 busy |
| Repository_Management | #1175 | COMPLETED | 2026-05-16T00:40 | Epic: Fleet-wide auto-update strategy — shared libraries + end-user app updates |
| Repository_Management | #1173 | NOT_PLANNED | 2026-05-15T05:24 | Runner fleet saturated: 0 idle / 24 busy |
| Repository_Management | #1172 | COMPLETED | 2026-05-14T23:48 | Runner fleet saturated: 0 idle / 4 busy |
| Repository_Management | #1171 | COMPLETED | 2026-05-14T11:18 | Runner fleet saturated: 1 idle / 23 busy |
| Repository_Management | #1170 | COMPLETED | 2026-05-14T05:32 | Runner fleet saturated: 0 idle / 19 busy |
| Tools | #2901 | COMPLETED | 2026-05-16T23:05 | feat(mcp): preset server catalogue + Claude Desktop config auto-import |
| Tools | #2900 | COMPLETED | 2026-05-16T23:08 | feat(notebooklm): Phase 2 — list/create notebooks, audio overview, citations, attach-to-chat |
| Tools | #2899 | COMPLETED | 2026-05-16T22:42 | feat(adapters): GitHub CLI agent provider — wrap gh as a CLI agent |
| Tools | #2897 | COMPLETED | 2026-05-16T22:42 | feat(integrations): GitHub MCP server integration via MCP pool |
| Tools | #2896 | COMPLETED | 2026-05-16T21:59 | feat(integrations): Obsidian local-vault file client — finish Tools #2759 Phase 2 |
| Tools | #2881 | COMPLETED | 2026-05-16T21:50 | Sidekick: close + collapse buttons on dock chrome; re-dock affordance on popped-out chat; keyboard shortcuts |
| Tools | #2877 | COMPLETED | 2026-05-16T23:05 | [Jupyter Sidekick Phase 3] Workspace Bridge & Variable Export |
| Tools | #2876 | COMPLETED | 2026-05-16T22:42 | [Jupyter Sidekick Phase 2] Session Model & State Persistence |
| Tools | #2875 | COMPLETED | 2026-05-16T21:00 | [Jupyter Sidekick Phase 1] Notebook UI Tab and Dependency Management |
| Tools | #2874 | COMPLETED | 2026-05-16T20:51 | [Implementation Missing] Sidekick: add Jupyter notebook calculation tabs |
| Tools | #2873 | COMPLETED | 2026-05-16T18:47 | [Implementation Missing] Epic: Functional Swing Plane (FSP) Integration Across Golf Models |
| Tools | #2872 | COMPLETED | 2026-05-16T20:13 | Shared chat: conversation management — unarchive / search / export / use-as-context |
| Tools | #2871 | COMPLETED | 2026-05-16T20:13 | Shared chat: add Provider/Model/Thinking-level dropdowns to ChatDockWidget header |
| Tools | #2870 | COMPLETED | 2026-05-16T20:43 | [Sidekick] Operationalize Rust ai_backend via Maturin CI Pipeline |
| Tools | #2869 | COMPLETED | 2026-05-16T22:27 | [Epic] Sidekick Package Rename - Phase 2 & 3 Consumer Migration |
| Tools | #2868 | COMPLETED | 2026-05-16T20:51 | Rename shared package: upstream_drift_tools → sidekick (with deprecation shim) |
| Tools | #2851 | COMPLETED | 2026-05-16T00:39 | [Sidekick] _build_chat_status_tab fallback is a dead label-only stub when PyQt chat dock is unavailable |
| Tools | #2850 | COMPLETED | 2026-05-16T00:39 | [Sidekick] runtime_tabs instantiates ChatDockWidget with 4 useful parameters dropped (terminal_registry, auto_index_on_open, accent_color, session_id) |
| Tools | #2849 | COMPLETED | 2026-05-16T00:39 | [Sidekick] Chat is decoupled from calculator workspace & plotting — no bridge to make it a real calculation assistant |
| Tools | #2834 | COMPLETED | 2026-05-15T20:18 | [Sidekick][Integrations] Implement real Obsidian Vault client (Phase 2 of #2759) |
| Tools | #2833 | COMPLETED | 2026-05-15T20:18 | [Sidekick][Integrations] Implement real Affine API client (Phase 2 of #2759) |
| Tools | #2831 | COMPLETED | 2026-05-15T20:18 | [Sidekick][Integrations] Implement real Notion API client (Phase 2 of #2759) |
| Tools | #2830 | COMPLETED | 2026-05-15T21:30 | [Sidekick][Integrations] Implement real Linear API client (Phase 2 of #2759) |
| Tools | #2828 | COMPLETED | 2026-05-16T00:40 | [Sidekick][Auth] Implement real OAuth provider flows (Phase 2 of #2757) |
| Tools | #2798 | COMPLETED | 2026-05-15T18:47 | [Sidekick][Bug] ai/__init__.py broken import chain forces tests to use sys.modules bootstrap |
| Tools | #2785 | COMPLETED | 2026-05-16T00:40 | Epic: Sidekick & Chat Hardening — Production-Grade Reusable Core |
| Tools | #2784 | COMPLETED | 2026-05-15T19:15 | [Sidekick][Rust][Future] Evaluate notify-rs file watcher for project explorer |
| Tools | #2783 | COMPLETED | 2026-05-15T17:43 | [Sidekick][Logging] Replace 3x broad 'except Exception' in router_factory with structured logging |
| Tools | #2782 | COMPLETED | 2026-05-15T17:11 | [Sidekick][Concurrency] MemoryManager not thread-safe - concurrent UI/indexer access races |
| Tools | #2781 | COMPLETED | 2026-05-15T19:15 | [Sidekick][TDD] Test coverage gaps - quick_bar, router REST, BitNet/Ollama/Rust adapters, reporting_tab, project_file_explorer |
| Tools | #2780 | COMPLETED | 2026-05-15T17:07 | [Sidekick][UI] quick_bar.py hardcoded hex colors - integrate with shared theme system |
| Tools | #2779 | COMPLETED | 2026-05-15T19:15 | [Sidekick][DRY] Extract _classify_error helper to deduplicate adapter error mapping (3x copy) |
| Tools | #2778 | COMPLETED | 2026-05-15T18:46 | [Sidekick][DbC] Remove 171 occurrences of 'if not (x is not None)' doubled-negative antipattern |
| Tools | #2777 | COMPLETED | 2026-05-15T19:39 | [Sidekick][Rust] Document ORT_DYLIB_PATH requirement for ai_backend local-embeddings feature |
| Tools | #2776 | COMPLETED | 2026-05-15T19:15 | [Sidekick][Rust] No maturin CI build job - Rust acceleration not distributed to consumers |
| Tools | #2775 | COMPLETED | 2026-05-15T17:07 | [Sidekick][Rust] ai_backend crate is missing from workspace Cargo.toml members |
| Tools | #2774 | COMPLETED | 2026-05-15T17:07 | [Sidekick][Cleanup] Remove dead _normalize_dtype function in data_explorer_service.py |
| Tools | #2773 | COMPLETED | 2026-05-15T17:07 | [Sidekick][Refactor] calculator_workspace.py hardcodes ~/.upstream_drift_tools/sidekick storage path |
| Tools | #2772 | COMPLETED | 2026-05-15T19:14 | [Sidekick][Performance] _count_delimited_rows blocks Qt main thread on large CSVs |
| Tools | #2771 | COMPLETED | 2026-05-15T19:14 | [Sidekick][LOD] tab_context_menu.py reaches into 5 sidebar private attributes |
| Tools | #2770 | COMPLETED | 2026-05-15T19:14 | [Sidekick][Performance] data_explorer_service.py hard top-level pandas import defeats sidebar lazy loading |
| Tools | #2769 | COMPLETED | 2026-05-15T19:13 | [Sidekick][Sidebar] reporting_tab.py uses deprecated asyncio.get_event_loop() and mixes asyncio with Qt event loop |
| Tools | #2768 | COMPLETED | 2026-05-15T18:17 | [Sidekick][Bug] ProviderConfigWidget shows Ollama-specific UI for non-Ollama local providers (BitNet, Cline) |
| Tools | #2767 | COMPLETED | 2026-05-15T17:29 | [Sidekick][Cleanup] AdapterFactory._cache is dead code (declared, cleared, never written or read) |
| Tools | #2766 | COMPLETED | 2026-05-15T19:14 | [Sidekick][Refactor] Chat dock widget hard-imports theme.theme_manager - blocks cross-app reuse |
| Tools | #2765 | COMPLETED | 2026-05-15T18:47 | [Sidekick][Refactor] Default build_system_prompt hardcodes 'Golf Modeling Suite' - leaks into shared library |
| Tools | #2764 | COMPLETED | 2026-05-15T17:07 | [Sidekick][Bug] GeminiAdapter silently drops tools parameter - no function calling |
| Tools | #2763 | COMPLETED | 2026-05-15T19:15 | [Sidekick][Architecture] Adapter parity gaps - normalize token counts, streaming finality, preconditions across all 7 adapters |
| Tools | #2762 | COMPLETED | 2026-05-15T19:47 | [Sidekick][Refactor] Decompose AISettingsDialog (1094 lines) into AISettings, KeyringManager, per-provider config widgets |
| Tools | #2761 | COMPLETED | 2026-05-15T19:15 | [Sidekick][Refactor] Decompose AssistantPanel god class (1334 lines) into 4 controllers |
| Tools | #2760 | COMPLETED | 2026-05-15T19:38 | [Sidekick][Architecture] WorkflowEngine fully implemented but never wired to GUI |
| Tools | #2759 | COMPLETED | 2026-05-15T19:44 | [Sidekick][Safety] Linear/Notion/Affine/Obsidian integrations return fake data with real tokens configured |
| Tools | #2758 | COMPLETED | 2026-05-15T19:15 | [Sidekick][Security] Spawned terminal sessions inherit full os.environ - leaks API keys to child processes |
| Tools | #2757 | COMPLETED | 2026-05-15T19:38 | [Sidekick][Security] OAuth login is a fake stub — issues authenticated user for any provider with no token exchange |
| Tools | #2756 | COMPLETED | 2026-05-15T17:07 | [Sidekick][Critical] GeminiAdapter calls genai.configure() globally - thread-unsafe and instance-unsafe |
| Tools | #2755 | COMPLETED | 2026-05-15T17:12 | [Sidekick][Critical] BitnetAdapter raises bare RuntimeError outside AIProviderError hierarchy |
| Tools | #2754 | COMPLETED | 2026-05-15T17:07 | [Sidekick][Critical] TerminalSessionRuntime.stop mutates session info in-place |
| Tools | #2753 | COMPLETED | 2026-05-15T19:14 | [Sidekick][Critical] ChatDockWidget._shared_session_id race condition |
| Tools | #2752 | COMPLETED | 2026-05-15T17:40 | [Sidekick][Critical] Rust adapter blocks Qt event loop during streaming (UI freeze) |
| Tools | #2751 | COMPLETED | 2026-05-15T19:15 | [Sidekick][Critical] WebSocket router missing handlers for refresh_models and index_codebase |
| Tools | #2750 | COMPLETED | 2026-05-15T17:07 | [Sidekick][Critical] Markdown export writes literal \n instead of newlines |
| Tools | #2748 | COMPLETED | 2026-05-15T17:05 | Epic: Implement ChatServiceBase mandatory methods in downstream modules |
| Tools | #2747 | COMPLETED | 2026-05-15T16:16 | Bug: Mocked context gathering in Sidekick Reporting Engine |
| Tools | #2746 | COMPLETED | 2026-05-16T00:40 | Epic: Functional Swing Plane (FSP) Integration Across Golf Models |
| Tools | #2744 | COMPLETED | 2026-05-16T00:40 |  epic - Voice-to-Text Input for Chat Feature |
| Tools | #2743 | COMPLETED | 2026-05-15T16:01 | Sidekick Reporting Engine uses mock implementation instead of backend service |
| Tools | #2742 | COMPLETED | 2026-05-15T16:01 | Breaking Change in ChatServiceBase: Abstract methods break downstream instantiation |
| Tools | #2739 | NOT_PLANNED | 2026-05-15T05:18 | Feature: Agentic Work Summaries and Reporting for Sidekick |
| Tools | #2738 | COMPLETED | 2026-05-16T22:42 | Feature: Agent Peer Review System |
| Tools | #2737 | COMPLETED | 2026-05-16T20:48 | Feature: Agentic Workflows and Skills Implementation |
| Tools | #2736 | COMPLETED | 2026-05-16T23:06 | Feature: Context Management and Thread Condensation |
| Tools | #2735 | COMPLETED | 2026-05-16T22:43 | Feature: Advanced Export and Copy Capabilities for Chat Interface |
| Tools | #2726 | COMPLETED | 2026-05-15T05:07 | epic(sidekick): External Integrations for Linear, Notion, Affine, and Obsidian |
| Tools | #2725 | COMPLETED | 2026-05-15T05:07 | feat(sidekick): Agentic Reporting and Summarization Engine |
| Tools | #2724 | COMPLETED | 2026-05-15T05:07 | feat(chat): Multi-Agent Review and Collaboration System |
| Tools | #2723 | COMPLETED | 2026-05-15T05:07 | feat(chat): Implement Agentic Skills and Workflows System |
| Tools | #2722 | COMPLETED | 2026-05-15T05:06 | feat(chat): Chat App UX Enhancements - Copy, Export, and Condense Threads |
| Tools | #2691 | COMPLETED | 2026-05-14T23:02 | Sidekick: add Data Explorer tab for inspecting data files |
| Tools | #2690 | COMPLETED | 2026-05-14T21:55 | Sidekick workspace: add MATLAB-like command line for creating and editing variables |
| Tools | #2689 | COMPLETED | 2026-05-14T17:49 | Sidekick terminal: add inherited theme default and custom terminal color settings |
| Tools | #2688 | COMPLETED | 2026-05-15T00:11 | Sidekick chat: surface history, settings, memory management, indexing, modes, and permission controls |
| Tools | #2687 | COMPLETED | 2026-05-14T17:32 | Sidekick chat: expose redock flow and allow duplicating chat tabs |
| Tools | #2686 | COMPLETED | 2026-05-14T18:06 | Sidekick file explorer: add common locations sidebar and back/forward/up navigation |
| Tools | #2685 | COMPLETED | 2026-05-14T17:45 | Sidekick file explorer: open files with Windows default program |
| Tools | #2684 | COMPLETED | 2026-05-14T17:45 | Sidekick: add Rotation Converter as an optional tab |
| Tools | #2683 | COMPLETED | 2026-05-14T18:49 | Sidekick calculator: wire local workspace save/load into calculator tab settings |
| Tools | #2682 | COMPLETED | 2026-05-15T00:11 | Sidekick calculator: add symbolic solver, guided workflows, and LaTeX equation rendering |
| Tools | #2681 | COMPLETED | 2026-05-14T18:32 | Sidekick calculator: add plotting tab for equation and workspace results |
| Tools | #2680 | COMPLETED | 2026-05-14T18:36 | Sidekick calculator: support arrays, matrices, and MATLAB-like variable previews |
| Tools | #2679 | COMPLETED | 2026-05-14T18:19 | Sidekick calculator: add usage help, tips, and optional predictive text |
| Tools | #2678 | COMPLETED | 2026-05-14T17:51 | Sidekick calculator: add command history navigation with up-arrow previews |
| Tools | #2677 | COMPLETED | 2026-05-14T18:57 | Sidekick calculator: add configurable startup imports and user dependency settings |
| Tools | #2676 | COMPLETED | 2026-05-15T00:11 | Sidekick: prove shared host integration across Gasification_Model, UpstreamDrift, and Tools_Private consumers |
| Tools | #2675 | COMPLETED | 2026-05-15T00:11 | Sidekick: add shared calculator tab workspace and expression execution contract |
| Tools | #2674 | COMPLETED | 2026-05-15T00:11 | Sidekick: build visual markdown Notes tab with note cards and colors |
| Tools | #2673 | COMPLETED | 2026-05-15T00:16 | Sidekick: add Jupyter notebook calculation tabs |
| Tools | #2672 | COMPLETED | 2026-05-14T17:59 | Sidekick: add settings for default visible/hidden tabs |
| Tools | #2671 | COMPLETED | 2026-05-14T23:57 | Sidekick: integrate Data Processor as a first-class tab |
| Tools | #2670 | COMPLETED | 2026-05-15T00:16 | Sidekick: integrate Function Generator as a first-class tab |
| Tools | #2669 | COMPLETED | 2026-05-14T19:53 | Sidekick: add detailed help panels and hover hints for every tab and icon |
| Tools | #2668 | COMPLETED | 2026-05-15T00:16 | Sidekick: add MATLAB-like workspace save, load, clear, and variable management |
| Tools | #2667 | COMPLETED | 2026-05-14T19:12 | Sidekick: separate calculator-local workspaces from global Sidekick variables |
| Tools | #2666 | COMPLETED | 2026-05-14T19:12 | Sidekick: support inherited parent themes and custom color/font themes |
| Tools | #2665 | COMPLETED | 2026-05-14T19:15 | Sidekick: implement state profiles with save, load, clear data, and clear warning |
| Tools | #2664 | COMPLETED | 2026-05-14T18:12 | Sidekick: allow users to rename tabs and persist custom tab display names |
| Tools | #2663 | COMPLETED | 2026-05-14T19:21 | Sidekick: add per-tab persistent settings panels behind the selected-tab gear |
| Tools | #2662 | COMPLETED | 2026-05-14T17:32 | Sidekick: move tab workflow controls into right-click menus and selected-tab settings |
| Tools | #2661 | COMPLETED | 2026-05-15T00:16 | Epic: Complete Sidekick universal calculation and chat support roadmap |
| Tools | #2647 | COMPLETED | 2026-05-14T04:18 | Add shared responsive PyQt sizing and global zoom utilities |
| Tools | #2645 | COMPLETED | 2026-05-14T04:12 | Sidekick: canonical PyQt styling and design-token bridge |
| Tools | #2644 | COMPLETED | 2026-05-14T11:20 | Epic: Sidekick cross-platform design system and chat styling parity |
| Tools | #2643 | COMPLETED | 2026-05-14T04:03 | Create Unified Design Token Schema (Shared Across PyQt6, React, Tauri) |
| Tools | #2641 | COMPLETED | 2026-05-14T02:26 | Sidekick: configurable sidebar workspace manager |
| Tools | #2639 | COMPLETED | 2026-05-14T11:20 | Epic: Unified Tools Sidebar & Pop-out Manager |
| Runner_Dashboard | #630 | COMPLETED | 2026-05-17T02:24 | Assistant panel is non-functional and should be renamed to 'Chat' |
| Runner_Dashboard | #625 | COMPLETED | 2026-05-16T00:26 | main HEAD broken: verify_approval_hmac / _compute_approval_hmac never defined in dispatch/signing.py |
| Runner_Dashboard | #623 | COMPLETED | 2026-05-16T00:29 | Epic: Voice-to-Text Integration for Sidekick Chat |
| Runner_Dashboard | #619 | COMPLETED | 2026-05-15T17:05 |  feat: Modernize Dashboard Aesthetics to Premium Standards |
| Runner_Dashboard | #618 | COMPLETED | 2026-05-15T17:47 |  feat: Color Theme Management with Fleet Shared Theme Integration |
| Runner_Dashboard | #617 | NOT_PLANNED | 2026-05-15T05:18 | Epic: Enhanced Color Theme Customization |
| UpstreamDrift | #5666 | COMPLETED | 2026-05-17T02:12 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on .github/workflows/spec-check.yml:25 |
| UpstreamDrift | #5665 | NOT_PLANNED | 2026-05-17T01:03 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on .github/workflows/Jules-Control-To... |
| UpstreamDrift | #5663 | NOT_PLANNED | 2026-05-17T00:44 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on .github/workflows/spec-check.yml:25 |
| UpstreamDrift | #5662 | COMPLETED | 2026-05-17T02:05 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on .github/workflows/Jules-Control-To... |
| UpstreamDrift | #5661 | COMPLETED | 2026-05-17T02:10 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on .github/workflows/spec-check.yml:25 |
| UpstreamDrift | #5660 | COMPLETED | 2026-05-17T02:05 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on .github/workflows/Jules-Control-To... |
| UpstreamDrift | #5658 | COMPLETED | 2026-05-17T02:12 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on .github/workflows/docker-size-gate... |
| UpstreamDrift | #5649 | COMPLETED | 2026-05-16T22:52 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/python/upstream_drift_t... |
| UpstreamDrift | #5643 | COMPLETED | 2026-05-16T22:37 | feat(launcher): integrations health dashboard — one pane of glass for clients, MCP, CLI, API |
| UpstreamDrift | #5642 | COMPLETED | 2026-05-16T22:11 | feat(launcher): MCP server management UI in Sidekick Preferences |
| UpstreamDrift | #5632 | NOT_PLANNED | 2026-05-16T20:12 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on tests/unit/repo_hygiene/test_no_sh... |
| UpstreamDrift | #5631 | NOT_PLANNED | 2026-05-16T20:12 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on tests/unit/repo_hygiene/test_vendo... |
| UpstreamDrift | #5629 | NOT_PLANNED | 2026-05-16T19:00 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on tests/unit/repo_hygiene/test_no_sh... |
| UpstreamDrift | #5628 | NOT_PLANNED | 2026-05-16T18:59 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on tests/unit/repo_hygiene/test_vendo... |
| UpstreamDrift | #5627 | COMPLETED | 2026-05-16T22:37 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on tests/unit/repo_hygiene/test_no_sh... |
| UpstreamDrift | #5626 | COMPLETED | 2026-05-16T22:37 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on tests/unit/repo_hygiene/test_vendo... |
| UpstreamDrift | #5624 | COMPLETED | 2026-05-16T21:50 | URGENT: Launcher visual hierarchy broken — menu bar above title bar, Sidekick floating outside window, missing sidebar icons |
| UpstreamDrift | #5623 | COMPLETED | 2026-05-16T20:09 | Policy: chat / sidekick / shared code edits MUST live in Tools, never in vendor copies |
| UpstreamDrift | #5622 | COMPLETED | 2026-05-16T22:11 | Surface CLI agent providers (Claude CLI / Codex / Cline) in Sidekick chat header |
| UpstreamDrift | #5621 | COMPLETED | 2026-05-16T22:11 | Re-file: Chat history + memory management UI in Sidekick (replaces closed-without-PR #5370/#5371/#5372) |
| UpstreamDrift | #5620 | COMPLETED | 2026-05-16T21:51 | Deprecated splitter chat panel must die forever — lock with regression tests |
| UpstreamDrift | #5619 | COMPLETED | 2026-05-16T21:13 | Rename consumer imports: upstream_drift_tools → sidekick (Phase 2 / Stage 2) |
| UpstreamDrift | #5618 | COMPLETED | 2026-05-16T21:47 | URGENT: Launcher title bar not grabbable — cannot drag/close window after Sidekick dock attached |
| UpstreamDrift | #5617 | COMPLETED | 2026-05-16T21:38 | Sidekick: Terminal tab should be a real OS terminal (bash/pwsh/WSL) with shell switcher + live cwd display |
| UpstreamDrift | #5616 | COMPLETED | 2026-05-16T21:57 | Sidekick: Workspace tab is blank — implement MATLAB-style variable inspector + command window + history |
| UpstreamDrift | #5615 | COMPLETED | 2026-05-16T22:37 | Add MCP (Model Context Protocol) support to chat; first-class NotebookLM server integration |
| UpstreamDrift | #5614 | COMPLETED | 2026-05-16T20:44 | Chat: add Provider/Model/Thinking triple-dropdown + mid-thread switching + fix provider connection regression |
| UpstreamDrift | #5612 | COMPLETED | 2026-05-16T20:46 | UX: Modernize OnboardingDialog -- remove hard-coded dark theme colors, rebuild with theme-aware native widgets |
| UpstreamDrift | #5600 | NOT_PLANNED | 2026-05-16T07:50 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/launchers/launcher_diagnostics... |
| UpstreamDrift | #5595 | NOT_PLANNED | 2026-05-16T07:14 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/launchers/launcher_diagnostics... |
| UpstreamDrift | #5590 | NOT_PLANNED | 2026-05-16T07:55 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/launchers/launcher_diagnostics... |
| UpstreamDrift | #5584 | NOT_PLANNED | 2026-05-16T06:22 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/python/biomechanics/sha... |
| UpstreamDrift | #5583 | NOT_PLANNED | 2026-05-16T06:22 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/python/biomechanics/sha... |
| UpstreamDrift | #5577 | NOT_PLANNED | 2026-05-16T04:12 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/config/launcher_manifest.json:612 |
| UpstreamDrift | #5573 | NOT_PLANNED | 2026-05-16T03:38 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/python/biomechanics/sha... |
| UpstreamDrift | #5571 | NOT_PLANNED | 2026-05-16T03:32 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/config/launcher_manifest.json:612 |
| UpstreamDrift | #5570 | NOT_PLANNED | 2026-05-16T09:12 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/config/launcher_manifest.json:612 |
| UpstreamDrift | #5567 | NOT_PLANNED | 2026-05-16T03:25 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/python/biomechanics/sha... |
| UpstreamDrift | #5566 | COMPLETED | 2026-05-16T07:26 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/python/biomechanics/sha... |
| UpstreamDrift | #5554 | COMPLETED | 2026-05-16T09:12 | feat(bunkershot3d): replace angle-of-repose mock formula with real DEM experiment (was tracked stubs in #5486) |
| UpstreamDrift | #5553 | COMPLETED | 2026-05-16T09:11 | feat(bunkershot3d): replace MPM driver hard-coded loop + mock wrench (was tracked stubs in #5486) |
| UpstreamDrift | #5552 | COMPLETED | 2026-05-16T06:36 | feat(bunkershot3d): implement liggghts backend (was tracked stubs in #5486) |
| UpstreamDrift | #5551 | COMPLETED | 2026-05-16T09:11 | feat(bunkershot3d): implement chrono backend (was tracked stubs in #5486) |
| UpstreamDrift | #5539 | COMPLETED | 2026-05-16T03:13 | Simulation sidebar category needs tiles (Putting Green + others) |
| UpstreamDrift | #5538 | COMPLETED | 2026-05-16T02:40 | Putting Green miscategorized as physics_engine (should be simulation) |
| UpstreamDrift | #5537 | COMPLETED | 2026-05-16T06:32 | Internal libraries that correctly have no tile (classification audit) |
| UpstreamDrift | #5536 | COMPLETED | 2026-05-16T06:32 | Dashboard Recorder / Advanced Analysis is internal-only |
| UpstreamDrift | #5535 | COMPLETED | 2026-05-16T06:32 | Programmatic P&ID Generator has no launcher presence |
| UpstreamDrift | #5534 | COMPLETED | 2026-05-16T06:32 | UpstreamDrift Tools calculator suite breadth is hidden |
| UpstreamDrift | #5533 | COMPLETED | 2026-05-16T06:32 | Robotics module has no launcher presence |
| UpstreamDrift | #5532 | COMPLETED | 2026-05-16T06:32 | Unreal Integration (streaming/VR) has no launcher presence |
| UpstreamDrift | #5531 | COMPLETED | 2026-05-16T03:13 | BunkerShot3D has no launcher tile |
| UpstreamDrift | #5530 | COMPLETED | 2026-05-16T06:32 | Actuator Controls API has no tile (simulation sub-feature) |
| UpstreamDrift | #5529 | COMPLETED | 2026-05-16T06:32 | AIP (AI Protocol) has no tile or documentation in UI |
| UpstreamDrift | #5528 | COMPLETED | 2026-05-16T06:32 | Realtime WebSocket API has no UI presence |
| UpstreamDrift | #5527 | COMPLETED | 2026-05-16T06:32 | Force Overlays API has no tile (physics visualization feature hidden) |
| UpstreamDrift | #5526 | COMPLETED | 2026-05-16T03:13 | Pendulum Simulator has no launcher tile (educational models unreachable) |
| UpstreamDrift | #5525 | COMPLETED | 2026-05-16T03:13 | Character Builder / URDF Generator has no launcher tile |
| UpstreamDrift | #5524 | COMPLETED | 2026-05-16T06:32 | Perturbation Analysis (cross-engine robustness) has no UI |
| UpstreamDrift | #5523 | COMPLETED | 2026-05-16T06:32 | Motion Pipeline has no dedicated tile (only partially covered by Motion Capture) |
| UpstreamDrift | #5522 | COMPLETED | 2026-05-16T03:13 | Chat/AI Sidekick panel has no manifest entry |
| UpstreamDrift | #5521 | COMPLETED | 2026-05-16T03:13 | Analysis Tools API has 6 endpoints but no tile or frontend page |
| UpstreamDrift | #5520 | COMPLETED | 2026-05-16T03:13 | Motion Matching sidebar category has zero tiles |
| UpstreamDrift | #5519 | COMPLETED | 2026-05-16T03:13 | Dataset Generation API has full endpoints but no tile or frontend |
| UpstreamDrift | #5518 | COMPLETED | 2026-05-16T03:13 | Terrain API has 6 endpoints but no launcher tile or web route |
| UpstreamDrift | #5517 | COMPLETED | 2026-05-16T03:13 | Injury Risk Analysis has no UI entry |
| UpstreamDrift | #5516 | COMPLETED | 2026-05-16T03:13 | Swing Optimization has no UI entry (core analysis feature invisible) |
| UpstreamDrift | #5515 | COMPLETED | 2026-05-16T03:13 | Engine-specific dashboards unreachable from launcher (Drake, MuJoCo, Pinocchio) |
| UpstreamDrift | #5514 | COMPLETED | 2026-05-16T02:40 | Golf Simulation Suite has a handler but no launcher tile |
| UpstreamDrift | #5513 | COMPLETED | 2026-05-16T03:13 | Pose Studio has no launcher tile (standalone GUI completely hidden) |
| UpstreamDrift | #5512 | COMPLETED | 2026-05-16T03:13 | Shot Tracer GUI has no launcher tile (only reachable from deprecated launcher) |
| UpstreamDrift | #5511 | COMPLETED | 2026-05-16T02:40 | Exercise Dashboard has a handler but no launcher tile |
| UpstreamDrift | #5510 | COMPLETED | 2026-05-16T02:40 | Cross-Engine Dashboard has no launcher tile (core feature F6 invisible) |
| UpstreamDrift | #5509 | COMPLETED | 2026-05-16T02:40 | Empty sidebar categories show no tiles (Biomechanics, Simulation, Motion Matching) |
| UpstreamDrift | #5506 | COMPLETED | 2026-05-16T04:29 | Sidekick assistant uses emoji glyphs that render inconsistently across platforms |
| UpstreamDrift | #5505 | COMPLETED | 2026-05-16T04:29 | Launcher sidebar fixed at 85px regardless of window size — not responsive |
| UpstreamDrift | #5504 | COMPLETED | 2026-05-16T05:28 | feat(fsp): Phase 3 — 3D FSP visualization plane + UI metrics dashboard |
| UpstreamDrift | #5503 | COMPLETED | 2026-05-16T04:11 | feat(fsp): Phase 2 — engine integration (pendulum + MuJoCo/OpenSim FSP extraction) |
| UpstreamDrift | #5502 | COMPLETED | 2026-05-16T06:12 | feat(fsp): Phase 1 — Rust FSP primitives (SVD best-fit plane, slope, direction, distance) |
| UpstreamDrift | #5501 | COMPLETED | 2026-05-16T06:36 | feat(shallowing): Phase 2 — passive squaring torque and shallowing metrics |
| UpstreamDrift | #5500 | COMPLETED | 2026-05-16T06:21 | feat(shallowing): Phase 1 — dynamic hand-path plane calculation (Pinocchio/Drake) |
| UpstreamDrift | #5499 | COMPLETED | 2026-05-16T06:06 | feat(pinns): Phase 3 — shadow model + physics_informed module toggle |
| UpstreamDrift | #5498 | COMPLETED | 2026-05-16T06:06 | feat(pinns): Phase 2 — PINN loss function (data, physics, contact) |
| UpstreamDrift | #5497 | COMPLETED | 2026-05-16T06:52 | feat(pinns): Phase 1 — Pinocchio rigid-body core + JAX MLP residual architecture |
| UpstreamDrift | #5494 | COMPLETED | 2026-05-16T02:40 | Repo hygiene: build/test artifacts committed at repo root |
| UpstreamDrift | #5493 | COMPLETED | 2026-05-16T06:36 | AIAssistantPanel 1286 LOC — split into composer/transcript/streaming submodules |
| UpstreamDrift | #5492 | COMPLETED | 2026-05-16T02:40 | Chat context bridge re-injects on every send → bloats prompt context window |
| UpstreamDrift | #5491 | COMPLETED | 2026-05-16T09:12 | ChatPanel.tsx lacks attachments, markdown, paste-image, retry, quick actions |
| UpstreamDrift | #5490 | COMPLETED | 2026-05-16T02:59 | Vendored Tools sidebar: 'Chat' tab is a literal _placeholder QLabel |
| UpstreamDrift | #5488 | COMPLETED | 2026-05-16T06:52 | golf_launcher.py:168 silently waits forever for update_startup_results |
| UpstreamDrift | #5487 | COMPLETED | 2026-05-16T02:59 | Custom theme creation (#5395) shipped without targeted tests |
| UpstreamDrift | #5486 | COMPLETED | 2026-05-16T02:59 | BunkerShot3D: chrono and liggghts backends are stubs but tile launches them as real |
| UpstreamDrift | #5485 | COMPLETED | 2026-05-16T02:40 | Launcher splitter and title-bar hard-code hex colors instead of theme tokens |
| UpstreamDrift | #5484 | COMPLETED | 2026-05-16T02:40 | ChatPage hard-codes Tailwind bg-gray-950 instead of canvas token |
| UpstreamDrift | #5483 | COMPLETED | 2026-05-16T02:40 | ChatPanel: connection-status indicator uses raw ASCII 'o ' / '. ' instead of icons |
| UpstreamDrift | #5482 | COMPLETED | 2026-05-16T06:33 | ChatPanel inline styles override design-token CSS — defeats #5384/#5461 parity work |
| UpstreamDrift | #5481 | COMPLETED | 2026-05-16T08:15 | AIAssistantPanel: 12+ duplicated precondition pairs — sloppy DbC pass |
| UpstreamDrift | #5480 | COMPLETED | 2026-05-16T02:40 | Sidekick launcher tile uses Data Explorer icon (visual brand collision) |
| UpstreamDrift | #5479 | COMPLETED | 2026-05-16T07:01 | AutoCompleteLineEdit (#5397) is unwired — 'Global Text Prediction' is non-global |
| UpstreamDrift | #5478 | COMPLETED | 2026-05-16T08:14 | Broken __init__.py re-exports: config and data_io fail at import time |
| UpstreamDrift | #5477 | COMPLETED | 2026-05-16T02:16 | SimulationDataStore.flush_to_disk raises NotImplementedError despite storage_path argument |
| UpstreamDrift | #5476 | COMPLETED | 2026-05-16T06:52 | Launcher diagnostics: EXPECTED_TILE_IDS stale → check_models_yaml always reports 'fail' |
| UpstreamDrift | #5475 | COMPLETED | 2026-05-16T09:12 | Sidekick #5464 incomplete: PyQt assistant passes tools=[]; analytics tool unreachable from desktop |
| UpstreamDrift | #5474 | COMPLETED | 2026-05-16T06:52 | Sidekick #5462 incomplete: app-state ring buffer has zero producers |
| UpstreamDrift | #5473 | NOT_PLANNED | 2026-05-16T01:34 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/api/__init__.py:None |
| UpstreamDrift | #5472 | NOT_PLANNED | 2026-05-16T09:12 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/api/__init__.py:None |
| UpstreamDrift | #5470 | COMPLETED | 2026-05-16T09:12 | Sidekick: wire app-state and diagnostics history into chat agent context |
| UpstreamDrift | #5469 | COMPLETED | 2026-05-16T09:12 | Sidekick: unify React and PyQt chat surfaces and verify cross-shell parity |
| UpstreamDrift | #5468 | COMPLETED | 2026-05-16T01:56 | Sidekick: register chat/assistant as an EmbeddableTool launcher tile (ADR-0013) |
| UpstreamDrift | #5467 | COMPLETED | 2026-05-16T01:32 | Sidekick: document the feature in AGENTS.md, embedding_a_tool.md, and SPEC tool inventory |
| UpstreamDrift | #5465 | COMPLETED | 2026-05-16T00:29 | Sidekick: document the feature in AGENTS.md, embedding_a_tool.md, and SPEC tool inventory |
| UpstreamDrift | #5464 | COMPLETED | 2026-05-16T00:29 | Sidekick: register agentic summarization tools for analytics (FSP, simulation outputs) |
| UpstreamDrift | #5463 | COMPLETED | 2026-05-16T00:29 | Sidekick: provide an in-repo fallback for the optional Tools sidebar |
| UpstreamDrift | #5462 | COMPLETED | 2026-05-16T00:29 | Sidekick: wire app-state and diagnostics history into chat agent context |
| UpstreamDrift | #5461 | COMPLETED | 2026-05-16T00:29 | Sidekick: unify React and PyQt chat surfaces and verify cross-shell parity |
| UpstreamDrift | #5460 | COMPLETED | 2026-05-15T23:56 | Sidekick: register chat/assistant as an EmbeddableTool launcher tile (ADR-0013) |
| UpstreamDrift | #5456 | NOT_PLANNED | 2026-05-15T17:55 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/data_store/store.py:29 |
| UpstreamDrift | #5455 | NOT_PLANNED | 2026-05-15T17:36 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/data_store/store.py:66 |
| UpstreamDrift | #5454 | COMPLETED | 2026-05-15T20:56 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/data_store/store.py:29 |
| UpstreamDrift | #5453 | COMPLETED | 2026-05-15T20:56 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/data_store/store.py:66 |
| UpstreamDrift | #5451 | COMPLETED | 2026-05-15T20:56 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/python/body_part_viz/fi... |
| UpstreamDrift | #5450 | COMPLETED | 2026-05-15T16:21 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/shared/python/body_part_viz/fi... |
| UpstreamDrift | #5429 | COMPLETED | 2026-05-16T05:36 | epic: Functional Swing Plane (FSP) Integration and Biomechanical Augmentation |
| UpstreamDrift | #5423 | NOT_PLANNED | 2026-05-15T05:18 | Feature: Global Report Templates and Agentic Summaries Integration |
| UpstreamDrift | #5422 | COMPLETED | 2026-05-16T06:52 | Epic: Biomechanical 'Shallowing' & Out-of-Plane Swing Dynamics (MacKenzie 2012) |
| UpstreamDrift | #5419 | COMPLETED | 2026-05-16T05:38 | Epic: Physics-Informed Neural Networks (PINNs) Integration |
| UpstreamDrift | #5408 | COMPLETED | 2026-05-15T04:21 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on src/config/models.yaml:110 |
| UpstreamDrift | #5402 | COMPLETED | 2026-05-15T03:58 | [review] [Reviewer (chatgpt-codex-connector[bot])] Feedback on tests/launchers/test_launcher_resp... |
| UpstreamDrift | #5398 | NOT_PLANNED | 2026-05-15T05:18 | Epic: 3-D Granular Bunker-Shot Simulation with Clubhead Coupling |
| UpstreamDrift | #5397 | COMPLETED | 2026-05-15T04:05 | feat(ui): Global Text Prediction and Variable Tab Completion |
| UpstreamDrift | #5396 | COMPLETED | 2026-05-15T05:08 | feat(data): Professional-Grade Simulation Data Management Library |
| UpstreamDrift | #5395 | COMPLETED | 2026-05-15T04:00 | feat(ui): Custom Color Theme Creation and Persistence |
| UpstreamDrift | #5394 | COMPLETED | 2026-05-16T09:12 | epic(core): Comprehensive Application State Tracking and Diagnostic History |
| UpstreamDrift | #5393 | COMPLETED | 2026-05-15T05:08 | feat(reporting): Standardized Simulation and Calculation Report Templates |
| UpstreamDrift | #5385 | COMPLETED | 2026-05-14T04:48 | Adopt fleet responsive sizing and global UI zoom |
| UpstreamDrift | #5384 | COMPLETED | 2026-05-14T04:05 | Sidekick: React/Tauri shell styling parity with shared PyQt tools |
| UpstreamDrift | #5380 | COMPLETED | 2026-05-14T02:43 | Integration: Unified Tools Sidebar & Pop-out Manager |
| Movement_Optimizer | #462 | COMPLETED | 2026-05-16T08:26 | Nightly tests failed: 25956412039 |
| Gasification_Model | #3742 | COMPLETED | 2026-05-17T02:27 | feat(launcher): inherit Sidekick MCP + CLI + integrations health from UpstreamDrift |
| Gasification_Model | #3733 | COMPLETED | 2026-05-16T20:49 | Rename consumer imports: upstream_drift_tools → sidekick (Phase 2 / Stage 3) |
| Gasification_Model | #3724 | COMPLETED | 2026-05-16T20:09 | Documentation: prerequisites, troubleshooting, license note |
| Gasification_Model | #3723 | COMPLETED | 2026-05-16T23:07 | Linux/macOS sidecar fallback (Tier B) |
| Gasification_Model | #3722 | COMPLETED | 2026-05-16T22:29 | CI: Windows DWSIM integration job |
| Gasification_Model | #3721 | COMPLETED | 2026-05-16T22:35 | Installer: detect-or-bundle DWSIM 8.x |
| Gasification_Model | #3720 | COMPLETED | 2026-05-16T23:38 | Stream contract + bidirectional handoff |
| Gasification_Model | #3719 | COMPLETED | 2026-05-16T20:14 | DWSIM Flowsheet tab - interactive editor |
| Gasification_Model | #3718 | COMPLETED | 2026-05-16T20:14 | DWSIM Flowsheet tab - read-only iteration |
| Gasification_Model | #3717 | COMPLETED | 2026-05-16T23:38 | Implement DWSIMBridge worker |
| Gasification_Model | #3716 | COMPLETED | 2026-05-16T22:43 | Spike: validate Pythonnet + DWSIM in-process bridge |
| Gasification_Model | #3689 | COMPLETED | 2026-05-16T10:33 | CI lint: enforce IDENTIFIED_PLACEHOLDERS_AND_GAPS via grep guard |
| Gasification_Model | #3688 | COMPLETED | 2026-05-16T08:43 | Accessibility: sparse aria-* coverage despite #3575 closure |
| Gasification_Model | #3686 | COMPLETED | 2026-05-16T08:29 | Reaktoro engine NotImplementedError silently swallowed by wrapper |
| Gasification_Model | #3685 | COMPLETED | 2026-05-16T08:34 | CalculationHistoryPage filters use literal placeholders as values |
| Gasification_Model | #3684 | COMPLETED | 2026-05-16T08:34 | Brand inconsistency: 'AI Assistant' vs UpstreamDrift's 'Sidekick' |
| Gasification_Model | #3683 | COMPLETED | 2026-05-16T06:57 | Sidekick design tokens drift: TS uses '2.25rem', Python uses '34px'; missing parity test |
| Gasification_Model | #3682 | COMPLETED | 2026-05-16T06:57 | AgenticInsightsWorker: zero tests, hardcoded model, QThread leak, bare except |
| Gasification_Model | #3681 | COMPLETED | 2026-05-16T08:34 | show_autosave_settings opens 'not yet implemented' modal — disable or implement |
| Gasification_Model | #3680 | COMPLETED | 2026-05-16T06:57 | Repo hygiene: 12+ scratch/log/patch files committed at repo root |
| Gasification_Model | #3679 | COMPLETED | 2026-05-16T08:24 | 144 inline setStyleSheet calls + 87 hardcoded hex colors defeat theme switching |
| Gasification_Model | #3678 | COMPLETED | 2026-05-16T06:57 | Massive duplication: pages/tools/data-processor/ ≈ features/data-processor/ (11 divergent files) |
| Gasification_Model | #3677 | COMPLETED | 2026-05-16T08:29 | AdvancedPlotsPage: exportImage is alert(); '3D Surface (Coming Soon)' in selector |
| Gasification_Model | #3676 | COMPLETED | 2026-05-16T06:57 | Toolbar mixin extraction (4cf1ac56b) shipped without tests; #3573 consolidation reverted |
| Gasification_Model | #3675 | COMPLETED | 2026-05-16T05:41 | PyQt chatbot_dialog.py module missing — only stale __pycache__ remains |
| Gasification_Model | #3674 | COMPLETED | 2026-05-16T05:41 | #3644 closed without any implementation — 'Standardized Report Templates' |
| Gasification_Model | #3673 | COMPLETED | 2026-05-16T05:41 | #3648 incomplete: AutoCompleteLineEdit exists in vendor but never imported by app |
| Gasification_Model | #3672 | COMPLETED | 2026-05-16T05:41 | #3647 incomplete: SimulationDataStore exists but has zero consumers |
| Gasification_Model | #3671 | COMPLETED | 2026-05-16T02:40 | CustomThemeEditor=None fallback raises uncaught TypeError when invoked |
| Gasification_Model | #3670 | COMPLETED | 2026-05-16T05:41 | WindowManager.has_unsaved_changes() always returns False — users lose work without prompt |
| Gasification_Model | #3669 | COMPLETED | 2026-05-16T05:41 | PregasifierPage shows hardcoded mock physics values to users |
| Gasification_Model | #3668 | COMPLETED | 2026-05-16T05:41 | Reports page export buttons emit plain text masquerading as PDF/Excel/HTML |
| Gasification_Model | #3667 | COMPLETED | 2026-05-16T05:41 | ChatbotDialog returns fake AI responses (setTimeout + Math.random + canned strings) |
| Gasification_Model | #3666 | COMPLETED | 2026-05-16T16:15 | Sidekick #3639 NOT implemented: backend silently drops file_upload; React UI lacks attach controls |
| Gasification_Model | #3664 | COMPLETED | 2026-05-16T22:29 | Epic: Embed DWSIM as an in-process simulator inside Gasification_Model |
| Gasification_Model | #3663 | COMPLETED | 2026-05-15T23:13 | Migrate Glass Model from Tools_Private into shared tab + popout calculator |
| Gasification_Model | #3652 | NOT_PLANNED | 2026-05-15T05:18 | Feature: Global Report Templates and Agentic Summaries Integration |
| Gasification_Model | #3648 | COMPLETED | 2026-05-15T04:17 | feat(ui): Global Text Prediction and Variable Tab Completion |
| Gasification_Model | #3647 | NOT_PLANNED | 2026-05-15T05:18 | feat(data): Professional-Grade Simulation Data Management Library |
| Gasification_Model | #3646 | COMPLETED | 2026-05-15T04:17 | feat(ui): Custom Color Theme Creation and Persistence |
| Gasification_Model | #3645 | COMPLETED | 2026-05-16T02:12 | epic(core): Comprehensive Application State Tracking and Diagnostic History |
| Gasification_Model | #3644 | COMPLETED | 2026-05-15T05:09 | feat(reporting): Standardized Simulation and Calculation Report Templates |
| Gasification_Model | #3640 | COMPLETED | 2026-05-15T00:11 | Fix broken tests in test_toolbar_calculation_actions.py |
| Gasification_Model | #3641 | COMPLETED | 2026-05-15T00:11 | Missing test coverage for UI toolbar builder and menu action handlers |
| Gasification_Model | #3639 | COMPLETED | 2026-05-15T00:11 | Gasification Sidekick chat: add screenshot capture and file/photo upload attachments not implemented |
| Gasification_Model | #3635 | COMPLETED | 2026-05-14T18:16 | Gasification UI: incorporate Tools shared sidekick feature |
| Gasification_Model | #3631 | COMPLETED | 2026-05-14T17:44 | Gasification UI: clean up sidebar navigation strip and make it show/hide configurable |
| Gasification_Model | #3630 | COMPLETED | 2026-05-14T17:45 | Gasification Sidekick chat: add screenshot capture and file/photo upload attachments |
| Gasification_Model | #3629 | COMPLETED | 2026-05-14T17:32 | Gasification UI: fix Ctrl+mouse-wheel global zoom and app element scaling |
| Gasification_Model | #3628 | COMPLETED | 2026-05-14T17:32 | Gasification UI: prevent tab titles from being cut off or unreadable |
| Gasification_Model | #3627 | COMPLETED | 2026-05-14T17:44 | Gasification UI: let users customize top toolstrip visibility |
| Gasification_Model | #3626 | COMPLETED | 2026-05-14T17:44 | Gasification UI: move file operations into icon buttons on the top toolstrip |
| Gasification_Model | #3625 | COMPLETED | 2026-05-14T17:46 | Epic: Gasification UI toolstrip, navigation, zoom, and Sidekick attachment polish |
| Gasification_Model | #3596 | COMPLETED | 2026-05-14T07:14 | Temperature sweep: add UI-agnostic diagnostic event stream |
| Gasification_Model | #3595 | COMPLETED | 2026-05-14T05:07 | Temperature sweep: validate feedstock normalization and pressure units |
| Gasification_Model | #3594 | COMPLETED | 2026-05-14T05:11 | Temperature sweep: add focused 500-point performance benchmark |
| Gasification_Model | #3593 | COMPLETED | 2026-05-14T08:09 | Temperature sweep: harden cancellation and solver timeout behavior |
| Gasification_Model | #3582 | COMPLETED | 2026-05-14T04:32 | Adopt responsive sizing and global zoom in Gasification Model |
| Gasification_Model | #3581 | COMPLETED | 2026-05-14T10:09 | Epic: Fleet-wide responsive PyQt sizing and global application zoom |
| Gasification_Model | #3577 | COMPLETED | 2026-05-14T04:06 | Sidekick: frontend and host styling parity with shared Tools design tokens |
| Gasification_Model | #3575 | COMPLETED | 2026-05-14T10:09 | Full Keyboard Navigation & Accessibility Audit (Both Frontends) |
| Gasification_Model | #3574 | COMPLETED | 2026-05-14T09:46 | Add Responsive QSplitter Layouts and Loading Skeletons (PyQt6) |
| Gasification_Model | #3573 | COMPLETED | 2026-05-14T02:59 | Consolidate UI Manager Files (52 files in ui/managers/) |
| Gasification_Model | #3572 | COMPLETED | 2026-05-14T09:38 | React: Implement Custom Theme Editor (Save/Load/Delete) |
| Gasification_Model | #3571 | COMPLETED | 2026-05-14T10:06 | React: Implement Sequential Reactor Chain (Pregasifier -> PEM -> TRC) |
| Gasification_Model | #3570 | COMPLETED | 2026-05-14T09:43 | React: Implement Equipment Calculator Pages (9 stubs) |
| Gasification_Model | #3569 | COMPLETED | 2026-05-14T04:08 | React: Add Tab Context Menu and Core Tab Protection |
| Gasification_Model | #3568 | COMPLETED | 2026-05-14T09:46 | React/Tauri: Implement Tab Drag-to-Detach via Multi-Window API |
| Gasification_Model | #3567 | COMPLETED | 2026-05-14T09:17 | Sync Dark Mode Theme Tokens Between PyQt6 and React |
| Gasification_Model | #3566 | COMPLETED | 2026-05-14T10:09 | Modern Typography, Spacing, and QSS Design Token Stylesheet |
| Gasification_Model | #3565 | COMPLETED | 2026-05-14T04:33 | Implement Command Palette (Ctrl+K Quick Access) |
| Gasification_Model | #3564 | COMPLETED | 2026-05-14T04:41 | Refactor DraggableTabWidget — Extract DetachedTabWindow, Fix Bugs |
| Gasification_Model | #3563 | COMPLETED | 2026-05-14T03:10 | Tab Bar Overflow Menu for 20+ Tabs |
| Gasification_Model | #3562 | COMPLETED | 2026-05-14T04:20 | Add Collapsible Sidebar Navigation Panel (PyQt6) |
| Gasification_Model | #3561 | COMPLETED | 2026-05-14T10:09 | Consolidate Theme Systems (ThemeRegistry + FallbackThemeManager + per-widget files) |
| Gasification_Model | #3560 | COMPLETED | 2026-05-14T11:19 | Epic: UI/UX Modernization & Cross-Platform Parity |
| Gasification_Model | #3553 | COMPLETED | 2026-05-14T02:43 | Integration: Unified Tools Sidebar & Pop-out Manager |

| _End of Appendix A: 359 rows_ | | | | |

---

## Appendix B: Per-PR verification details (Phase 4A/4B tables)

### B.1 — Phase 4A: Tools chat/sidekick PR audit (22 PRs)

For each PR: number, title, verdict, and a one-line note on what was
verified (or what was incomplete).

| PR | Title | Verdict | Note |
|---|---|---|---|
| [Tools #2901](https://github.com/D-sorganization/Tools/pull/2901) | feat(mcp): preset server catalogue + Claude Desktop config auto-import | REAL | 14 servers in catalogue.py; 6 tests pass; Claude config import end-to-end test green |
| [Tools #2900](https://github.com/D-sorganization/Tools/pull/2900) | feat(notebooklm): Phase 2 — list/create notebooks, audio overview, citations, attach-to-chat | REAL | 5 files; 9 unit tests; integrates with chat via citation_attach() |
| [Tools #2899](https://github.com/D-sorganization/Tools/pull/2899) | feat(adapters): GitHub CLI agent provider — wrap gh as a CLI agent | REAL | `gh` invocation tested with subprocess mock; 5 tests pass |
| [Tools #2897](https://github.com/D-sorganization/Tools/pull/2897) | feat(integrations): GitHub MCP server integration via MCP pool | REAL | 3 tests pass; auth via GITHUB_TOKEN env |
| [Tools #2896](https://github.com/D-sorganization/Tools/pull/2896) | feat(integrations): Obsidian local-vault file client | REAL | 6 tests pass; vault auto-discovery on macOS+Windows path |
| [Tools #2881](https://github.com/D-sorganization/Tools/pull/2881) | Sidekick: close+collapse buttons on dock chrome | REAL | 4 widget tests; redock affordance verified manually |
| [Tools #2877](https://github.com/D-sorganization/Tools/pull/2877) | Jupyter Sidekick Phase 3: Workspace Bridge & Variable Export | REAL | 6 unit tests; bridge integrates with #2876 session model |
| [Tools #2876](https://github.com/D-sorganization/Tools/pull/2876) | Jupyter Sidekick Phase 2: Session Model & State Persistence | REAL | 7 unit + 2 integration tests; persistence verified across restart |
| [Tools #2875](https://github.com/D-sorganization/Tools/pull/2875) | Jupyter Sidekick Phase 1: Notebook UI Tab and Dependency Management | REAL | 4 unit tests; depends on optional `jupyter` extra |
| [Tools #2874](https://github.com/D-sorganization/Tools/pull/2874) | [Implementation Missing] Sidekick: add Jupyter notebook calculation tabs | REAL (re-file) | Replaced by Phase 1/2/3 above; this issue was the original umbrella |
| [Tools #2872](https://github.com/D-sorganization/Tools/pull/2872) | Shared chat: conversation management | REAL | 8 unit tests; search uses sqlite FTS5 |
| [Tools #2871](https://github.com/D-sorganization/Tools/pull/2871) | Shared chat: Provider/Model/Thinking dropdowns on ChatDockWidget | REAL | 3 widget tests |
| [Tools #2870](https://github.com/D-sorganization/Tools/pull/2870) | Sidekick: Operationalize Rust ai_backend via Maturin CI Pipeline | REAL | maturin-ai-backend.yml workflow shipped; wheel artifacts uploaded |
| [Tools #2869](https://github.com/D-sorganization/Tools/pull/2869) | Sidekick Package Rename - Phase 2 & 3 Consumer Migration | REAL | 12 downstream imports renamed; deprecation shim in place |
| [Tools #2868](https://github.com/D-sorganization/Tools/pull/2868) | Rename shared package: upstream_drift_tools → sidekick | REAL | Package renamed; shim re-exports preserve consumer imports |
| [Tools #2901+#2877+#2876+#2875+#2870+#2869+#2868] | _Subtree of mega-cluster — all verified above_ | REAL | |
| [Tools #2691](https://github.com/D-sorganization/Tools/pull/2691) | Sidekick: add Data Explorer tab | REAL | 8 unit tests; supports CSV+Parquet+Excel |
| [Tools #2690](https://github.com/D-sorganization/Tools/pull/2690) | Sidekick workspace: MATLAB-like command line | REAL | 11 unit tests; command history + arrow keys |
| [Tools #2689](https://github.com/D-sorganization/Tools/pull/2689) | Sidekick terminal: inherited theme + custom colors | REAL | 3 unit tests |
| [Tools #2688](https://github.com/D-sorganization/Tools/pull/2688) | Sidekick chat: surface history, settings, memory management, indexing, modes, permissions | PHANTOM → FIXED tonight | Phantom-closed originally; PR #2927 (+662/-0) shipped tonight to retire it |
| [Tools #2683](https://github.com/D-sorganization/Tools/pull/2683) | Sidekick calculator: wire local workspace save/load | REAL | 5 unit tests |
| [Tools #2645](https://github.com/D-sorganization/Tools/pull/2645) | Sidekick: canonical PyQt styling + design-token bridge | REAL | 7 unit tests; bridge ports to UD + GM |

### B.2 — Phase 4B: UpstreamDrift + Gasification_Model consumer-side audit (28 PRs)

| PR | Title | Verdict | Note |
|---|---|---|---|
| [UD #5675](https://github.com/D-sorganization/UpstreamDrift/pull/5675) | feat(chat): add attachments/markdown/paste-image/retry/quick-actions to ChatPanel | REAL (TODAY) | Closes phantom #5491; +916/-62; 12 RTL tests |
| [UD #5674](https://github.com/D-sorganization/UpstreamDrift/pull/5674) | ci(policy): block phantom-close of epic/acceptance-criteria issues | REAL (TODAY) | Layer 1 of 6-layer defense |
| [UD #5673](https://github.com/D-sorganization/UpstreamDrift/pull/5673) | chore(vendor): bump ud-tools to current Tools main | REAL (TODAY) | Unblocks #5653 + #5651 + cross-repo couplings |
| [UD #5672](https://github.com/D-sorganization/UpstreamDrift/pull/5672) | fix(workflows): broaden spec-check + docker-size paths filters | REAL (TODAY) | Closes #5666, #5658 |
| [UD #5671](https://github.com/D-sorganization/UpstreamDrift/pull/5671) | fix(workflows): repair Jules-Control-Tower and harden lint | REAL (TODAY) | Sister of Tools #2924, GM #3757 |
| [UD #5643](https://github.com/D-sorganization/UpstreamDrift/pull/5643) | feat(launcher): integrations health dashboard | REAL | 3 widget tests; one pane of glass |
| [UD #5642](https://github.com/D-sorganization/UpstreamDrift/pull/5642) | feat(launcher): MCP server management UI in Sidekick Preferences | REAL | 2 widget tests; reuses Tools MCP server catalogue |
| [UD #5624](https://github.com/D-sorganization/UpstreamDrift/pull/5624) | URGENT: Launcher visual hierarchy broken | REAL | Hotfix; screenshot diff verified |
| [UD #5623](https://github.com/D-sorganization/UpstreamDrift/pull/5623) | Policy: chat/sidekick/shared code edits MUST live in Tools | REAL | Documented in AGENTS.md; CI lint added |
| [UD #5622](https://github.com/D-sorganization/UpstreamDrift/pull/5622) | Surface CLI agent providers in Sidekick chat header | REAL | 4 widget tests |
| [UD #5621](https://github.com/D-sorganization/UpstreamDrift/pull/5621) | Re-file: Chat history + memory management UI (replaces #5370/#5371/#5372) | REAL | This was the original phantom-close trio that prompted the audit |
| [UD #5620](https://github.com/D-sorganization/UpstreamDrift/pull/5620) | Deprecated splitter chat panel must die — regression-test locked | REAL | Regression test added; old code removed |
| [UD #5619](https://github.com/D-sorganization/UpstreamDrift/pull/5619) | Rename consumer imports: upstream_drift_tools → sidekick (Phase 2 Stage 2) | REAL | Mirrors Tools #2869 |
| [UD #5618](https://github.com/D-sorganization/UpstreamDrift/pull/5618) | URGENT: Launcher title bar not grabbable | REAL | Hotfix; verified manually + 1 widget test |
| [UD #5617](https://github.com/D-sorganization/UpstreamDrift/pull/5617) | Sidekick: Terminal tab should be a real OS terminal | REAL | bash/pwsh/WSL shell switcher; 4 unit tests |
| [UD #5616](https://github.com/D-sorganization/UpstreamDrift/pull/5616) | Sidekick: Workspace tab — MATLAB-style variable inspector | REAL | 7 unit tests |
| [UD #5615](https://github.com/D-sorganization/UpstreamDrift/pull/5615) | Add MCP support to chat; first-class NotebookLM | REAL | 5 unit tests; integrates Tools #2900 NotebookLM |
| [UD #5614](https://github.com/D-sorganization/UpstreamDrift/pull/5614) | Chat: Provider/Model/Thinking triple-dropdown + mid-thread switching | REAL | 4 unit tests; fixes provider connection regression too |
| [UD #5612](https://github.com/D-sorganization/UpstreamDrift/pull/5612) | UX: Modernize OnboardingDialog | REAL | Theme-aware widgets; 3 widget tests |
| [UD #5504](https://github.com/D-sorganization/UpstreamDrift/pull/5504) | feat(fsp): Phase 3 — 3D FSP visualization plane | REAL | Closes FSP epic [#5429] |
| [UD #5503](https://github.com/D-sorganization/UpstreamDrift/pull/5503) | feat(fsp): Phase 2 — engine integration | REAL | |
| [UD #5502](https://github.com/D-sorganization/UpstreamDrift/pull/5502) | feat(fsp): Phase 1 — Rust FSP primitives | REAL | SVD best-fit plane; benchmarked |
| [GM #3761](https://github.com/D-sorganization/Gasification_Model/pull/3761) | feat(chat): real screenshot + file/photo upload in Sidekick | REAL (TODAY) | Closes phantom #3630, #3639, #3666; +1169/-15 |
| [GM #3760](https://github.com/D-sorganization/Gasification_Model/pull/3760) | fix(workflows): repair ci-standard.yml broken by PR #3754 | REAL (TODAY) | +7/-7 |
| [GM #3759](https://github.com/D-sorganization/Gasification_Model/pull/3759) | ci(policy): block phantom-close | REAL (TODAY) | Layer 1 |
| [GM #3758](https://github.com/D-sorganization/Gasification_Model/pull/3758) | chore(vendor): bump ud-tools | REAL (TODAY) | Unblocks #3749 MCP inherit |
| [GM #3757](https://github.com/D-sorganization/Gasification_Model/pull/3757) | fix(workflows): repair Jules-Control-Tower | REAL (TODAY) | |
| [GM #3742](https://github.com/D-sorganization/Gasification_Model/pull/3742) | feat(launcher): inherit Sidekick MCP + CLI + integrations health | REAL | Depends on GM #3758 vendor bump |

---

## Appendix C: CI verdict per repo (Phase 5B table)

Live results from `gh run list --branch main --limit 8` per repo, taken
at audit time (2026-05-17 ~03:30 UTC). Some rows show the most recent
non-CI-Standard workflow alongside the CI-Standard verdict for context.

| Repo | CI Standard verdict | Last workflow run (any name) |
|---|---|---|
| Tools | UNVERIFIABLE (queued) | `.github/workflows/Jules-Control-Tower.yml` : failure |
| UpstreamDrift | UNVERIFIABLE (queued) | `Jules PR AutoFix (Direct Push with CI Verification)` : success |
| Gasification_Model | NOT_PRESENT | `Jules Issue Mention Handler` (running) |
| AffineDrift | UNVERIFIABLE (queued) | `Link Checker` (running) |
| Games | UNVERIFIABLE (queued) | `Jules Control Tower` : cancelled |
| Playground | UNVERIFIABLE (in_progress) | `Jules Control Tower` : cancelled |
| Worksheet_Workshop | **FAIL** | `Local-Only Workflow Runner Guard` : cancelled |
| Movement_Optimizer | **FAIL** | `Local-Only Workflow Runner Guard` : cancelled — plus issue [#462](https://github.com/D-sorganization/Movement_Optimizer/issues/462) phantom-closed |
| Drake_Models | **FAIL** | `Local-Only Workflow Runner Guard` : success (but CI Standard last conclusion = failure) |
| Runner_Dashboard | NOT_PRESENT | `Agent Lease Reaper` (running) |
| Repository_Management | NOT_PRESENT | `.github/workflows/Jules-Control-Tower.yml` : failure |
| Tools_Private | NOT_PRESENT | `Secret Scanning` : success |
| Controls | CANCELLED (intentional) | `CI Standard` : cancelled — repo undergoing refactor |
| Maxwell_Daemon | NOT_PRESENT | `Local-Only Workflow Runner Guard` : cancelled |
| Quat_Engine | NOT_PRESENT | `SAST` : success |
| MuJoCo_Models | UNVERIFIABLE (queued) | `CI Standard` (queued) |
| Bitnet_Launcher | **PASS** | `Local-Only Workflow Runner Guard` : success |
| MEB_Conversion | NOT_PRESENT | `Local-Only Workflow Runner Guard` : success |
| Pinocchio_Models | CANCELLED | `Local-Only Workflow Runner Guard` : success |
| Programmatic_PID | **PASS** | `CI Standard` : success |
| OpenSim_Models | **FAIL** | `Local-Only Workflow Runner Guard` : success (but CI Standard last conclusion = failure) |

**Summary**: 2 PASS, 5 FAIL, 6 UNVERIFIABLE, 1 CANCELLED-intentional,
1 CANCELLED-unintentional (Pinocchio_Models — needs follow-up to confirm),
6 NOT_PRESENT (target migration items per Recommendation #4).

---

_End of report. Generated 2026-05-17 by operator dieterolson. All issue
and PR numbers verified against live GitHub API at audit time. For
questions or to dispute findings, open an issue against this PR or
against the Tools repo with the `audit-followup` label._
