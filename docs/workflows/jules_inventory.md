# Jules-\* workflow inventory (Tools)

Snapshot taken 2026-09-03 for Tools #4923 (Fleet Readiness Program,
Repository_Management#1505 Phase 1) as input to the fleet retirement campaign
Repository_Management#1483. **The retirement has since been executed** - see
"Decision (Repository_Management#1483)" at the foot of this file. Acceptance for #4923 is "Jules workflow
count documented and <= 5 kept" — this inventory recommends keeping 5.

Method: `ls .github/workflows/Jules-*.yml jules-assessment-runner.yml`, triggers
and `permissions:` parsed from the YAML, last run via
`gh run list --workflow <file> --limit 1` (date + conclusion). Documented owner
means a named person/team in the YAML header or in `docs/`; none of the 28 has
one. "Token permissions" lists the top-level `permissions:` block and any
job-level escalation. Runs dated 2026-05-30 05:04-05:05 are one burst of
28 failures from a single dispatch and count as "not in use since".

| Workflow | Trigger | Last run (UTC) | Token permissions (top-level / job) | Owner | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Jules-Archivist.yml | workflow_call | 2026-05-30 failure | contents: read / archive-tasks: contents: write | none | retire |
| Jules-Assessment-Generator.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents: read / generate-assessments: contents+issues: write | none | retire |
| Jules-Assessment-Remediator.yml | workflow_dispatch, schedule | 2026-08-02 failure | contents: read / remediate: contents+pull-requests+issues: write | none | retire (scheduled, failing) |
| Jules-Auto-Assign-Issues.yml | issues | 2026-08-04 success | issues: write | none | **keep** (works; issues-only token) |
| Jules-Auto-Rebase.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents: read / auto-rebase-prs: contents+pull-requests: write | none | retire |
| Jules-Auto-Refactor.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents: read / auto-refactor: contents+pull-requests: write | none | retire |
| Jules-Auto-Repair.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents+actions+checks: read / intelligent-repair: contents+pull-requests: write | none | retire (callee of Control-Tower; guarded via caller in #4923) |
| Jules-Cleaner.yml | pull_request | 2026-08-04 cancelled | none top-level / cleanup: pull-requests+contents: write | none | retire (writes on every PR, no owner) |
| Jules-Code-Quality-Fixer.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents: read / fix-issues: contents+pull-requests+issues: write | none | retire |
| Jules-Code-Quality-Reviewer.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents: read / review: contents+issues: write | none | retire |
| Jules-Completist.yml | workflow_call, workflow_dispatch, schedule | 2026-08-02 cancelled | contents: read / audit: contents+pull-requests+issues: write | none | retire (scheduled, cancelled) |
| Jules-Comprehensive-Assessment.yml | schedule, workflow_dispatch | 2026-08-02 cancelled | contents: read / comprehensive-assessment: contents+issues: write | none | retire (scheduled, cancelled) |
| Jules-Control-Tower.yml | push, pull_request, workflow_run, workflow_dispatch | 2026-05-22 cancelled | contents+actions+checks: read; `secrets: inherit` into 10 write-capable callees | none | **keep** only while any callee is kept; every job same-repo-guarded in #4923 |
| Jules-Critics-Comments.yml | workflow_call, workflow_dispatch, schedule | 2026-08-02 cancelled | contents+pull-requests: write (top-level) | none | retire |
| Jules-Diff-Verifier.yml | pull_request, workflow_dispatch | 2026-09-03 skipped | contents: read, pull-requests: write, issues: read | none | **keep** (runs on every PR today; read-only checkout) |
| Jules-Documentation-Auditor.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents: read / audit-documentation: contents+pull-requests+issues: write | none | retire |
| Jules-Documentation-Scribe.yml | workflow_call | 2026-05-30 failure | contents: read / audit-documentation: contents+pull-requests: write | none | retire |
| Jules-Hotfix-Creator.yml | workflow_call | 2026-05-30 failure | contents: read / create-hotfix: contents+pull-requests: write | none | retire |
| Jules-Issue-Mention-Handler.yml | issue_comment | 2026-08-04 success | contents+pull-requests+issues: write (top-level) | none | **keep** (in use), but narrow the top-level token to the job that pushes |
| Jules-Issue-Resolver.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents: read / resolve-issues: contents+pull-requests+issues: write | none | retire |
| Jules-Laymans-Terms-Writer.yml | workflow_call, workflow_dispatch, schedule | 2026-08-02 cancelled | contents+pull-requests: write (top-level) | none | retire |
| Jules-PR-AutoFix.yml | workflow_run (CI Standard), workflow_dispatch | 2026-08-04 cancelled | contents+pull-requests: write, actions+checks: read (top-level) | none | **keep** if any auto-fix is wanted; same-repo guard added in #4923; otherwise retire with Control-Tower |
| Jules-PR-Cleanup.yml | workflow_dispatch | 2026-05-30 failure | contents: read, pull-requests: write | none | retire |
| Jules-Sentinel.yml | workflow_call, workflow_dispatch | 2026-05-30 failure | contents: read / security-audit: contents+issues+security-events: write | none | retire (CodeQL re-enabled in #4923 covers the scan) |
| Jules-Supersede-Check.yml | push, workflow_dispatch | 2026-08-03 failure | contents: read, pull-requests: write | none | retire (fails on every push) |
| Jules-Tech-Custodian.yml | workflow_call | 2026-05-30 failure | none top-level / refactor-worst-file: contents+pull-requests: write | none | retire |
| Jules-Test-Generator.yml | workflow_call | 2026-05-30 failure | none top-level (inherits caller) | none | retire |
| jules-assessment-runner.yml | workflow_dispatch | 2026-05-30 failure | none top-level (inherits default) | none | retire |

## Summary

- 28 Jules workflows (27 `Jules-*` + `jules-assessment-runner.yml`); 0 have a
  documented owner.
- 2 have had a successful run since June (`Auto-Assign-Issues`,
  `Issue-Mention-Handler`); 1 runs on every PR and skips
  (`Diff-Verifier`); 24 last ran as failures/cancellations, 22 of them in the
  single 2026-05-30 dispatch burst or the 2026-08-02 scheduled burst.
- 20 hold `contents: write` at job or top level; 2 are reachable from
  `workflow_run` and are now same-repository-guarded (Tools #4923).
- Recommended keep set (5): `Auto-Assign-Issues`, `Issue-Mention-Handler`,
  `Diff-Verifier`, `PR-AutoFix`, `Control-Tower` — the last two only if the
  fleet wants automated CI repair at all; if not, retire both and the count
  drops to 3.
- Recommended retire set (23): everything else, in one governed fleet PR under
  Repository_Management#1483 (`docs/workflows/WORKFLOW_TRACKING.md` rows to be
  removed in the same PR).

## Decision (Repository_Management#1483)

Executed 2026-09-03 as one governed campaign PR under the Fleet Readiness
Program (Repository_Management#1505). **25 of the 28 retired, 3 kept.**

### Kept (3)

| Workflow | Why |
| --- | --- |
| `Jules-Auto-Assign-Issues.yml` | In use (success 2026-08-04) and holds an `issues: write` token only. |
| `Jules-Issue-Mention-Handler.yml` | In use (success 2026-08-04). Token narrowed here: top level is now `contents: read`, and `contents: read` + `issues: write` sits on `fix-issue` alone. It never pushes and never uses `GITHUB_TOKEN` on a PR, so the old top-level `contents`/`pull-requests`/`issues: write` was unneeded. |
| `Jules-Diff-Verifier.yml` | Runs on every PR with a read-only checkout. |

### Retired (25)

The inventory recommended retiring 23. `Jules-PR-AutoFix.yml` and
`Jules-Control-Tower.yml` were left conditional on "whether the fleet wants
automated CI repair at all". It does not: CI repair in this fleet is done by
the Claude and Codex remediation agents and the desktop Autofix. Neither
workflow has run at all since 2026-08-04 (Control-Tower not since 2026-05-22),
and Control-Tower is the *only* caller of ten of the retired reusable
workflows, so retiring it and its callees together keeps the call graph
consistent - leaving either side would have left a broken pair.

Explicit note on the 30-day rule: no retired workflow has had a successful run
inside the 30 days before 2026-09-03. The closest are `Jules-PR-AutoFix.yml`
and `Jules-Cleaner.yml`, whose last successes were 2026-08-04 and 2026-08-03 -
exactly at/just outside the boundary, and every run of either since then was
cancelled. Nothing retired here is a required status check: the only required
context in the Tools `main` ruleset is `quality-gate`
(`gh api repos/D-sorganization/Tools/rules/branches/main`). None of the 25 is
one of the three agent-governance workflows that WORKFLOW_GOVERNANCE section
2.3 mandates - Tools ships none of those three, which is a separate
pre-existing gap and is not touched here.

`tests/ops/test_workflow_run_security_guards.py` kept its `workflow_run`
same-repository-guard contract; the known-set assertion is now the empty set
and the two tests that pinned the retired files were removed.

### Restore

Any single workflow comes back with one command (base commit `90b7ff3c4639fd9e34864a97ffc6982706230c87`):

```bash
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Archivist.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Assessment-Generator.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Assessment-Remediator.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Auto-Rebase.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Auto-Refactor.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Auto-Repair.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Cleaner.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Code-Quality-Fixer.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Code-Quality-Reviewer.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Completist.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Comprehensive-Assessment.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Control-Tower.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Critics-Comments.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Documentation-Auditor.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Documentation-Scribe.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Hotfix-Creator.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Issue-Resolver.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Laymans-Terms-Writer.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-PR-AutoFix.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-PR-Cleanup.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Sentinel.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Supersede-Check.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Tech-Custodian.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/Jules-Test-Generator.yml
git checkout 90b7ff3c4639fd9e34864a97ffc6982706230c87 -- .github/workflows/jules-assessment-runner.yml
```
