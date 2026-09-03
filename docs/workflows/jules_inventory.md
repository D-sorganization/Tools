# Jules-\* workflow inventory (Tools)

Snapshot taken 2026-09-03 for Tools #4923 (Fleet Readiness Program,
Repository_Management#1505 Phase 1) as input to the fleet retirement campaign
Repository_Management#1483. **Nothing is deleted by this document**; deletions
are a fleet decision made in #1483. Acceptance for #4923 is "Jules workflow
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
