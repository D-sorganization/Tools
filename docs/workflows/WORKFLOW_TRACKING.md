# Workflow Tracking Document

This document lists GitHub Actions workflows currently present in this repository.

Global status note: Paused for catching up.

| Workflow File                      | Status                           | Purpose                                           |
| ---------------------------------- | -------------------------------- | ------------------------------------------------- |
| agent-metrics-dashboard.yml        | Defined (see global status note) | See workflow YAML header/name for execution role. |
| auto-issue-resolver.yml            | Defined (see global status note) | See workflow YAML header/name for execution role. |
| auto-update-prs.yml                | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Bot-CI-Trigger.yml                 | Defined (see global status note) | See workflow YAML header/name for execution role. |
| check-tools-manifest.yml           | Defined (see global status note) | See workflow YAML header/name for execution role. |
| ci-failure-digest.yml              | Defined (see global status note) | See workflow YAML header/name for execution role. |
| ci-standard.yml                    | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Code-Metrics.yml                   | Defined (see global status note) | See workflow YAML header/name for execution role. |
| codeql-analysis.yml                | Active (PR to main, push to main, weekly; RM #1507) | CodeQL security-and-quality for python + javascript-typescript on d-sorg-fleet (Tools #4923). |
| Comment-to-Issue-Converter.yml     | Defined (see global status note) | See workflow YAML header/name for execution role. |
| docs-governance.yml                | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Archivist.yml                | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Assessment-AutoFix.yml       | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Assessment-Generator.yml     | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Assessment-Remediator.yml    | Defined (see global status note) | See workflow YAML header/name for execution role. |
| jules-assessment-runner.yml        | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Auto-Assign-Issues.yml       | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Auto-Rebase.yml              | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Auto-Refactor.yml            | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Auto-Repair.yml              | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Cleaner.yml                  | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Code-Quality-Fixer.yml       | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Code-Quality-Reviewer.yml    | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Comment-Processor.yml        | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Completist.yml               | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Comprehensive-Assessment.yml | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Consolidator.yml             | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Control-Tower.yml            | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Critics-Comments.yml         | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Documentation-Auditor.yml    | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Documentation-Scribe.yml     | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-DRY-Orthogonality.yml        | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Hotfix-Creator.yml           | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Issue-Mention-Handler.yml    | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Issue-Resolver.yml           | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Laymans-Terms-Writer.yml     | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-PR-AutoFix.yml               | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-PR-Cleanup.yml               | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-PR-Compiler.yml              | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Sentinel.yml                 | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Supersede-Check.yml          | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Tech-Custodian.yml           | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Jules-Test-Generator.yml           | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Maintenance-Global-Control.yml     | Defined (see global status note) | See workflow YAML header/name for execution role. |
| Nightly-Doc-Organizer.yml          | Defined (see global status note) | See workflow YAML header/name for execution role. |
| release.yml                        | Defined (bump gated on feat/fix/perf or dispatch; RM #1507) | Version bump PR + GitHub release; changelog delta via scripts/release_changelog.py. |
| pr-auto-labeler.yml                | Defined (see global status note) | See workflow YAML header/name for execution role. |
| PR-Comment-Responder.yml           | Defined (see global status note) | See workflow YAML header/name for execution role. |
| stale-cleanup.yml                  | Defined (see global status note) | See workflow YAML header/name for execution role. |
| tauri-build.yml                    | Defined (see global status note) | See workflow YAML header/name for execution role. |
| topology-governance.yml            | Defined (see global status note) | See workflow YAML header/name for execution role. |

## Maintenance

Update this file whenever workflows are added, removed, enabled, or disabled.
For governance, see Repository_Management/docs/architecture/WORKFLOW_GOVERNANCE.md.

- 2026-09-02: `ci-standard.yml` concurrency group changed to `ci-standard-${{ github.ref == 'refs/heads/main' && github.sha || github.ref }}` (per-commit on main; `cancel-in-progress: true` unchanged) so a later push never cancels an in-flight main run (main-green campaign, Repository_Management#1507).
- 2026-09-03: `codeql-analysis.yml.disabled` re-enabled as `codeql-analysis.yml` (python, javascript-typescript; PR + weekly). Every job of the two `workflow_run` consumers (`Jules-PR-AutoFix.yml`, `Jules-Control-Tower.yml`) now carries `github.event.workflow_run.head_repository.full_name == github.repository`. The undated `--ignore-vuln CVE-2026-4539` was removed from `ci-standard.yml` pip-audit (fixed in pygments 2.20.0; requirements pins >=2.21.0). Jules-* workflows are inventoried in `jules_inventory.md` for the Repository_Management#1483 retirement campaign; none deleted here (Tools #4923, Repository_Management#1507).
