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
| codeql-analysis.yml.disabled       | Disabled                         | See workflow YAML header/name for execution role. |
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
