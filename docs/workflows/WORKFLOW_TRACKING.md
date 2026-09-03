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
| ci-standard.yml                    | Defined (required: `quality-gate`, `tests (3.11)`; RM #1507) | Lint/type/security gate + whole-tree sharded pytest (`scripts/ci_test_shards.py`, Tools #4913). |
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

- 2026-09-02: `ci-standard.yml` concurrency group changed to `ci-standard-${{ github.ref == 'refs/heads/main' && github.sha || github.ref }}` (per-commit on main; `cancel-in-progress: true` unchanged) so a later push never cancels an in-flight main run (main-green campaign, Repository_Management#1507).
- 2026-09-03: `ci-standard.yml` gains one stdlib-only job `divergence-ledger-gate` (pull_request only, `ubuntu-24.04`, `fetch-depth: 0`) running `python scripts/check_divergence_ledger.py` — a PR touching a ledgered shadowed module must name its paired UpstreamDrift PR (`UD-PAIR:`) unless the ledger rules the module tools-canonical (Tools #4915; governed edit, Repository_Management#1507).
- 2026-09-03: `cross-repo-python-integration.yml` fails when a downstream lacks `tests/shared_contracts/` (was a green skip) and no longer tolerates a failed downstream checkout; `release.yml` `validate` runs `scripts/check_wheel_build.py --check`, `github-release` attaches the `ud_tools` wheel + CycloneDX SBOM, and the new `wheel-artifact` job uploads `tools-wheel-<sha>` on every push to main (Tools #4920, Repository_Management#1507).
- 2026-09-03: `ci-standard.yml` `tests` job is now a `python-version × shard` matrix over the partition in `scripts/ci_test_shards.py` (whole tree, incl. `src/pendulum_simulator` and `src/movement_optimizer`); new `tests-gate` job owns the required `tests (3.11)` context, verifies every shard passed and applies the single coverage floor from pyproject to the combined data. The `core_tests` allowlist, changed-file selection, branch-name conditionals and the Provider-Contract step (its suites are in the `tests-shared` shard) are gone (Tools #4913, Repository_Management#1507).
