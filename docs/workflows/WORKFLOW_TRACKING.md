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
| Jules-Auto-Assign-Issues.yml       | Active (issues; `issues: write` only) | Assigns newly opened issues; kept by Repository_Management#1483 (last success 2026-08-04). |
| Jules-Issue-Mention-Handler.yml    | Active (issue_comment `@jules`; token narrowed to `fix-issue`) | Hands an @jules-mentioned issue to the Jules API; kept by Repository_Management#1483. |
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
- 2026-09-03: `codeql-analysis.yml.disabled` re-enabled as `codeql-analysis.yml` (python, javascript-typescript; PR + weekly). Every job of the two `workflow_run` consumers (`Jules-PR-AutoFix.yml`, `Jules-Control-Tower.yml`) now carries `github.event.workflow_run.head_repository.full_name == github.repository`. The undated `--ignore-vuln CVE-2026-4539` was removed from `ci-standard.yml` pip-audit (fixed in pygments 2.20.0; requirements pins >=2.21.0). Jules-* workflows are inventoried in `jules_inventory.md` for the Repository_Management#1483 retirement campaign; none deleted here (Tools #4923, Repository_Management#1507).
- 2026-09-03: **Jules suite retired** under Repository_Management#1483 (program #1505). 25 workflow files deleted; the 3 kept are `Jules-Auto-Assign-Issues.yml`, `Jules-Diff-Verifier.yml` and `Jules-Issue-Mention-Handler.yml` (whose top-level `contents`/`pull-requests`/`issues: write` token is narrowed to `contents: read` top-level plus `contents: read, issues: write` on the `fix-issue` job). 5 stale rows for files that no longer exist (`Jules-Assessment-AutoFix.yml`, `Jules-Comment-Processor.yml`, `Jules-Consolidator.yml`, `Jules-DRY-Orthogonality.yml`, `Jules-PR-Compiler.yml`) were removed in the same pass. Per-workflow evidence and the restore commands are in `jules_inventory.md`.
