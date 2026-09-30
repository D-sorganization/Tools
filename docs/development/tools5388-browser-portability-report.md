# Tools #5388 Browser Portability Report

## Scope and integration

Implemented the root-approved bounded plan in the owned `bot/luna-tools5388-20260930` worktree. The earlier `NOT dispatched` label in the plan was superseded by this dispatch. The plan is at
`C:/Users/diete/Repositories/docs/development/luna-backlog-20260929/tools5388-browser-root-plan.txt`.

Fetched `origin/main` at `3c62bbdf20cf0a1ffc878aa6b76f6865143ff3b5` and merged it into the branch. The resulting HEAD is `3d4d3825ef72ba626cd89d66d2649508c1476810`. This preserved the existing #5388 SPEC row and incorporated main's #5387 row and generated inventory changes. The already-approved #5388 browser test and SPEC entry are committed locally as `736f2365f`; no changes were pushed and no PR was created.

## Implementation

- The existing Escape regression fixture launches Playwright's bundled Chromium with no Chrome channel. The real browser assertions remain intact: both values clear, the Clear button hides, focus returns to From, default is prevented, delayed conversion does not refill cleared fields, later conversion works, Escape elsewhere preserves values, and modal Escape still closes the modal.
- `ci-standard.yml` provisions Chromium only for `matrix.shard == 'src-rest'`, after Python dependencies and before tests. It uses the installed Python Playwright version to install matching Chromium and Linux system dependencies, with the shared apt lock. `PLAYWRIGHT_BROWSERS_PATH` is set to `$RUNNER_TEMP/playwright-browsers` for both provisioning and later test steps. Missing root/passwordless sudo or a failed provisioning command fails the job.
- The regression path remains assigned to the existing `src-rest` shard. No lane, browser mock, skip, timeout change, runner change, or global browser cache was added.
- Added a workflow contract test covering shard assignment, the unchanged 3.11/3.12 matrix, conditional and ordered provisioning, the job-local browser path, apt locking, fail-closed provisioning, and bundled Chromium use.

## TDD and browser evidence

Workflow-contract RED, before adding provisioning:

```powershell
py -3.12 -m pytest -q tests/ops/test_ci_test_shards.py -k src_rest_provisions_job_local_bundled_chromium_after_dependencies
```

Result: failed at the missing provisioning-step lookup (`StopIteration`). After implementation, the targeted contract passed; the complete contract module now passes 17 tests.

Actual-browser baseline used Python 3.12.10, the declared-compatible Playwright 1.63.0, and Chromium installed into a temporary `PLAYWRIGHT_BROWSERS_PATH`. The test file was run against the pre-feature application files from base `0454a5aaa6ea18234fe84db1c0badc8284fa1fff`:

```powershell
& $pythonPath -m pytest -q -o addopts= -o timeout=60 --confcutdir $baseTests (Join-Path $baseTests 'test_keyboard_escape.py')
```

Result: **RED**, with both focused-field cases failing because Escape did not clear the inputs; the elsewhere/modal case passed (2 failed, 1 passed).

The same test against the merged PR worktree and the same isolated Chromium installation:

```powershell
& $pythonPath -m pytest -q -o addopts= -o timeout=60 --confcutdir 'src\web_applications\unit_converter\tests' 'src\web_applications\unit_converter\tests\test_keyboard_escape.py'
```

Result: **GREEN**, 3 passed. No system Chrome was used.

## Validation

- `py -3.12 -m pytest -q -n 0 tests/ops/test_ci_test_shards.py` — 17 passed (one existing `qt_api` config warning).
- `py -3.12 -m pytest -q -n 0 tests/test_python_version_contract.py` — 9 passed (one existing `qt_api` config warning).
- `py -3.12 scripts/ci_test_shards.py --check` — 1,769 test files partitioned across 7 shards.
- `py -3.12 scripts/validate_workflows.py` — 48 workflow files validated.
- Ruff check and format check on the two changed Python test files — passed.
- `py -3.12 shared_scripts/fleet_hooks.py spec-changelog` — passed; #5387 and #5388 dated rows are both present.
- `git diff --check` — passed.
- `py -3.12 -m scripts.build_tools_module_inventory --check` initially reported the Rust file-watcher shard stale. Root review established the exact canonical delta: the existing lexical mapper associates the new Escape test with a Rust `debounce` path based on test prose. The shard and descriptor were regenerated canonically and the existing inventory regression/check now pass. This mapped association is lexical only; it does not establish Rust execution or coverage.

## Remaining gates

- Root review of the local workflow and test diff.
- CI on Linux to exercise the actual `src-rest` provisioning permissions, apt dependencies, and both existing Python matrix versions. This Windows run does not establish Linux provisioning behavior.
- Root accepted the narrow canonical two-file inventory delta. The lexical mapped-test association does not establish Rust execution or coverage; no mapper semantics or test prose were changed.

No PR was created, no merge was attempted, and no CI run was manually rerun.
