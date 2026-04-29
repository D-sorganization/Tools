# TODO/FIXME Count Reconciliation

Issue: https://github.com/D-sorganization/Tools/issues/2360

## Authoritative Count

The authoritative count on `origin/main` at commit `29b4eb77` is **49**.

This count uses the scope requested in issue #2360:

```bash
git grep -nE 'TODO|FIXME|XXX|HACK|KLUDGE' -- 'src/**' 'scripts/**' '*.py' '*.cpp' '*.h' '*.ts' '*.js' | grep -v 'test_' | wc -l
```

PowerShell-safe equivalent used for this reconciliation:

```powershell
$matches = git grep -nE 'TODO|FIXME|XXX|HACK|KLUDGE' -- 'src/**' 'scripts/**' '*.py' '*.cpp' '*.h' '*.ts' '*.js'
($matches | Select-String -NotMatch 'test_').Count
```

The prior reported values, 3,775 and 5,419, are not reproducible with the issue's canonical scope on current `origin/main`.

## Enforcement Groundwork

`scripts/check_todo_issue_links.py` now provides two narrow controls:

- `--count-only` prints the canonical reconciled count.
- `--check-staged` rejects newly staged TODO/FIXME/XXX/HACK/KLUDGE markers in the canonical code/script scope unless the added line links to a GitHub issue.

The pre-commit configuration runs the staged check on every commit. This does not categorize or rewrite legacy markers; it prevents the reconciled debt surface from growing without traceability.

## Duplicate Status

Issue #2364 was already closed as a duplicate of #2360 on 2026-04-26 with a maintainer comment. No additional duplicate action was needed.
