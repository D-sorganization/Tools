# Scientific Import Inventory Review

Tools #5101 repairs a capability-inventory defect discovered during impact
program #5068 / shaft issue #5072. The previous regular expression omitted
whitespace after Python `import` and `from`, missing ordinary scientific
imports while matching text such as `importnumpy` in comments and strings.
The private shaft spectrum on the separate T3 branch exposed this defect;
this correction starts from protected main `d9dec3602605f5c03eb7dff507a02ef7c2995b1b`.

## Detection Contract

The existing generator keeps path-marker precedence and its non-Python rules.
For Python files it now delegates to a small AST inspector. Absolute imports
of the existing scientific root names are detected, including aliases,
submodules, multiple names, whitespace and nested/conditional imports. Source
is parsed but never imported or executed. Python module names remain case
sensitive; filename suffix recognition follows the inventory's existing
case-insensitive language detection. Comments, strings, unrelated names and
relative imports do not supply an absolute scientific-library signal.

Unparseable Python without a decisive path marker raises an explicit error
instead of silently claiming non-calculation. All 2,342 pre-change governed
Python modules parse in the Python 3.12 validation runtime. Dynamic imports,
arbitrary arithmetic and unnamed local numerical libraries still require
other signals or review. The inventory remains conservative discovery, not a
complete scientific assessment of everything a module can calculate.

## Complete Generated Delta

`INVENTORY_IMPORT_REVIEW.json` records every changed existing entry, the
matching import statements and line numbers, and source hashes for every
reclassified module. The inventory grows from 3,626 to 3,627 modules because
the new parser helper is itself governed. There are no removed modules.

Calculation candidates increase from 866 to 1,276: 410 previously missed
Python modules become provisional, review-required candidates. Non-calculation
entries change from 2,760 to 2,351. Every one of those 410 entries has a real
absolute scientific import. Only classification-derived metadata changes in
them; existing source hashes, owners, identifiers and traceability are retained.
All remain publication-blocked. No existing calculation is downgraded.

The other existing-entry changes are the generator's own source hash/length
and its new test links. The helper adds one non-calculation tooling entry.
No source formula, runtime scientific model, equation approval, registered
textbook pathway or existing publication approval is changed.

This broad delta is intentional conservative review routing. For example,
`camera_preferences.py` imports `math` solely to validate finite zoom values;
the old test's non-calculation expectation depended on the broken detector.
It now correctly expects a provisional candidate, while pure configuration
provides the non-calculation control. The label does not establish that camera
preferences contains a substantive scientific model or needs a new textbook.

## TDD and Verification

- Initial focused run: 14 failures and seven passes, reproducing missed imports,
  false positives and silent parse-failure classification.
- First parser implementation: all 21 tests pass (0.82 s).
- A suffix-case extension fails before its correction; `.PY` now follows the
  same language rule as `.py`.
- The first broader run passes 50 tests and catches the two stale camera
  expectations. The source's actual `math.isfinite` call is reviewed before
  updating those expectations.
- Final import, inventory/schema/freshness and merge-driver suite: 53 passes
  (48.79 s). Negative source snippets demonstrate that inventory never executes
  source code; schema and publication-blocker contracts remain enforced.
- All nine manual gates pass before and after the change. Repository-wide
  Ruff 0.14.10 passes (3,718 Python files); scoped mypy passes for both producer
  modules. Normal delivery is pending in this checkpoint.

Preserve this issue's scope when integrating main. Do not hand-edit generated
classification rows or copy the unmerged T3 implementation into this branch.
After the classifier merges, T3 must regenerate its inventory to replace the
documented false-negative spectrum label. The impact, acoustic, physical-data
and blinded-validation requirements remain separate and open.

Published as `f2ef920be9c4c5e34542f04ffbc4af459e31e66a` through all normal
commit/push hooks, including unit tests, Bandit, dependency audit and fleet
guardrails. PR #5103 is open. Prettier reformats the two JSON review/handoff
records; parsed before/after values are verified identical before committing.
Protected main `183b4bb1f` then adds impact-interval contact completion (#5088)
and merges cleanly. Its source and API changes are retained. All 95 combined
impact-interval, public API, import/inventory and merge-driver contracts pass
(87.90 s). All nine final manual gates pass. All 410 reclassified source hashes
and provisional/publication-blocked states remain identical to the complete
review record after this main merge. Current-head protected CI remains required.
