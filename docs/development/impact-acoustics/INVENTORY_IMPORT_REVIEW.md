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

The normal push rejects two missing return annotations in the incoming contact
completion test helpers. Both now declare the existing `ImpactIntervalResult`
return contract; no production calculation changes. Prettier also formats the
incoming swing API baseline; parsed JSON equality is checked before acceptance.

Protected renderer main `b64a70f39` is integrated. The sole merge conflict is
the root handoff hash/line count, recomputed from the combined root document;
all other incoming metadata is preserved. All 410 reviewed candidate source
hashes remain identical. The annotation repair and value-identical API
formatting are published at `1c2c9b19d` through every normal push hook.

The first post-renderer regression times out while reading tracked source in
the deterministic inventory test. An isolated rerun, with the same code and
60-second per-test limit, passes all 95 inventory/import/merge, impact and API
tests (111.13 s). No expectation or timeout is relaxed.

## CI Freshness-Test Granularity

At published head `ace9a007b`, the Python 3.12 unit shard loses a worker during
`test_inventory_is_deterministic_and_fresh`. Its replacement passes the same
test. The single allowed flake retry repeats that pattern: job `102214787890`
reports a worker crash, then a replacement passes in 43.34 s. There is no
reported assertion mismatch; the logs do not establish an OOM diagnosis.
No second speculative rerun is requested. Other public tests and governance
checks pass; the private Gasification_Model consumer separately fails at
repository lookup before tests, with credential configuration unresolved.

A local Windows cProfile run of one unchanged generator check takes 135.662 s
under instrumentation. File opens account for 97.230 s cumulative; import
classification takes 11.623 s and all AST parsing 9.052 s. These overlapping
profile costs are not additive and are not CI timing measurements. Source
reads dominate this local observation; replacing a correct AST parser with
an unsafe heuristic is not justified by the profile.

The original test performs two full repository builds under one 60-second
deadline: the CLI projection check followed by independent equality with the
checked-in registry. These become two single-purpose tests, retaining each
original assertion verbatim. Both still discover, read and inspect the actual
tracked files; no mocked inventory, persistent source cache, reduced denominator,
schema bypass or timeout increase is introduced. This addresses test granularity
without changing producer behavior. All 96 combined import/inventory/schema,
merge-driver, contact-completion and public API tests pass (95.10 s). The CLI
check takes 22.52 s and independent reproducibility 26.08 s, each below the
unchanged 60-second limit. Scoped Ruff and all nine final manual gates pass.
Normal delivery is pending; only current-head CI can qualify the CI outcome.

## Main 21690dcfc Integration

Continue the task-owned PR #5103 in a new isolated integration checkout from
32c7b38cb. Preserve the original 81b28da05 checkout and the original JSON review.
All nine baseline manual gates pass. Main adds the protected #5109 theme
refactor: shared Catppuccin palette/style consumption in six modules, the help
menu's accurate optional-action annotation, and its six architecture controls.
The production merge has no source conflict. Handoff context/hash metadata and
three generated inventory files conflict; regenerate the inventory once from
the fully merged source, retaining the prior PR context pending final checks.
The optional in-merge regeneration driver is disabled only for this command so
these generated conflicts remain explicit; no validation hook is disabled.

Before the local typing repair, every production module equals current main.
The prior two impact test-helper return annotations remain as the sole embedded
source-tree test difference. Classifier producer/parser and both inventory test
files equal the PR head. The canonical comparison retains all 3,632 module paths,
1,281 calculation candidates and 2,351 non-calculation entries. Every original
410 candidate remains provisional and publication-blocked with the same owner.

Four original reviewed candidate hashes have legitimately evolved: measurement
and spectral files through already reviewed #5106, and asteroid renderer and
Function Generator through #5109. The latter retains its scientific NumPy import.
The complete new JSON delta records these changes without altering the original
review. No classification is downgraded or promoted to scientific approval.

Validation of the merged source: 15 new-theme/API controls pass in 7.63 s;
202 combined import/inventory/merge/contact/theme/API tests pass in 59.46 s with
11 existing theme deprecation warnings. CLI freshness and independent full
reproduction take 13.89 and 15.98 s under their unchanged 60-second deadlines.
Root Ruff 0.14.10 passes (3,728 Python files). The actual push mypy command
exposes 12 unused no-any-return suppressions in the incoming Function Generator
file. Removing only those comments leaves its executable AST identical and
makes all eight checked files pass; no numerical or visual behavior changes.
Regenerate/recheck the final inventory after this source-byte change.

The historical GUI stall #5114 is still not explained. Its original source
passes 2 isolated and 21 mixed PyQt tests, and actual Python 3.11 CI on 32c7b38cb
passes the rate shard; Python 3.12 was still in its test step at the last read.
See the linked issue investigation. Private checkout failure remains separate.
All nine final structural checks pass after refreshing the root handoff hash.
The canonical final handoff manifest records this integration checkout and
PR #5103; its dirty pre-commit base is explicit. Normal publication and
current-head CI remain required.
