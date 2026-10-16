# F-TL-023

**Design-manual governance: 3.4k LOC checkers + 9.7 MB content-hash inventory + merge driver govern one calculation**

| Severity | Category   | Confidence | Effort | Status   |
| -------- | ---------- | ---------- | ------ | -------- |
| 🟧 High  | governance | high       | L      | proposed |

**Locations:** `scripts/check_design_manual_governance.py:1`, `scripts/build_tools_module_inventory.py:1`, `scripts/git/regenerate_module_inventory_during_merge.py:1`, `.pre-commit-config.yaml:86`, `manuals/tools/calculation-registry.json:6`, `manuals/tools/publication-projection.json:11`

## Finding

- Severity: High - Category: governance
- Confidence: high
- Location: scripts/check_design_manual_governance.py (779 LOC); scripts/build_tools_module_inventory.py (447); scripts/tools_exemplar_contract.py (441); scripts/tools_module_inventory_contract.py (371); scripts/tools_textbook_chapter_contract.py (369); scripts/render_tools_design_manual.py (274); manuals/tools/manifests/module-inventory/ (156 shards, 9.6 MB, 3,892 entries); scripts/git/regenerate_module_inventory_during_merge.py; .pre-commit-config.yaml:86-120; manuals/tools/calculation-registry.json:6; manuals/tools/publication-projection.json:11; manuals/tools/dist/
- Evidence: `check_design_manual_governance` → "8 QMD sources, 1 registered calculations, 1 registered textbook chapters, release=provisional". Inventory entry for `config/coverage_baseline.json`: 40 lines of JSON, `"classification": "non-calculation"`, eleven `"not-applicable"` states, ten empty traceability arrays. `"inventory_commit": "916fb9f…"`, `"immutable_source_commit": "920e6fd…"` while HEAD is `b630cd3` — every `--check` still passes.
- Failure scenario: every PR touching any tracked file must regenerate a 9.6 MB JSON (hence the merge driver and the `pre-merge-commit` hook with its 20-line comment on why `pre-commit` does not fire for merges). That is pure merge-conflict surface with no reader: nothing consumes `entries-config.json`. The one governed calculation (D-plane geometry) is a 300-line chapter. The checkers verify structure and explicitly disclaim meaning ("passing the linter is structural evidence, not calculation approval"). The repo paid ~15k LOC (scripts + 12.9k LOC of tests/architecture and tests/ops meta-tests) for a manual engineers do not use, while F-TL-001..004 shipped.
- Why a newer engineer misses this: each piece answered a real incident ("inventory stale", "merge conflicts on inventory"); the sum is a second codebase whose only test is itself.
- Proposed change: (1) treat the inventory scope as a _governance-policy change_: first amend the governing contract in `config/design_manual_governance.json` (which today requires every tracked implementation module to be classified, so a narrower manifest would fail the gate), scoping the strict inventory to calculation kernels (~200 entries, <500 KB), then regenerate; (2) move `manuals/tools/dist/*` to release artifacts; (3) collapse the nine `--check` scripts into one `python -m scripts.check_design_manual`; (4) make embedded commit hashes fail on drift or remove them.
- Verification: `git ls-files manuals | wc -l` drops from 198 to <40; one command in CLAUDE.md instead of nine.
- Effort: L

## Draft Pull Request

**Title:** `chore(manuals): shrink design-manual governance to what a reader uses`

### Why This Matters (The Pitch)

The repo carries about 3.4k lines of checker scripts, a 9.6 MB content-hash inventory of every tracked file, a custom git merge driver and a pre-merge-commit hook, and together they govern one registered calculation. The inventory has 3,892 entries and nothing consumes it, yet every PR touching any tracked file must regenerate it, which is why the merge driver exists. Its embedded commit hashes do not match HEAD and every check still passes. The checkers disclaim meaning in their own words. Meanwhile the numeric defects in F-TL-001 through 004 shipped. The change inventories only calculation kernels, moves built artifacts out of git, collapses nine check scripts into one command, and makes commit hashes fail on drift. Less to clone, lint, merge and read; fewer conflicts for every contributor.

### Proposed PR Body

## Problem

`scripts/check_design_manual_governance.py` (779 LOC), `scripts/build_tools_module_inventory.py` (447), `scripts/tools_exemplar_contract.py` (441), `scripts/tools_module_inventory_contract.py` (371), `scripts/tools_textbook_chapter_contract.py` (369), `manuals/tools/manifests/module-inventory/` (156 shards, 9.6 MB, 3,892 entries) and `.pre-commit-config.yaml:86-120` govern 1 registered calculation (`manuals/tools/calculation-registry.json:6`). Embedded commits `916fb9f` and `920e6fd` differ from HEAD `b630cd3`.

## Change

- [ ] Inventory only calculation kernels (about 200 entries, under 500 KB)
- [ ] Move `manuals/tools/dist/*` to release artifacts
- [ ] Collapse the nine `--check` scripts into one `python -m scripts.check_design_manual`
- [ ] Make embedded commit hashes fail on drift, or remove them
- [ ] Remove the merge driver and pre-merge-commit hook once the inventory is small

## Verification

`git ls-files manuals | wc -l` drops from 198 to under 40; CLAUDE.md lists one command instead of nine.

## Risk and Rollback

Large change touching governance tooling; stage it in the four steps above so each can be reverted alone. Dropping the inventory loses per-file hash history, which has no listed consumer.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-TL-038 covers the broader script and meta-test volume.

Refs: adversarial review 2026-10-16, F-TL-023

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
