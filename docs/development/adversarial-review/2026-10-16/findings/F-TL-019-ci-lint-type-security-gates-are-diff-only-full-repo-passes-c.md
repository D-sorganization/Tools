# F-TL-019

**CI lint/type/security gates are diff-only; full-repo passes continue-on-error; 43 stale ruff per-file-ignore paths**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | S      | proposed |

**Locations:** `.github/workflows/ci-standard.yml:311`, `.github/workflows/ci-standard.yml:363`, `.github/workflows/ci-standard.yml:379`, `ruff.toml:176`, `ruff.toml:207`, `ruff.toml:296`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: .github/workflows/ci-standard.yml:300-316, :337-360, :362-392, :416-433; ruff.toml:176, :207, :296, :303 (43 missing paths in scratch `ruff_missing.txt`); src/folder_tool/ruff.toml
- Evidence: ci-standard.yml:365-366 `# Uses continue-on-error: true so existing accumulated violations do not block PRs`. ruff.toml:176 `"src/shared/python/upstream_drift_tools/bootstrap.py" = ["E501"]` — the package is a 29-line shim; the path does not exist.
- Failure scenario: CLAUDE.md "`ruff check` — zero violations" holds only for pinned ruff 0.14.10 and the root config; the nested `src/folder_tool/ruff.toml` selects rules (RUF, TRY, SIM, N, EXE) that fail 47 times on ruff 0.16.8, and 72 files would be reformatted — the next ruff bump in pre-commit reds the tree while CI (diff-only, pinned) stays green. 43 of ~200 per-file-ignores are dead weight nobody will know is dead.
- Why a newer engineer misses this: a per-file-ignore for a missing file is silently a no-op; ruff does not warn.
- Proposed change: a 10-line test that every literal path in `ruff.toml` per-file-ignores exists; delete the 43 entries; make the full-repo ruff check blocking (it passes today, so the cost is zero).
- Verification: the test.
- Effort: S

## Draft Pull Request

**Title:** `ci(lint): block on full-repo ruff, drop 43 dead per-file-ignores`

### Why This Matters (The Pitch)

The zero-violations lint gate in CLAUDE.md holds only for the pinned ruff and the root config. CI runs ruff, format and mypy on changed files only; the full-repo passes are continue-on-error. A nested src/folder_tool/ruff.toml selects rules that fail 47 times under ruff 0.16.8, and 72 files would be reformatted, so the next ruff bump in pre-commit turns the tree red while CI stays green. Meanwhile 43 of about 200 per-file-ignores in ruff.toml point at paths that no longer exist, silent no-ops nobody can tell from live ones. The change adds a ten-line test that every literal ignore path exists, deletes the 43 entries, and makes the full-repo ruff check blocking, which passes today so the cost is zero. Maintainers get a lint gate that means what it says.

### Proposed PR Body

## Problem

`.github/workflows/ci-standard.yml:300-316, :337-360, :362-392, :416-433` run lint on changed files; the full-repo steps carry `continue-on-error: true` (:365-366). `ruff.toml:176, :207, :296, :303` list 43 per-file-ignore paths that do not exist, e.g. `src/shared/python/upstream_drift_tools/bootstrap.py`. `src/folder_tool/ruff.toml` yields 47 errors on ruff 0.16.8.

## Change

- [ ] Add a test that every literal path in `ruff.toml` per-file-ignores exists
- [ ] Delete the 43 stale entries
- [ ] Make the full-repo ruff check blocking (it passes today on the pinned version)

## Verification

The new per-file-ignore existence test passes, and the full-repo ruff step is green without `continue-on-error`.

## Risk and Rollback

A future ruff bump will surface violations immediately, which is the intent. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: nested `src/folder_tool/ruff.toml` violations under ruff 0.16.8 (47 errors, 72 format diffs).

Refs: adversarial review 2026-10-16, F-TL-019

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
