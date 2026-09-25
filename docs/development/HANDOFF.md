# Current handoff — Knowledge-pack engine (Tools#5345)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-5345`
- Branch: `feat/5345-knowledge-pack`; commit SELF; PR: see DL-#5345
- Issue: #5345 (K0 of Repository_Management#1772: Vision Quest, Disciple, Sidekick Wizards)
- Built: `src/shared/python/ai/knowledge/` (manifest, chunking, sources, pack, cli). It uses only the stdlib and PyYAML and imports no other Tools module, so Runner_Dashboard vendors it (RD#1479). The pack format is gated by `PRAGMA user_version = 1`.
- Contract: `build_pack(manifest, roots, out) -> PackInfo`; `KnowledgePack.open(p).search(q, k=8, include_superseded=False) -> list[Passage]`, `.info()`, `.is_stale(roots)`. Status precedence: manifest override > front-matter `status:` > current. Ties break by authority (published > findings > reviews > product > reference > notes).
- Baseline: `knowledge` is added to `VENDORED_PACKAGES`; new file `tests/api_baselines/knowledge_api_baseline.json`. Regeneration also rewrote the theme baseline, which was reverted by hand.
- Validation: `py -3.12 -m pytest tests/shared/python/ai/knowledge tests/test_shared_package_api_stability.py -o addopts=""` -> 42 passed; ruff and mypy clean; smoke build of RM `staff/knowledge/findings.yml` over local UD + AffineDrift -> 10,288 passages in 2.8 s.
- Next: K3a Tools#5346 (per-product Wizard packs for Sidekick); K4 Tools#5347 (refresh job).

---

# Past handoff — Retire the review-comment-to-issue converter (RM#1755)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-retire-converter`
- Branch: `chore/retire-comment-converter`
- Issue: Repository_Management#1755
- What was removed: `.github/workflows/Comment-to-Issue-Converter.yml` (already disabled 2026-09-25; no processor script, tests, or lingering references were present).
- Validation: `py -3.12 <RM>/scripts/campaigns/review_comment_converter_retirement/retire_converter.py --repo . --check` -> exit 0 after `--apply`.
- Next step: open the draft PR for review.

---

# Night Watch Development Log Maintenance — 2026-09-23

## Identity

- Repository: D-sorganization/Tools
- Working directory: `/home/dieterolson/staff-worktrees/Tools-run-350067a6bf64`
- Branch: `staff/night-watch-task-239d87`
- Baseline commit: `96d5681328ac62e53d94371f07e09c95f0a3b2cd`
- Implementation commit: SELF
- Pull request: SELF
- Governing issue/epic: Night Watch scheduled pass (no governing issue)

## Objective and status

Docs compliance sweep: reconciled the development log state table with actual
merged PR and closed-issue records on GitHub. Found 29+ entries still marked
`in_review` or `in_progress` whose governing PRs had already merged (some as
recently as 2026-09-23, others as old as 2026-09-08). Marked 30 entries
`shipped` and 1 entry `parked`. The log is now current.

## Files and decisions

- `docs/development/DEVELOPMENT_LOG.md`: 31 entries reconciled. 30 marked
  `shipped` (each with the correct PR reference); 1 (DL-#5132, calibration
  numerical recovery) marked `parked` because PR #5136 was closed without
  merge and the issue is closed. Last-audited timestamp updated to
  2026-09-23 by night-watch.
- `docs/development/HANDOFF.md`: this file, replacing the stale codex session
  HANDOFF from #5322.

## Validation

- No source code changed; only `docs/development/` files touched.
- All formerly-active entries now have accurate `shipped` or `parked` states.
- Zero `in_review` / `in_progress` entries remain.

## Blockers and risks

- DL-#5132 (calibration numerical recovery, PR #5136): PR closed without
  merge and issue closed. Marked `parked`. Retry would require a new issue
  and a new entry.
- DL-#0001 (backup-pyo3-split): pre-existing parked orphan with no governing
  issue; not changed in this pass.

## Next steps

- Open the draft PR, let CI pass, and merge.
- A future Night Watch pass should audit the older DL-00NN entries for
  candidates to move to `abandoned`.

## Change log

- `SELF` — Night Watch 2026-09-23: reconcile 31 stale development log entries
  to `shipped` or `parked` based on verified GitHub merge records.
