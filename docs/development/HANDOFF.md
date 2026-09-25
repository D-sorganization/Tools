# Current handoff — Knowledge-pack golden Q&A evaluation + optional MiniLM hybrid ranking (Tools#5347)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/agy-5347`
- Branch: `agy/issue-5347`; commit SELF; PR: #5351 (draft)
- Issue: #5347 (K4 of Repository_Management#1772: Q&A evaluation and optional MiniLM hybrid ranking)
- Built: `src/shared/python/ai/knowledge/eval.py` (GoldenQACase, EvalSummary, evaluate_pack, load_golden_set, CLI) and optional dense MiniLM embeddings with hybrid BM25 + cosine Reciprocal Rank Fusion ranking in `src/shared/python/ai/knowledge/pack.py`. Off by default and gated by manifest's `embeddings: true`. Core remains stdlib + PyYAML only.
- Rebase (2026-09-25): rebased the single feature commit onto `main` after #5350 (Sidekick Wizards) merged; hand-edited `tests/api_baselines/knowledge_api_baseline.json` to add only this PR's new symbols; regenerated the module inventory. Fixed `get_minilm_embedder` to log a warning and return `None` instead of silently swallowing the import failure, and made `_search_hybrid` fall back to plain BM25 with a one-time warning when no embedder is available. Also fixed two pre-existing mypy errors surfaced by full-context checking: `getattr(embedder, "embed", embedder)` was untyped (added `_as_embed_fn` helper), and `GoldenQACase.from_dict` narrowed `tuple[str, ...]` fields to `tuple[str]` on first assignment (added explicit annotations).
- Validation: `py -3.12 -m pytest tests/shared/python/ai/knowledge src/shared/python/ai/tests -q` -> 149 passed, 1 skipped (sentence-transformers probe unavailable in this environment); `tests/test_shared_package_api_stability.py` -> 10 passed; `ruff check` and `ruff format --check` clean on changed files; `MYPYPATH=src:src/python/src py -3.12 -m mypy --ignore-missing-imports --follow-imports=silent` clean on the changed package (verified with `sentence_transformers`/`transformers` follow-imports skipped locally to route around an unrelated mypy 1.13 INTERNAL ERROR crash inside this box's globally-installed `transformers` package — not a repo dependency, so CI's clean venv should not hit it); module inventory `--check` passes.
- Note: a concurrent Antigravity session was live in this same worktree while this rebase was done and left an uncommitted, untracked `src/shared/python/ai/knowledge/embeddings.py` (an in-progress split of the embedding helpers out of `pack.py`) plus at least one direct edit to `pack.py` and a revert of `tests/api_baselines/knowledge_api_baseline.json` that this session detected and undid. Neither is part of this PR's commits. Whoever picks this up next should check with that session before touching `embeddings.py` or deleting it.
- Next: address review feedback; do not mark ready or arm auto-merge without owner sign-off.

---

# Past handoff — Sidekick Wizards (Tools#5346)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-5346`
- Branch: `feat/5346-sidekick-wizards`, stacked on `feat/5345-knowledge-pack` (PR #5348); commit SELF; PR not created until #5348 merges.
- Issue: #5346 (K3a of Repository_Management#1772)
- Built:
  - `ai/knowledge/wizard.py`: `WizardConfig` + `load_wizard_config`, `KnowledgeContext.render()`, `WizardKnowledge` with TTL-cached freshness. It needs only the stdlib and PyYAML.
  - `ai/wizards.py`: the Sidekick glue. It holds a per-root cache, calls `register_app_context` and provides `knowledge_for_context`.
  - `BaseAgentAdapter.build_context_instruction_section` appends Wizard knowledge. Every adapter already uses this path.
  - `search_knowledge_base` prefers the pack.
  - `RAGContextProvider` emits a DeprecationWarning.
- Host contract: `knowledge/wizard.yml` (key, name, description, capabilities, manifest=`knowledge/pack.yml`, pack=`.knowledge/pack.sqlite`, k, roots). Build with `python -m shared.python.ai.knowledge build knowledge/pack.yml --root <Repo>=. --out .knowledge/pack.sqlite`.
- Validation: `py -3.12 -m pytest tests/shared/python/ai src/shared/python/ai/tests tests/test_shared_package_api_stability.py -o addopts="" -n 8` -> 570 passed; ruff and mypy clean. The knowledge API baseline gains wizard (the theme baseline rewrite was reverted).
- Next: open the PR after #5348 merges; hosts UD#10943 and GM#5089 (tier:cli); refresh job Tools#5347.

---

# Past handoff — Knowledge-pack engine (Tools#5345)

- Repository: D-sorganization/Tools
- Worktree: `Tools-worktrees/claude-5345`
- Branch: `feat/5345-knowledge-pack`; commit SELF; PR: see DL-#5345
- Issue: #5345 (K0 of Repository_Management#1772: Vision Quest, Disciple, Sidekick Wizards)
- Built: `src/shared/python/ai/knowledge/` (manifest, chunking, sources, pack, cli). It uses only the stdlib and PyYAML and imports no other Tools module, so Runner_Dashboard vendors it (RD#1479). The pack format is gated by `PRAGMA user_version = 1`.
- Contract: `build_pack(manifest, roots, out) -> PackInfo`; `KnowledgePack.open(p).search(q, k=8, include_superseded=False) -> list[Passage]`, `.info()`, `.is_stale(roots)`. Status precedence: manifest override > front-matter `status:` > current. Ties break by authority (published > findings > reviews > product > reference > notes).
- Baseline: `knowledge` is added to `VENDORED_PACKAGES`; new file `tests/api_baselines/knowledge_api_baseline.json`. Regeneration also rewrote the theme baseline, which was reverted by hand.
- Validation: `py -3.12 -m pytest tests/shared/python/ai/knowledge tests/test_shared_package_api_stability.py -o addopts=""` -> 42 passed; ruff and mypy clean; smoke build of RM `staff/knowledge/findings.yml` over local UD + AffineDrift -> 10,288 passages in 2.8 s.
- CI follow-up: the module inventory was regenerated with `py -3.12 -m scripts.build_tools_module_inventory`, then LF-normalized because the Windows write is CRLF. The divergence-ledger gate needs `UD-PAIR:` in the PR body (paired with UD#10943; `ai/knowledge` has no UD copy).
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
