# AGENT_HANDOFF — Tools

> **Update this file in every implementation commit and every push to `main`.**
> Current-state only; history lives in git. Last updated: 2026-08-14.

## Where This Repo Is Headed

Shared engineering monorepo consumed by UpstreamDrift and Gasification_Model, plus
several first-party apps (rate_of_closure, pendulum_simulator, rotation_converter,
movement_optimizer, p1am_control_system). Public API changes here are breaking
changes downstream.

## Active Epics (one line each)

- **#4103 swing→impact→flight platform** — consolidated and landing; PyQt6 and React
  clients over one shared physics core.
- **#4104 P7 wasm-pack core** — **not delivered.** The React app still runs the
  TypeScript physics twins; reference it with `Part of`, never `Closes`.
- **#4142 / #4433 variation and Morris screening** — carried by the variation
  consolidation, not by the rate-of-closure remainder.
- **#4205 / #4260 / #4267 / #4377 ground study and release qualification** — request,
  execution and playback paths exist; qualification evidence is still declaration-only.
- **#4085–#4088 P1AM SCADA product and historian** — consolidated separately.
- **#3973 (P0) NaN clears active alarms** — open and untouched by any PR; both engines
  affected.
- **#3975 CI collects no embedded tests** — 162 test files under `src/**/tests/`
  never run, so the pendulum and `swing_sim` embedded suites gate nothing.

## Per-Tool Handoff Docs

- `src/rate_of_closure/AGENT_HANDOFF.md` — module-registry contract, club-assembly
  binding, release-evidence derivation, packaging rules for collected tests.
- `src/pendulum_simulator/AGENT_HANDOFF.md`
- `src/rotation_converter/AGENT_HANDOFF.md`

New tools: copy `docs/AGENT_HANDOFF_TEMPLATE.md` to `src/<tool>/AGENT_HANDOFF.md`
and fill it in from the tool's real state.

## Gate Commands

```bash
python3 -m ruff check .                       # CI pins ruff==0.14.10
python3 -m ruff format --check .
MYPYPATH=src:src/python/src python3 -m mypy --ignore-missing-imports --follow-imports=skip <changed files>
python3 -m pytest -n auto --timeout=60
python3 scripts/check_module_size_budget.py --max-lines 1200 --include src
python3 scripts/check_minimum_test_contract.py
python3 scripts/check_test_assertions.py --changed-files changed_python_files.txt
python3 scripts/check_docs_governance.py
python3 -m bandit -ll -ii <changed src files>
```

Required checks on `main` are **`quality-gate`** (hosted) and **`tests (3.11)`**
(self-hosted `d-sorg-fleet`, 3.10/3.11/3.12 matrix behind a `pick-runner` job).
A `tests (3.11)` check that has not been *created* means `pick-runner` is still
queued — that is a queue, not a failure.

## Do-Not List

- **Never `git commit --no-verify`** except for a genuinely broken hook; file an issue
  in Repository_Management if a guardrail flags code you did not touch.
- **Never `datetime.UTC`.** The matrix runs Python 3.10. Use `timezone.utc` with
  `# noqa: UP017` on the same physical line, or a `sys.version_info` guard. Check with
  `git grep -n '^from datetime import UTC' -- '*.py'` after any format pass.
- **Never format with a non-pinned ruff.** A newer build reflows files that CI's
  0.14.10 then rejects in Format Check.
- **Never merge with `--merge`.** Three stacked rulesets require linear history and
  allow only squash/rebase.
- **Never add a third-party import to a collected test path without declaring it in
  `requirements.txt`.** A missing package is a collection error that fails the entire
  lane before any test runs (this cost #4447 a full red run on `uvicorn`).
- **Never assume a stack tip subsumes its base chain.** Parents kept advancing after
  children branched; verify containment by ancestry before closing anything.
- On Windows, `MYPYPATH` needs `;` separators, and PyQt6/matplotlib must be installed
  in the venv you type-check with, or mypy reports a false clean.

## Roadmap (ordered)

1. Land the open consolidations and let `tests (3.11)` adjudicate each merged tree.
2. Fix #3973 (NaN clears active alarms) — highest-value open P0.
3. Collect the 162 embedded test files (#3975) so the pendulum and `swing_sim`
   suites actually gate.
4. Deliver the #4104 wasm-pack core and retire the TypeScript physics twins.
5. Address #4033 (vacuous safety tests) and the remaining still-valid issues ranked
   in `_review/pr-consolidation-2026-08-13/ISSUE_AUDIT_TOOLS.md`.
