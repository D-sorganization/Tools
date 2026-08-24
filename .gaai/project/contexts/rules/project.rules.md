# Project Rules — Tools (GAAI Fleet)

## Safety

1. All AI work uses an isolated feature branch and worktree. Never commit or
   push directly to `main`.
2. PRs target the protected default branch, `main`, as required by `CLAUDE.md`
   and the fleet release contract. Auto-merge may be armed only when the
   repository policy allows it and must still wait for required checks and
   review; never bypass a hold, review, or protected check.
3. No destructive git history operations.
4. No secret commits (.env, API keys, credentials).

## Quality Gates (CI) — MANDATORY PRE-PR CHECKLIST

**Before creating a PR, the delivery agent MUST run these commands and fix all issues:**

```bash
# Step 1: Auto-format (fixes most issues automatically)
python3 -m ruff format .

# Step 2: Lint and auto-fix what's possible
python3 -m ruff check --fix .

# Step 3: Verify clean (must exit 0)
python3 -m ruff format --check .
python3 -m ruff check .

# Step 4: Run tests on changed files
python3 -m pytest -x --timeout=60 -q
```

**If any step fails after auto-fix, manually resolve before proceeding. Do NOT create a PR with known lint/format failures.**

5. `ruff check` must pass on ALL modified Python files before PR creation.
6. `ruff format --check` must pass (this repo uses ruff format, NOT black. NEVER run black.).
7. No new `print()` calls in `src/` (use logging).
8. Minimum test coverage: 10%. Delta-based checks — coverage must not decrease on touched files.
9. Cross-repo shared library caution: changes to shared packages may affect Gasification_Model, UpstreamDrift, and AffineDrift. Verify downstream impact before modifying shared code.
10. No TODO/FIXME comments unless a tracked GitHub issue exists.

## CI Watch (Post-PR)

11. After creating a PR, invoke the `ci-watch-and-fix` skill to monitor CI and remediate failures.
12. Do NOT mark a story as `done` until CI passes or the skill escalates after 3 remediation cycles.

## Escalation

11. If a story requires modifying CI pipelines in a breaking way — escalate.
12. If a story touches shared/core modules affecting multiple subsystems — escalate.
13. If a story modifies a shared library consumed by other repos — escalate and verify consumer compatibility.

---

## Coding Principles (Mandatory — enforced in QA)

### TDD (Test-Driven Development)

- Write tests BEFORE implementation code.
- Every new public function/method must have at least one test.
- Test file must exist before or in the same commit as the implementation.
- If modifying existing code, add tests for the modified behavior first.

### DRY (Don't Repeat Yourself)

- No duplicated logic blocks >5 lines. Extract shared logic into helpers.
- Before writing new utility code, search for existing implementations.
- If you find yourself copying code, refactor into a shared function.

### DbC (Design by Contract)

- Public functions must validate preconditions (raise ValueError/TypeError on invalid input).
- Document postconditions in docstrings for non-trivial functions.
- Use assert statements for invariants in non-hot-path code.

### LOD (Law of Demeter)

- No method chains >2 levels (e.g., `a.b.c.d()` violates LOD).
- Functions should only call methods on: self, parameters, objects they create, direct attributes.
- If you need deep access, add a delegating method to the intermediate object.
