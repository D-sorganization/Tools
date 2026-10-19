# F-TL-015

**safe_eval allows str/bytes repetition bombs reachable over HTTP**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | security | high       | S      | proposed |

**Locations:** `src/shared/python/safe_eval.py:308`, `src/shared/python/safe_eval.py:309`, `src/shared/python/calc_backend/routers/ode_solver.py:82`

## Finding

- Severity: Medium - Category: security
- Confidence: high
- Location: src/shared/python/safe_eval.py:306-310, :446-476; src/shared/python/calc_backend/routers/ode_solver.py:82; src/p1am_control_system/backend/data_explorer_expression.py
- Evidence:

```python
if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
    if len(node.value) > MAX_STR_CONSTANT_LENGTH:
```

Module docstring (:72-76) claims #3290 closed the "repetition bomb" (`"x"*10**9`).

- Failure scenario (run): `safe_eval_math('"abcdefghij" * 5000000')` → 50 MB str in 0.03 s; `b"\x00"*100000000` → 100 MB; `"a" * n` with a namespace int works too; `'"a"*(10**9)'` passes validation (exponent 9) and allocates 1 GB. Posted to the ODE-solver endpoint this is a one-request memory DoS.
- Why a newer engineer misses this: the guard was written for the _constant_ and for `**`; `*` with a string operand is a different code path.
- Proposed change: in `validate_expression`, reject `ast.BinOp(op=Mult)` when either operand is a `str`/`bytes` constant, and reject `str`/`bytes` constants entirely for the math namespaces; optionally `resource.setrlimit(RLIMIT_AS)` in the HTTP routers.
- Verification: `pytest.raises(ValueError)` for the three expressions above.
- Effort: S

## Draft Pull Request

**Title:** `fix(safe-eval): reject string and bytes repetition in math expressions`

### Why This Matters (The Pitch)

One HTTP request can make this server allocate a gigabyte. The safe_eval module claims to have closed the repetition bomb, but the guard covers string constants and the power operator, not multiplication with a string operand. safe_eval_math of a ten-character string times five million returns a 50 MB string in 0.03 seconds, a bytes literal times 100 million returns 100 MB, and a string times 10 to the 9 passes validation and allocates 1 GB. Post that to the ODE-solver endpoint in calc_backend and you have a single-request memory denial of service. The fix is a few lines in the validator rejecting Mult when either operand is a string or bytes constant, and optionally rejecting string constants entirely for math namespaces. The calc_backend service and the P1AM data explorer are the beneficiaries, and the three failing expressions are already written for the test.

### Proposed PR Body

## Problem

`src/shared/python/safe_eval.py:306-310, :446-476` bounds string constants and `**` but not `*` with a str/bytes operand, despite the docstring (:72-76) claiming #3290 closed it. Reachable via `src/shared/python/calc_backend/routers/ode_solver.py:82` and `src/p1am_control_system/backend/data_explorer_expression.py`. `'"abcdefghij" * 5000000'` yields a 50 MB str; `'"a"*(10**9)'` allocates 1 GB.

## Change

- [ ] In `validate_expression`, reject `ast.BinOp(op=Mult)` when either operand is a str/bytes constant
- [ ] Reject str/bytes constants entirely in math namespaces
- [ ] Optionally `resource.setrlimit(RLIMIT_AS)` in the HTTP routers

## Verification

`pytest.raises(ValueError)` for the three expressions above.

## Risk and Rollback

Expressions that legitimately multiply strings will be rejected; none are expected in a math evaluator. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-TL-015

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
