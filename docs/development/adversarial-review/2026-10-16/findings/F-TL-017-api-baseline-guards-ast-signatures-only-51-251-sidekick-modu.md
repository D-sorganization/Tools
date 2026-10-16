# F-TL-017

**API baseline guards AST signatures only; 51/251 sidekick modules unguarded via empty/missing **all**; 19 dangling exports**

| Severity | Category   | Confidence | Effort | Status   |
| -------- | ---------- | ---------- | ------ | -------- |
| 🟧 High  | governance | high       | M      | proposed |

**Locations:** `tests/test_sidekick_public_api_stability.py:237`, `tests/test_sidekick_public_api_stability.py:265`, `tests/test_shared_package_api_stability.py:151`, `src/shared/python/sidekick/lab/mocap/calibration.py:453`, `src/shared/python/sidekick/lab/__init__.py:7`

## Finding

- Severity: High - Category: governance
- Confidence: high
- Location: tests/test_sidekick_public_api_stability.py:209-239, :265-267; tests/test_shared_package_api_stability.py:139-157; src/shared/python/sidekick/lab/mocap/calibration.py:453; src/shared/python/sidekick/lab/**init**.py:7
- Evidence:

```python
__all__: list[str] = []      # calibration.py:453 — after 11 public classes/functions
```

test: `api_info[symbol] = {"type": "unknown"}` is accepted silently; `assert set(current_api.keys()) == set(baseline_api.keys())` fails on _any_ new file.

- Failure scenario (measured with scratch `check_all.py`): 251 modules; 28 have no `__all__`, 23 declare `__all__ = []` (15 in `lab/mocap`, exporting 94 public defs — `estimate_pnp_pose`, `bundle_adjust_layout`, `solve_pose`…); 19 `__all__` names are not bound in their module (`utils/state_manager.py` exports `"state_manager"`; `ui/tools_sidebar/__init__.py` exports 8 names that resolve nowhere → `from … import SidekickSidebar` is an `ImportError` the baseline calls "stable"); 107 baseline symbols are typed `"unknown"`. Nothing compares return values, exceptions or units — F-TL-001..006 all pass the "contract". Because adding a file fails the test, contributors learn to run `--regenerate-api-baseline` reflexively; a baseline that is regenerated on every PR is a changelog.
- Why a newer engineer misses this: "AST-based stability validation" sounds rigorous; it is a diff of signature strings.
- Proposed change: (1) `test_every_sidekick_module_declares_nonempty_all_or_has_no_public_defs`; (2) fail on `"unknown"`; (3) treat new modules as additions (as the shared-package test does); (4) behavioural golden tests for the ~30 numeric entry points downstream calls, with units and `rel=1e-9`.
- Verification: `python3 check_all.py src/shared/python/sidekick` reports 0 dangling / 0 unexported.
- Effort: M

## Draft Pull Request

**Title:** `test(api): guard exports and behaviour, not just signature strings`

### Why This Matters (The Pitch)

The public-API contract looks rigorous and protects very little. It diffs AST signature strings, so 51 of 251 sidekick modules are fully unguarded: 28 have no **all** and 23 declare an empty one, including 15 lab/mocap modules exporting 94 public definitions. Nineteen **all** names are not bound in their module, so from the sidebar package import SidekickSidebar is an ImportError the baseline calls stable. 107 baseline symbols are typed unknown. Nothing compares return values, exceptions or units, which is why every numeric defect in F-TL-001 through 006 passes the contract. Because any new file fails the test, contributors regenerate the baseline reflexively, and a baseline regenerated on every PR is a changelog, not a guard. The change adds an **all** presence test, fails on unknown, treats new modules as additions, and starts golden tests for about thirty numeric entry points that downstream repos call. UpstreamDrift and Gasification_Model benefit most.

### Proposed PR Body

## Problem

`tests/test_sidekick_public_api_stability.py:209-239, :265-267` and `tests/test_shared_package_api_stability.py:139-157` compare AST signatures only. Measured: 251 modules, 28 with no `__all__`, 23 with `__all__ = []` (15 in `lab/mocap`, 94 public defs, e.g. `src/shared/python/sidekick/lab/mocap/calibration.py:453`), 19 unbound `__all__` names, 107 symbols typed `"unknown"`.

## Change

- [ ] Add `test_every_sidekick_module_declares_nonempty_all_or_has_no_public_defs`
- [ ] Fail on `"unknown"` symbol types
- [ ] Treat new modules as additions, as the shared-package test does
- [ ] Add behavioural golden tests for about 30 numeric entry points with units and `rel=1e-9`

## Verification

`python3 check_all.py src/shared/python/sidekick` reports 0 dangling / 0 unexported.

## Risk and Rollback

The first run will fail until the 51 unguarded modules are fixed, so land the tests together with `__all__` repairs. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: golden tests can land per calculator after F-TL-001..006.

Refs: adversarial review 2026-10-16, F-TL-017

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
