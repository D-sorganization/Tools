# F-TL-018

**Python-floor contract test accepts any workflow with 3.10 in a matrix; conftest contradicts CLAUDE.md**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | S      | proposed |

**Locations:** `tests/test_python_version_contract.py:164`, `tests/test_python_version_contract.py:175`, `conftest.py:124`, `CLAUDE.md:57`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: tests/test_python_version_contract.py:150-179; conftest.py:124; CLAUDE.md:57-66
- Evidence:

```python
for workflow in workflows.glob("*.yml"):
    if workflow == CI_STANDARD: continue
    for match in _MATRIX_RE.finditer(...): covered.add(...)
assert own in covered
```

conftest.py:124: `# workflows. The CI matrix therefore runs a 3.10 lane on purpose.`; CLAUDE.md:57: "`ci-standard.yml` runs `["3.11", "3.12"]` and must not go below 3.11".

- Failure scenario: `src/rotation_converter`, `src/pendulum_simulator` and `psa_package` declare `>=3.10` and are pure Python; the only 3.10 matrices are the seven `maturin-*.yml` wheel builds, which never import those packages. The test passes because _some_ file contains `3.10`. The 3.10 claim is untested — exactly what the docstring says it prevents. (I grepped the three packages for 3.11-only features and found none, today.)
- Why a newer engineer misses this: the test name describes the intent; the implementation checks a weaker proxy.
- Proposed change: map each sub-package to workflows whose `run:` blocks reference its path, or drop the 3.10 claim and delete 150 lines of contract.
- Verification: temporarily set `rotation_converter` to `>=3.9` → test must fail.
- Effort: S

## Draft Pull Request

**Title:** `test(py-floor): tie each 3.10 claim to a workflow that runs it`

### Why This Matters (The Pitch)

The Python-floor contract says every 3.10 claim has a workflow behind it, and it passes by accepting any workflow with 3.10 in a matrix. The only 3.10 matrices are seven maturin wheel builds that never import rotation_converter, pendulum_simulator or psa_package, so the 3.10 claim for those pure-Python packages is untested, which is exactly what the docstring says the test prevents. Meanwhile conftest.py says CI runs a 3.10 lane on purpose and CLAUDE.md says ci-standard must not go below 3.11. Either map each sub-package to a workflow whose run blocks reference its path, or drop the 3.10 claim and delete 150 lines of contract. Contributors and agents get one coherent answer on supported Python, and the review found no 3.11-only features today, so dropping is low-risk.

### Proposed PR Body

## Problem

`tests/test_python_version_contract.py:150-179` adds any non-`ci-standard` workflow whose matrix has 3.10 to `covered` and asserts membership. `conftest.py:124` says the CI matrix runs a 3.10 lane on purpose; `CLAUDE.md:57-66` says `ci-standard.yml` runs 3.11 and 3.12 and must not go below 3.11.

## Change

- [ ] Map each sub-package to workflows whose `run:` blocks reference its path, and require a 3.10 lane for it
- [ ] Or drop the 3.10 claim and delete the contract
- [ ] Reconcile `conftest.py:124` with `CLAUDE.md:57`

## Verification

Temporarily set `rotation_converter` to `>=3.9`; the test must fail.

## Risk and Rollback

Tightening could fail CI until a real 3.10 lane exists; dropping the claim changes declared support. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-TL-018

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
