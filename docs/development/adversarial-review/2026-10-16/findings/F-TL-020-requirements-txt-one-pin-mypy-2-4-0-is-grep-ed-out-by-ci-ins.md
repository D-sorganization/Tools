# F-TL-020

**requirements.txt 'ONE pin' mypy==2.4.0 is grep'ed out by CI (installs 1.13.0); lock comments against its own numpy pin**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | hygiene  | high       | M      | proposed |

**Locations:** `requirements.txt:20`, `requirements.txt:6`, `requirements-lock.txt:8`, `requirements-lock.txt:11`, `.github/workflows/ci-standard.yml:267`, `.github/workflows/ci-standard.yml:268`

## Finding

- Severity: Medium - Category: hygiene
- Confidence: high
- Location: requirements.txt:6, :20; requirements-lock.txt:8-11; .github/workflows/ci-standard.yml:267-268, :278; pyproject.toml:26, :78, :194
- Evidence:

```
requirements.txt:20   mypy==2.4.0  # ... ONE pin (Tools #4913): ci-standard.yml ... install this exact version
ci-standard.yml:267   python -m pip install mypy==1.13.0
ci-standard.yml:268   grep -v -E '^mypy==' requirements.txt > requirements-ci.txt
requirements-lock.txt:8  # Held below 2.4.0 to satisfy requirements.txt's upper bound (numpy 2.4+
requirements-lock.txt:11 numpy==2.4.6
```

Three numpy floors: `>=1.24.0` (pyproject core), `>=2.0.1,<2.5.0` (pyproject `p1am` extra and requirements.txt), lock `2.4.6`. Plus `uv.lock` (880 KB), `package-lock.json` (613 KB), `setup.py`.

- Failure scenario: a developer following requirements.txt gets mypy 2.4.0 and different diagnostics than CI; requirements.txt:6 justifies a numpy cap by a mypy-1.13 crash while the same file pins mypy 2.4.0. Nobody can answer "what versions does CI test with?" from the declared sources.
- Why a newer engineer misses this: each file was edited to fix one incident; the comments record the incident, not the truth.
- Proposed change: one source — `pyproject.toml` `dev` extra pinned, `uv.lock` as the lock; delete `requirements-lock.txt`, `requirements-rate-pyqt.txt`, `setup.py`; CI uses `uv sync`. Add a test that the mypy version CI installs equals the `dev` pin.
- Verification: `grep -rn 'mypy==' .github requirements* pyproject.toml` returns one value.
- Effort: M

## Draft Pull Request

**Title:** `build(deps): one source of truth for mypy and numpy versions`

### Why This Matters (The Pitch)

Nobody can answer what versions CI tests with from the declared sources. requirements.txt says mypy 2.4.0 is the one pin and that CI installs this exact version. CI then greps that line out of the file and installs mypy 1.13.0. The same file justifies a numpy cap by a mypy 1.13 crash while pinning mypy 2.4.0, the lock file argues with its own pin, and numpy has three different floors across pyproject, requirements and the lock. A developer following requirements.txt gets different diagnostics from CI. The change makes the pyproject dev extra the single pinned source with uv.lock as the lock, drops the redundant lock and setup files, and adds a test that CI's mypy version equals the dev pin. Contributors get reproducible type-check results, and the verification is one grep.

### Proposed PR Body

## Problem

`requirements.txt:20` pins `mypy==2.4.0 # ... ONE pin`, while `.github/workflows/ci-standard.yml:267-268` installs `mypy==1.13.0` and greps the pin out of `requirements.txt`. `requirements-lock.txt:8-11` and `pyproject.toml:26, :78, :194` give three numpy floors (`>=1.24.0`, `>=2.0.1,<2.5.0`, lock `2.4.6`).

## Change

- [ ] Pin versions once in the `pyproject.toml` `dev` extra, with `uv.lock` as the lock
- [ ] Delete `requirements-lock.txt`, `requirements-rate-pyqt.txt` and `setup.py`
- [ ] Switch CI to `uv sync`
- [ ] Add a test that the mypy version CI installs equals the `dev` pin

## Verification

`grep -rn 'mypy==' .github requirements* pyproject.toml` returns one value.

## Risk and Rollback

Changing the mypy version CI uses may surface new type errors; land with the pin CI actually passes on today. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-TL-020

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
