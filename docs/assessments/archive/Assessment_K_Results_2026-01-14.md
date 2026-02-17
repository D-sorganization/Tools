# Assessment K: Reproducibility & Provenance Results

**Date:** 2026-01-14
**Assessor:** Jules

## 1. Environment Reproducibility

**Score: 3/10**

- **Python**: **Major Gap**. No `poetry.lock` or `pip-tools` compile. `requirements.txt` is unpinned or loose.
- **Node**: Uses `package-lock.json` (good) but `pnpm` requirement adds friction.

## 2. Determinism

**Score: 5/10**

- **Testing**: Tests were flaky due to global state (mocks), now fixed.
- **Data**: `Data_Processor` seems deterministic. `Solar System` (physics) should be, but depends on time steps.

## Remediation Roadmap

- **Immediate**: Freeze dependencies into a lock file.
- **Short-term**: Add a Dockerfile to define the canonical environment.
