# Assessment K: Tools Repository Reproducibility & Data Provenance Review

## 1. Executive Summary

- Scientific and calculation-based tools rely on deterministic inputs, earning a high score (9/10).
- Data parsing libraries explicitly use `AssertionError` and `ValueError` when loading corrupted formats.
- Configuration schemas (e.g., UI themes, tool states) lack deep immutability, though local execution guarantees repeatability in the short term.
- **Top Risk**: Machine Learning optimization GUIs (`src/optimizer_gui`) and physical models (`matlab/`) require better configuration serialization for long-term repeatability of simulation results.

## 2. Scorecard (0-10)

| Category                     | Description                                   | Score |
| ---------------------------- | --------------------------------------------- | ----- |
| Data Determinism             | Consistent results from the same inputs       | 9     |
| Experiment/State Tracking    | Serialization of tool configuration           | 6     |
| Dependency Lockfiles         | Precise versions stored in `uv` or `Pipfile`  | 7     |
| Model Provenance             | E.g., Matlab pendulum tracking parameters     | 5     |
| Calculation Auditing         | Clear formulas used under the hood            | 9     |

*Evidence for Tracking (6)*: Calculators provide instant results but do not generally allow the user to save a "calculation session" to disk.
*Evidence for Model Provenance (5)*: `pendulum_model.m` is completely stubbed out, meaning no physical simulation occurs to be tracked.

## 3. Provenance Gap Table

| ID    | Severity | Domain/File | Description | Fix Recommendation | Effort |
| ----- | -------- | ----------- | ----------- | ------------------ | ------ |
| K-001 | Major    | `matlab/models/` | Stubbed code | Implement actual physical simulation | L |
| K-002 | Minor    | `shared/calculators/` | Stateless UIs | Introduce save/load JSON configurations | M |
| K-003 | Minor    | `data_processing` | Formula Logging | Store `safe_eval` strings alongside results | S |

## 4. Remediation Plan

**Immediate (48 Hours):**
- Document in `calculators/` whether stateless UIs are intentional design decisions vs missing features.

**Short-Term (2 Weeks):**
- Expand the Model Generation API to log metadata (e.g., input arrays, chosen solver) alongside its final mesh or output model so the generation steps are fully reproducible.

**Long-Term (6 Weeks):**
- Unify dependency locking natively in the repository build steps to prevent dependency drift across team members over a multi-year horizon.
