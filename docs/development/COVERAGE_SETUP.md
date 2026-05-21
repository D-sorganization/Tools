# Coverage Measurement and Ratcheting — Phase 1 Setup

## Overview

This document describes the Phase 1 implementation of comprehensive test coverage measurement and ratcheting for the Tools monorepo.

**Current Status:**

- Coverage configuration: ✓ Complete
- CI integration: ✓ Integrated into `ci-standard.yml`
- Baseline established: ✓ 6.25% (core test suite)
- Policy enforcement: ✓ Configured

---

## Quick Start

### Run Coverage Locally

```bash
# Generate HTML report + XML/JSON + check against baseline
./scripts/run_coverage.sh --baseline

# Or run pytest directly with coverage
python3 -m pytest tests/ --cov=src --cov-report=html --cov-fail-under=60
```

Then open `htmlcov/index.html` in a browser.

### Check Coverage Against Policy

```bash
# Compare current coverage to baseline and policy thresholds
python3 scripts/measure_coverage.py \
  --coverage-file coverage.xml \
  --baseline-file config/coverage_baseline.json \
  --policy-file config/coverage_policy.json
```

---

## Configuration Files

### `.coveragerc` — Coverage Measurement Configuration

Located at repository root.

**Key settings:**

- `source = src` — Measure only code in `src/` directory
- `branch = true` — Measure both line and branch coverage
- `parallel = true` — Support parallel test execution
- `fail_under = 25` — Minimum total coverage threshold (25%)
- `output = coverage.xml` — Machine-readable format for CI

**Omissions (excluded from coverage):**

- `*/tests/*`, `*/test_*.py` — Test code itself
- `src/data_processing/`, `src/document_processing/`, etc. — Legacy directories
- `conftest.py`, `setup.py` — Infrastructure files

See `.coveragerc` for full list of excluded patterns.

### `config/coverage_policy.json` — Coverage Policy and Thresholds

```json
{
  "minimum_total_percent": 25.0,
  "max_total_drop_percent": 2.0,
  "tracked_packages": {
    "src/shared/python/notes": 49.0,
    "src/shared/python/upstream_drift_tools": 15.0,
    "src/shared/python/signal_toolkit": 10.0,
    "src/shared/python/model_generation": 10.0
  },
  "hot_path_modules": {
    "src/pressure_drop_calculator": 80.0,
    "src/rotation_converter": 80.0,
    "src/shared/python/model_generation": 80.0,
    "src/shared/python/upstream_drift_tools": 80.0
  }
}
```

**Thresholds explained:**

- **minimum_total_percent** (25%): Total repository coverage must stay above 25%
- **max_total_drop_percent** (2%): If baseline is 6%, PR cannot drop below 4%
- **tracked_packages**: Per-package minimums (higher confidence modules)
- **hot_path_modules**: Critical path modules requiring 80% coverage (Phase 2+ target)

### `config/coverage_baseline.json` — Current Baseline

Current baseline (established 2026-04-30):

```json
{
  "total_percent": 6.25,
  "package_percent": {}
}
```

This is the snapshot against which regressions are measured. Updated when baseline is intentionally raised.

### `assessments/coverage_baseline.json` — Detailed Baseline Record

Full snapshot of module-level coverage at baseline, for tracking and reporting.

---

## CI Integration

### Where Coverage Runs

**File:** `.github/workflows/ci-standard.yml`

**Job:** `tests` (runs for Python 3.10, 3.11, 3.12)

**Steps:**

1. **Run Tests with Coverage** — Uses pytest-cov to generate `coverage.xml`

   ```bash
   python -m pytest "${core_tests[@]}" \
     --cov=src \
     --cov-report=xml:coverage.xml \
     --cov-fail-under=0
   ```

2. **Coverage Policy Gate** — Compares XML to baseline

   ```bash
   python3 scripts/check_coverage_policy.py \
     --coverage-file coverage.xml \
     --policy-file config/coverage_policy.json \
     --baseline-file config/coverage_baseline.json \
     --output-json coverage_trend_${python_version}.json
   ```

3. **Upload Coverage Artifact** — Stores trend data
   ```yaml
   - uses: actions/upload-artifact@v4
     with:
       name: coverage-trend-${{ matrix.python-version }}
       path: coverage_trend_*.json
   ```

### How to Interpret CI Failures

**Check the logs:**

1. Look for "Coverage policy evaluation" section in CI logs
2. Check which packages/thresholds failed
3. Review the "coverage*trend*\*.json" artifacts in CI

**Common failures:**

- `total coverage X% below minimum 25%` → Increase test coverage
- `total coverage X% regressed beyond allowed drop` → Tests passing but coverage decreased
- `package ... coverage X% below threshold Y%` → Module-specific coverage shortfall

---

## Scripts and Tools

### `scripts/measure_coverage.py`

Compare current coverage against baseline and policy thresholds.

**Usage:**

```bash
python3 scripts/measure_coverage.py \
  --coverage-file coverage.xml \
  --baseline-file config/coverage_baseline.json \
  --policy-file config/coverage_policy.json \
  --output-dir coverage_reports
```

**Output:**

- `coverage_reports/coverage_report.json` — Detailed policy evaluation

### `scripts/run_coverage.sh`

End-to-end coverage measurement with HTML report generation.

**Usage:**

```bash
./scripts/run_coverage.sh [--baseline-comparison]
```

**Output:**

- `htmlcov/index.html` — Interactive coverage report (open in browser)
- `coverage.xml` — CI-readable format
- `coverage.json` — JSON metrics
- `coverage.log` — Test output log

### `scripts/check_coverage_policy.py`

(Existing) Validates coverage against policy during CI.

---

## Test Coverage Status

### Current Baseline (2026-04-30)

**Overall:** 6.25% (core test suite: 476 tests)

**Top modules by coverage:**

| Module                                 | Coverage         | Status                 |
| -------------------------------------- | ---------------- | ---------------------- |
| src/shared/python/safe_eval.py         | 100.0%           | ✓ Complete             |
| src/shared/python/contracts.py         | 78.95%           | ✓ Strong               |
| src/shared/python/notes                | 49.3%            | ✓ Tracked              |
| src/shared/python/gui_launcher         | 44.49%           | → Target 80% (Phase 2) |
| src/shared/python/upstream_drift_tools | 23.2%            | → Target 80% (Phase 2) |
| src/rotation_converter                 | 10.17%           | → Target 80% (Phase 2) |
| src/pressure_drop_calculator           | Not yet measured | → Target 80% (Phase 2) |

### Phase 1 Targets (Current)

✓ **Total repository:** ≥25% (currently 6.25%, ratchet over time)
✓ **Tracked packages:** Enforce per-package minimums (10-49%)
→ **Hot-path modules:** Target 80% (Phase 2+)

### Hot-Path Critical Modules (for Phase 2+)

These modules should reach **80% coverage** priority:

1. **src/pressure_drop_calculator** — Core process engineering
2. **src/rotation_converter** — Robotics kinematics
3. **src/shared/python/model_generation** — URDF generation
4. **src/shared/python/upstream_drift_tools** — Shared library

---

## Best Practices

### Running Tests

**For coverage-aware testing:**

```bash
# Run with coverage measurement
python3 -m pytest tests/ --cov=src --cov-report=html

# Run subset with coverage
python3 -m pytest tests/shared/python/ --cov=src/shared/python --cov-report=term

# Check without enforcing minimum
python3 -m pytest tests/ --cov=src --cov-fail-under=0
```

### Improving Coverage

**General strategies:**

1. **Identify untested code:** Open `htmlcov/index.html`, look for red lines
2. **Add test cases:** Write tests for uncovered paths
3. **Run locally first:** `./scripts/run_coverage.sh` to see impact before PR
4. **Commit baseline updates:** When intentionally raising baseline

### Baseline Updates

**When to update baseline:**

- After intentional test expansion (0.5%+ increase)
- As part of planned coverage ratcheting (Phase 2+)
- **NOT** for every small fluctuation

**How to update:**

```bash
# After improving coverage
python3 -m pytest tests/ --cov=src --cov-report=json:coverage.json
# Copy coverage metrics to config/coverage_baseline.json and assessments/
```

---

## Phase 2 + Future Work

Phase 1 establishes the foundation. Future phases will:

1. **Phase 2:** Ratchet total coverage from 25% → 35% with enforcement
2. **Phase 2b:** Reach 80% on hot-path modules (pressure drop, rotation, model generation)
3. **Phase 3:** Per-package coverage ratcheting (all tracked packages ≥50%)
4. **Phase 4+:** Integration with UpstreamDrift and Gasification_Model CI

---

## Troubleshooting

### Coverage.xml Not Generated

**Check:**

1. Verify pytest-cov installed: `pip install pytest-cov`
2. Verify coverage command: `python3 -m pytest --cov=src --cov-report=xml`
3. Check `.coveragerc` exists and is readable

### Coverage Lower Than Expected

**Common causes:**

1. Tests not importing modules under measurement
2. Conditional imports not exercised
3. GUI/Qt code skipped in headless CI

**Solutions:**

1. Verify test paths in `pythonpath` config (pyproject.toml)
2. Check `QT_QPA_PLATFORM=offscreen` is set (for headless tests)
3. Add test\_\*.py files to `pythonpath` if needed

### Policy Gate Failing

Run locally to debug:

```bash
python3 scripts/measure_coverage.py --coverage-file coverage.xml
```

Check `coverage_reports/coverage_report.json` for details.

---

## Files Modified / Created

**Phase 1 deliverables:**

- ✓ `.coveragerc` — Updated with complete configuration
- ✓ `pyproject.toml` — Coverage settings (already present)
- ✓ `config/coverage_policy.json` — Updated with hot-path modules
- ✓ `config/coverage_baseline.json` — Baseline snapshot (6.25%)
- ✓ `assessments/coverage_baseline.json` — Detailed module coverage record
- ✓ `scripts/measure_coverage.py` — Coverage comparison tool
- ✓ `scripts/run_coverage.sh` — Local coverage measurement script
- ✓ `.github/workflows/ci-standard.yml` — Coverage reporting (already integrated)
- ✓ `COVERAGE_SETUP.md` — This documentation

**Related existing files:**

- `scripts/check_coverage_policy.py` — Existing policy validation (reused)
- `scripts/check_coverage_gates.py` — Existing gate enforcement (reused)

---

## References

- **Issue:** #2406 — Comprehensive test coverage measurement and ratcheting
- **Coverage.py docs:** https://coverage.readthedocs.io/
- **pytest-cov docs:** https://pytest-cov.readthedocs.io/
- **Policy enforcement:** `config/coverage_policy.json`
