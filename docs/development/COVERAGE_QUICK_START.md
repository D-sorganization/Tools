# Coverage Measurement — Quick Start

## TL;DR

Generate coverage reports locally:

```bash
# Run tests with coverage + generate HTML report
python3 -m pytest tests/ --cov=src --cov-report=html --cov-fail-under=0
open htmlcov/index.html
```

Or use the convenience script:

```bash
./scripts/run_coverage.sh
```

---

## Key Files

| File                              | Purpose                                                                   |
| --------------------------------- | ------------------------------------------------------------------------- |
| `.coveragerc`                     | Coverage measurement configuration (source paths, exclusions, thresholds) |
| `config/coverage_baseline.json`   | Current ratchet baseline (60%) — used by CI for regression detection      |
| `config/coverage_policy.json`     | Policy thresholds (minimum total, per-package targets)                    |
| `scripts/measure_coverage.py`     | Compare coverage to baseline and policy                                   |
| `scripts/run_coverage.sh`         | End-to-end local coverage measurement                                     |
| `COVERAGE_SETUP.md`               | Complete documentation                                                    |
| `assessments/hot_path_modules.md` | Critical modules requiring 80% coverage (Phase 2+)                        |

---

## Local Workflow

### Generate HTML Report

```bash
python3 -m pytest tests/ --cov=src --cov-report=html
# Report in: htmlcov/index.html
```

### Check Against Baseline

```bash
python3 scripts/measure_coverage.py \
  --coverage-file coverage.xml \
  --baseline-file config/coverage_baseline.json \
  --policy-file config/coverage_policy.json
```

### Run Full Pipeline

```bash
./scripts/run_coverage.sh --baseline
```

---

## CI Integration

The PR CI workflow (`.github/workflows/ci-standard.yml`) automatically:

1. **Measures coverage** during test execution

   ```bash
   python -m pytest ... --cov=src --cov-report=xml:coverage.xml
   ```

2. **Validates touched tracked packages against policy**

   ```bash
   python3 scripts/check_coverage_policy.py \
     --coverage-file coverage.xml \
     --policy-file config/coverage_policy.json \
     --baseline-file config/coverage_baseline.json \
     --changed-files changed_python_files.txt
   ```

3. **Uploads results**
   - `coverage_trend_*.json` artifact contains trend data
   - Check CI logs for "Coverage policy evaluation" section

The nightly full-suite workflow (`.github/workflows/full-suite-nightly.yml`)
runs `tests/ src/` with repo-wide coverage and calls
`scripts/check_coverage_policy.py` without `--changed-files`. That lane owns the
total-coverage non-regression ratchet.

---

## Interpreting Results

### HTML Report (Local)

Open `htmlcov/index.html` in a browser:

- **Green:** Covered code (tested)
- **Red:** Uncovered code (not tested)
- **Yellow:** Partially covered (some branches untested)

Click into modules to see line-by-line coverage.

### CI Failures

Check workflow logs for "Coverage Policy Gate" section:

```
Coverage policy evaluation:
- total: 6.96%
- src/shared/python/notes: 49.3%

[PASS] src/shared/python/notes: 49.3% (threshold: 49.0%)
[PASS] src/shared/python/upstream_drift_tools: 23.2% (threshold: 15.0%)
```

Common failures:

- `total ... below effective minimum` → Full-suite coverage dropped below the ratchet floor
- `total ... regressed beyond allowed drop` → Coverage decreased >2% from baseline
- `package ... below threshold` → Specific module fell short

---

## Current Status

**Baseline:**

- Total ratchet: 60%
- PR gate: changed tracked-package thresholds only
- Nightly full-suite gate: repo-wide total non-regression, max drop 2%

**Hot-path modules (aspirational 80% targets — NOT a CI gate):**

> The `hot_path_modules_phase2` config block these targets referred to was dead
> config (read by no code) and has been removed (issue #3357). The list below is
> a planning roadmap only; CI does not enforce it.

- `src/pressure_drop_calculator` — Core process engineering
- `src/rotation_converter` — Robotics kinematics
- `src/shared/python/model_generation` — URDF generation
- `src/shared/python/upstream_drift_tools` — Shared library

See `assessments/hot_path_modules.md` for detailed information.

---

## Common Tasks

### Improve Coverage for a Module

1. Open `htmlcov/index.html`, find your module
2. Look for red lines (uncovered code)
3. Write tests to cover those lines
4. Re-run `pytest --cov=src --cov-report=html` to verify
5. Commit tests with your changes

### Check Specific Module Coverage

```bash
python3 -m pytest tests/ \
  --cov=src/shared/python/notes \
  --cov-report=term-missing
```

Shows which lines in `notes/` are uncovered.

### Update Baseline (After Intentional Improvement)

After expanding test coverage significantly:

```bash
# Generate new baseline
python3 -m pytest tests/ --cov=src --cov-report=json:coverage.json

# Update config/coverage_baseline.json with new totals
# (Coordinate with team before committing baseline changes)
```

---

## Troubleshooting

### "No module named 'pytest_cov'"

Install the package:

```bash
pip install pytest-cov
```

### coverage.xml Not Generated

Verify flags:

```bash
python3 -m pytest tests/ --cov=src --cov-report=xml
```

### HTML Report Not Opening

Make sure graphviz is installed (optional, for branch coverage graphs):

```bash
sudo apt install graphviz  # Linux
brew install graphviz       # macOS
```

### Tests Pass but Coverage Low

Common causes:

- Test files in different pythonpath than source
- Conditional imports not exercised
- GUI/display code (headless CI sets `QT_QPA_PLATFORM=offscreen`)

Check `pyproject.toml` `[tool.pytest.ini_options]` pythonpath config.

---

## Next Steps

See:

- **Full setup details:** `COVERAGE_SETUP.md`
- **Phase 2 roadmap:** `assessments/hot_path_modules.md`
- **Issue #2406:** Comprehensive test coverage measurement and ratcheting
