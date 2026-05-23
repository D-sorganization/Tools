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
| `config/coverage_baseline.json`   | Current baseline (6.25%) — used by CI for regression detection            |
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

The CI workflow (`.github/workflows/ci-standard.yml`) automatically:

1. **Measures coverage** during test execution

   ```bash
   python -m pytest ... --cov=src --cov-report=xml:coverage.xml
   ```

2. **Validates against policy**

   ```bash
   python3 scripts/check_coverage_policy.py \
     --coverage-file coverage.xml \
     --policy-file config/coverage_policy.json \
     --baseline-file config/coverage_baseline.json
   ```

3. **Uploads results**
   - `coverage_trend_*.json` artifact contains trend data
   - Check CI logs for "Coverage policy evaluation" section

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

- `total ... below minimum 6.0%` → Coverage dropped too much
- `total ... regressed beyond allowed drop` → Coverage decreased >2% from baseline
- `package ... below threshold` → Specific module fell short

---

## Current Status

**Baseline (2026-04-30):**

- Total: 6.25% (expanding over time)
- Core test suite: 476 tests
- Policy: minimum 6%, max drop 2%

**Hot-path modules (Phase 2 targets — 80%):**

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
