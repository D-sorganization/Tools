# Benchmarking Infrastructure

This directory contains performance benchmarking infrastructure and baseline metrics for the Tools library.

## Quick Start

### Run Benchmarks Locally

```bash
# Run all benchmarks with detailed output
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" --benchmark-only -v

# Run specific benchmark file
python3 -m pytest tests/benchmarks/test_rotation_converter_perf.py --override-ini="addopts=" --benchmark-only

# Run and compare against baseline
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" --benchmark-only \
  --benchmark-compare=.benchmarks/baseline.json \
  --benchmark-compare-fail=min:10%
```

### Key Files

- **`baseline.json`** — Current performance baseline (pytest-benchmark JSON format)
  - Generated via: `pytest tests/benchmarks/ --benchmark-json=.benchmarks/baseline.json`
  - Use for regression detection: `--benchmark-compare=baseline.json`

- **`PERFORMANCE_SLA.md`** — Service Level Agreements for critical operations
  - Pressure drop: < 100ms per calculation
  - Rotation converter: < 50ms per conversion
  - Data processor: < 500ms for 10K-row filtering

- **`results.json`** — Latest CI run results (auto-generated, ephemeral)

## Benchmark Structure

All benchmarks live in `tests/benchmarks/`:

```
tests/benchmarks/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_pressure_drop_perf.py        # Pressure drop calculator benchmarks
├── test_rotation_converter_perf.py   # Rotation conversion benchmarks
└── test_data_processor_perf.py       # FFT filtering benchmarks
```

### Benchmark Markers

Benchmarks use pytest markers for categorization:

```python
pytestmark = pytest.mark.benchmark  # Required; enables pytest-benchmark
@pytest.mark.performance           # Optional; groups performance tests
```

### Running Specific Benchmarks

```bash
# Run only rotation converter benchmarks
python3 -m pytest tests/benchmarks/test_rotation_converter_perf.py \
  --override-ini="addopts=" --benchmark-only

# Run only a specific test class
python3 -m pytest tests/benchmarks/test_rotation_converter_perf.py::TestRotationConverterBasic \
  --override-ini="addopts=" --benchmark-only

# Run single benchmark
python3 -m pytest tests/benchmarks/test_rotation_converter_perf.py::TestRotationConverterBasic::test_euler_to_quaternion_conversion \
  --override-ini="addopts=" --benchmark-only
```

## pytest-benchmark Features

### JSON Export

```bash
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" \
  --benchmark-only \
  --benchmark-json=.benchmarks/results.json
```

Produces structured JSON with full statistics (min/max/mean/stddev/rounds/iterations).

### Regression Detection

```bash
# Fail if any benchmark regresses > 10%
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" \
  --benchmark-only \
  --benchmark-compare=.benchmarks/baseline.json \
  --benchmark-compare-fail=min:10%
```

Exit codes:
- `0` — All within bounds
- `1` — One or more regressions
- `2` — Benchmark execution error

### Custom Comparison

```bash
# Compare two specific JSON files
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" \
  --benchmark-only \
  --benchmark-compare=/previous/results.json \
  --benchmark-compare=/new/results.json
```

### Report Options

```bash
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" \
  --benchmark-only \
  --benchmark-columns=min,max,mean,stddev,median,iqr,ops,rounds \
  --benchmark-histogram          # Generate histogram PNG for each benchmark
  --benchmark-save-data          # Save aggregated data for trending
  -v
```

## CI Integration

### GitHub Actions Workflow

`.github/workflows/benchmark-suite.yml` runs on:
- Push to `main`/`master`
- Every pull request
- Manual trigger via Actions tab

**Workflow Steps:**
1. Checkout code and set up Python
2. Run full benchmark suite (serial, 30-minute timeout)
3. Compare against baseline.json (10% regression threshold)
4. Generate markdown report
5. Upload results as artifact
6. Comment benchmark summary on PR

**Branch Protection Rule** (Optional):
```yaml
required_status_checks:
  - benchmark-suite / benchmark-foundation  # Can make advisory-only
```

## Creating New Benchmarks

### Benchmark Template

```python
"""Performance benchmarks for <module>.

Measures the performance of <operation> under <conditions>.
SLA target: < Xms.
"""

from __future__ import annotations

import pytest
import numpy as np

try:
    from module import function_to_benchmark
except (ImportError, NameError):
    pytest.skip("module not available", allow_module_level=True)

pytestmark = pytest.mark.benchmark


@pytest.mark.performance
class TestMyBenchmarks:
    """Performance benchmarks for my feature."""

    def test_basic_operation(self, benchmark):
        """Benchmark basic operation.

        SLA: < 50ms
        Tests: Single operation under standard conditions
        """
        result = benchmark(function_to_benchmark, arg1, arg2)
        assert result is not None

    def test_scaled_operation(self, benchmark, sample_large_array):
        """Benchmark operation on large dataset.

        SLA: < 500ms
        Tests: Scaling to 10K elements
        """
        result = benchmark(function_to_benchmark, sample_large_array)
        assert len(result) == len(sample_large_array)
```

### Fixture Pattern

Use `conftest.py` fixtures for common test data:

```python
@pytest.fixture
def sample_large_array():
    """Large array for scale benchmarks (10000 elements)."""
    return np.random.randn(10000)
```

Access in test:
```python
def test_operation(self, benchmark, sample_large_array):
    result = benchmark(function, sample_large_array)
```

### SLA Documentation

Every benchmark class should document:
- **Target SLA** in docstring
- **Test coverage** (what gets measured)
- **Rationale** (why this SLA matters)

Update `PERFORMANCE_SLA.md` with new entries:

```markdown
| Operation | Module | SLA Target | Actual | Status | Notes |
|-----------|--------|-----------|--------|--------|-------|
| My operation | my_module | < 50ms | ~20μs | ✓ PASS | Description |
```

## Troubleshooting

### "Can't have both --benchmark-only and --benchmark-disable options"

**Cause:** Default pytest config has `--benchmark-disable` in `addopts`.

**Fix:** Use `--override-ini="addopts="`

```bash
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" --benchmark-only
```

### "ImportError: No module named 'module'"

**Cause:** Benchmark imports a module that's not installed.

**Fix:** Install with dev dependencies:

```bash
pip install -e ".[dev]"
```

Or skip the benchmark:

```python
try:
    from module import function
except ImportError:
    pytest.skip("module not available", allow_module_level=True)
```

### "benchmark fixture not recognized"

**Cause:** pytest-benchmark plugin not installed.

**Fix:**

```bash
pip install pytest-benchmark>=4.0.0
```

### Benchmark takes too long

**Cause:** Default min_time calibration is slow for fast operations.

**Fix:** Adjust in conftest.py or CLI:

```python
# conftest.py
def pytest_benchmark_configure(config):
    config.benchmarkconfig.min_time = 0.000001  # 1 microsecond
```

Or via CLI:
```bash
pytest tests/benchmarks/test_rotation_converter_perf.py \
  --override-ini="addopts=" \
  --benchmark-min-rounds=5 \
  --benchmark-only
```

## Performance Regression Workflow

1. **Local Development:**
   ```bash
   git checkout -b feature/my-optimization
   # Implement changes...
   python3 -m pytest tests/benchmarks/ --override-ini="addopts=" \
     --benchmark-only \
     --benchmark-compare=.benchmarks/baseline.json
   ```

2. **PR Submission:**
   - GitHub Actions auto-runs benchmarks
   - Results posted as comment on PR
   - If regression > 10%, workflow fails (advisory)

3. **Investigation:**
   - Review PR diff to understand cause
   - Profile with `cProfile` or `py-spy`
   - Consider algorithmic vs. implementation optimization

4. **Resolution:**
   - **Option A:** Revert change
   - **Option B:** Optimize further
   - **Option C:** Update SLA (requires team approval)

5. **Merge:**
   - Once regression resolved, PR merges
   - Baseline auto-updates on main push

## Baseline Management

### Updating Baseline

After a justified SLA change or major optimization:

```bash
# Run benchmarks and save as new baseline
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" \
  --benchmark-only \
  --benchmark-json=.benchmarks/baseline.json

# Commit to git
git add .benchmarks/baseline.json
git commit -m "perf: update benchmark baseline (issue #2413)"
```

### Baseline History

Baselines are version-controlled in git:

```bash
git log --oneline -- .benchmarks/baseline.json
```

View historical trends:

```bash
git show <commit>:.benchmarks/baseline.json | \
  python3 -c "import json, sys; \
  data = json.load(sys.stdin); \
  print('\n'.join(f\"{b['name']}: {b['stats']['mean']*1e6:.1f}μs\" \
    for b in data['benchmarks']))"
```

## Advanced: Customizing pytest-benchmark

### Disable Garbage Collection

By default, GC is disabled during benchmarks (more stable). To enable:

```bash
pytest tests/benchmarks/ --benchmark-disable-gc
```

### Warmup Iterations

Increase for JIT warm-up on cached operations:

```bash
pytest tests/benchmarks/ --benchmark-warmup=100
```

### Timer Resolution

Force specific timer (e.g., for comparing across platforms):

```python
# conftest.py
import time
import pytest

def pytest_benchmark_configure(config):
    config.benchmarkconfig.timer = time.time  # Wall-clock (vs. perf_counter)
```

### Integration with External Benchmarks

pytest-benchmark integrates with:
- **asv** (Airspeed Velocity) for historical tracking
- **pytest-json-report** for CI/CD dashboards
- **codespeed** for web-based performance tracking

See: https://pytest-benchmark.readthedocs.io/

## References

- **pytest-benchmark docs:** https://pytest-benchmark.readthedocs.io/
- **Our SLA spec:** `PERFORMANCE_SLA.md`
- **CI workflow:** `.github/workflows/benchmark-suite.yml`
- **GitHub issue:** #2413 (Systematic performance optimization)

---

**Last Updated:** 2026-04-30
**Maintainer:** Performance Working Group
