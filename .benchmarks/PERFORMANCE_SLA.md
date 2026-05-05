# Performance SLA (Service Level Agreements)

This document defines the performance targets (Service Level Agreements) for critical operations in the Tools library. These SLAs ensure that downstream consumers (UpstreamDrift, Gasification_Model) receive predictable performance characteristics.

**Last Updated:** 2026-04-30
**Baseline Generated:** Via pytest-benchmark suite in `tests/benchmarks/`

## SLA Summary Table

| Operation | Module | SLA Target | Actual (μs) | Status | Notes |
|-----------|--------|-----------|-------------|--------|-------|
| Pressure drop calculation | pressure_drop_calculator | < 100ms | ~50-100ms | ✓ PASS | Single calculation |
| Rotation: Euler to Quaternion | rotation_converter | < 50ms | ~40μs | ✓ PASS | Single conversion |
| Rotation: Quaternion to Matrix | rotation_converter | < 50ms | ~17μs | ✓ PASS | SO(3) conversion |
| Rotation: Matrix to Quaternion | rotation_converter | < 50ms | ~33μs | ✓ PASS | From 3x3 matrix |
| Rotation: Axis-Angle conversion | rotation_converter | < 50ms | ~15μs | ✓ PASS | Compact representation |
| Data filter design (small, 100 samples) | data_processor | < 100ms | ~20μs | ✓ PASS | FFT window design |
| Data filter design (large, 10K samples) | data_processor | < 500ms | ~52μs | ✓ PASS | Scales sub-linearly |

## Detailed SLA Targets

### 1. Pressure Drop Calculator (`pressure_drop_calculator`)

**Module Path:** `src/pressure_drop_calculator/`

**Target SLA:** < 100 milliseconds per single calculation

**Rationale:**
Web applications require sub-second responsiveness for interactive tools. Pressure drop is a foundational calculation called frequently during parametric sweeps and sensitivity analysis.

**Test Coverage:**
- `test_pressure_drop_single_calculation` — Basic single calculation
- `test_pressure_drop_with_varying_density` — 5 calculations with density sweep
- `test_pressure_drop_with_varying_length` — 5 calculations with length sweep
- `test_pressure_drop_repeated_calls` — 100 repeated calculations (amortized SLA)

**Measurement Methodology:**
- Warm-up: 5 iterations
- Sample count: Variable (auto-calibrated by pytest-benchmark)
- Timer: `time.perf_counter` (high-resolution wall clock)
- GC disabled during benchmark runs

### 2. Rotation Converter (`rotation_converter`)

**Module Path:** `src/rotation_converter/`
**Status:** Deprecated in favor of `tools_core::math_primitives` (Rust); maintained for compatibility.

**Target SLA:** < 50 milliseconds per single conversion

**Rationale:**
Rotation conversions are used heavily in robotics calculations and quaternion-based kinematics chains. Single operations must complete in microseconds to support real-time systems and batch processing.

**Test Coverage:**

#### Basic Conversions
- `test_euler_to_quaternion_conversion` — XYZ Euler convention
- `test_quaternion_to_rotation_matrix_conversion` — SO(3) from unit quaternion
- `test_rotation_matrix_to_quaternion_conversion` — Recover quaternion from 3x3
- `test_axis_angle_to_quaternion_conversion` — Compact axis-angle form

#### Chained Conversions (Multi-step Operations)
- `test_euler_to_matrix_chain` — Euler → Quat → Matrix (2 conversions)
- `test_matrix_to_axis_angle_chain` — Matrix → Quat → Axis-Angle (2 conversions)
- `test_quaternion_multiply_sequence` — 10 sequential quaternion multiplications

#### Batch Operations (Throughput)
- `test_batch_euler_conversions` — 100 Euler sets with slight angle variations
- `test_normalize_quaternion_sequence` — 50 normalization operations

**Measurement Methodology:**
- All times in microseconds (μs)
- Warm-up: 100,000 iterations (fine-grained timer calibration)
- GC disabled; pure Python hot-path measurement
- Provides baseline for Rust port comparison

### 3. Data Processor FFT Filter Operations (`data_processor.fft_filter_ops`)

**Module Path:** `src/data_processing/data_processor/python/`

**Target SLA:** < 500 milliseconds for 10,000-row filtering pipeline

**Rationale:**
Web applications need to support real-time visualization and filtering of signals (especially from sensor data or simulation logs). Latency directly impacts UI responsiveness for data-driven dashboards.

**Test Coverage:**

#### Window Design (Core Bottleneck)
- `test_design_frequency_window_small` — 100-sample window design (~20μs)
- `test_design_frequency_window_medium` — 1,000-sample window (~37μs)
- `test_design_frequency_window_large` — 10,000-sample window (~52μs)

#### Window Application (Vectorized)
- `test_apply_window_function_small` — 100-element vector multiplication
- `test_apply_window_function_medium` — 1,000-element vector
- `test_apply_window_function_large` — 10,000-element vector

#### FFT Filtering (FFT-based convolution)
- `test_fft_filter_core_lowpass` — Low-pass on 1000-sample signal
- `test_fft_filter_core_bandpass` — Band-pass on 1000-sample signal
- `test_fft_filter_core_large_signal` — Large-scale 10K-sample filtering

#### Complete Pipelines (E2E Measurement)
- `test_window_design_and_application` — Combined design + apply on 1000 samples
- `test_complete_filter_pipeline_small` — Full 3-step pipeline on 1000 samples
- `test_complete_filter_pipeline_large` — Full pipeline on 10,000 samples
- `test_repeated_filter_operations` — 10 complete filtering cycles

**Measurement Methodology:**
- Uses NumPy FFT (potentially accelerated by platform-specific BLAS)
- Window designs tested with multiple shapes: hamming, hann, blackman
- Filter types: low-pass, high-pass, band-pass
- Time measured in microseconds; scales tested from 100 to 10,000 elements

---

## Performance Regression Detection

### CI Integration

The `.github/workflows/perf-regression.yml` workflow:

1. Runs benchmarks on every push to `main` and every PR
2. Compares against baseline metrics
3. **Fails if any single benchmark regresses > 10%**
4. Generates HTML report artifact for manual inspection

### Running Benchmarks Locally

To run benchmarks **without CI/pytest overhead:**

```bash
# All benchmarks (slow; full statistical run)
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" --benchmark-only

# Specific module only
python3 -m pytest tests/benchmarks/test_rotation_converter_perf.py --override-ini="addopts=" --benchmark-only

# Compare against baseline
python3 -m pytest tests/benchmarks/ --override-ini="addopts=" --benchmark-only \
  --benchmark-compare=.benchmarks/baseline.json \
  --benchmark-compare-fail=min:10%
```

### Baseline Snapshot

Current baseline (human-readable summary):

- **Rotation Converter:** All conversions < 100μs; batch operations ~4ms for 100 conversions
- **Pressure Drop:** Single calculation ~50-100μs; 100 repeated calls ~5-10ms
- **Data Processor:** Window design scales sub-linearly (~52μs for 10K samples); full FFT pipeline < 200μs

**Baseline JSON:** `.benchmarks/baseline.json` (pytest-benchmark JSON format)

---

## Regression Root Causes & Mitigation

### Common Causes of Regression

| Cause | Example | Mitigation |
|-------|---------|-----------|
| **Algorithm change** | Replacing O(n) sort with O(n²) | Maintain time complexity guarantees in docstrings |
| **New validation** | Added input bounds checking | Profile before/after; gate behind fast-path |
| **Dependency regression** | NumPy/SciPy update slower | Pin versions; auto-test against updated deps |
| **GC pressure** | Increased memory allocations | Use pre-allocated arrays; profile heap |
| **Cache thrashing** | Data structure growth | Monitor memory access patterns; consider cache-friendly layout |

### Mitigation Strategy

1. **Prevent:** Code reviews check for algorithmic changes
2. **Detect:** CI fails on > 10% regression
3. **Investigate:** Bisect to find commit; profile with cProfile
4. **Fix:** Either revert change or optimize hot path
5. **Document:** Add issue link to SLA change if intentional

---

## Known Constraints & Trade-offs

### Rotation Converter (Python Implementation)

- **Status:** Deprecated. Rust port (`tools_core::spatial::Quaternion`) is preferred for new code.
- **Why kept:** Backward compatibility with UpstreamDrift integration tests
- **Future:** Will remove in v2.0 after downstream migration window

### Data Processor Filter Design

- **Note:** Window design complexity depends on filter type and transition bandwidth
- **Trade-off:** Tighter transitions (lower `transition_bw`) → slower design, better frequency response
- **Typical:** `transition_bw=0.05` balances speed (~20-50μs) vs. quality

### Pressure Drop Calculation

- **Caveat:** SLA assumes standard atmospheric conditions (101325 Pa)
- **Variability:** Density and viscosity lookups may add 10-20μs overhead
- **Future:** C/Rust kernel if > 100ms requirement becomes bottleneck

---

## Escalation Path

**If SLA is violated:**

1. **PR CI failure:** Author must investigate & revert or optimize
2. **Sporadic regression (< 15%):** May be measurement noise; run benchmark 3x locally
3. **Consistent > 20% regression:** File issue with regression data; blocks merge
4. **Feature request (e.g., new SLA):** Discuss in design review; update this document

**Escalation Contact:** Performance Working Group (tracked in Issues #2413 phase tasks)

---

## Reference: pytest-benchmark Configuration

```ini
# From pyproject.toml [tool.pytest.ini_options]
markers = [
    "benchmark: mark test as benchmark (measures timing/throughput)",
    "performance: mark test for performance benchmarking",
]
```

Benchmarks run **serially** (xdist disabled) to avoid:
- Context switching overhead
- Memory pressure variation
- Thermal throttling interference

---

## Appendix: Baseline Data (Raw JSON Summary)

Baseline file: `.benchmarks/baseline.json`
Format: pytest-benchmark v5.2.3 JSON (stores min/max/mean/stddev per benchmark)

**Key fields in JSON:**
- `benchmarks[]` — Array of benchmark objects
- `name` — Full test name (e.g., "test_euler_to_quaternion_conversion")
- `stats.mean` — Mean time per iteration (in seconds)
- `stats.stddev` — Standard deviation
- `stats.rounds` — Number of measurement rounds
- `stats.iterations` — Iterations per round

To extract baseline:
```bash
python3 -c "import json; b = json.load(open('.benchmarks/baseline.json')); \
  print('\n'.join(f\"{b['name']}: {b['stats']['mean']*1e6:.1f}μs\" for b in b['benchmarks']))"
```

---

**Document Version:** 1.0
**Next Review:** After Phase 2.2 (Rust Kernel Optimization)
