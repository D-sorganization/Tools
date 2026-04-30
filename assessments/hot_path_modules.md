# Hot-Path Module Identification

## Overview

Hot-path modules are critical functions used frequently across the codebase or by downstream consumers (UpstreamDrift, Gasification_Model). These require ≥80% coverage as a quality gate.

**Phase 1 Status:** Identified and documented
**Phase 2 Status:** Coverage enforcement to begin

---

## Identified Hot-Path Modules

### 1. src/pressure_drop_calculator

**Priority:** P0 — Critical process engineering

**Responsibility:**
- Hydraulic pressure drop calculations
- Friction factor and flow regime determination
- Used by downstream calculators and simulations

**Current Coverage:** Not yet measured (needs dedicated tests)
**Target Coverage:** 80%
**Reason for 80% threshold:** Core mathematical engine, downstream dependency

**Key modules to test:**
- `pressure_drop_calculator/__init__.py` — Main API
- `pressure_drop_calculator/calculators/*` — Individual calculator implementations
- `pressure_drop_calculator/correlations/*` — Physics correlations

---

### 2. src/rotation_converter

**Priority:** P0 — Core robotics mathematics

**Responsibility:**
- Quaternion ↔ Euler angle conversions
- Rotation matrix operations
- Gimbal lock handling
- Used in humanoid motion capture and inverse kinematics

**Current Coverage:** 10.17% (436/4289 lines)
**Target Coverage:** 80%
**Reason for 80% threshold:** Low-level math (no refactoring tolerance), shared library

**Key modules to test:**
- `rotation_converter/_contracts.py` — Preconditions/postconditions (currently 78.95%)
- `rotation_converter/quaternion_math.py` — Quaternion operations
- `rotation_converter/euler_angles.py` — Euler angle conversions
- `rotation_converter/gimbal_lock.py` — Gimbal lock detection/avoidance

**Test Status:**
- `tests/rotation_converter/test_rotation_core.py` exists
- Partial coverage of core functions (test_gimbal_lock, test_quaternion_roundtrips)
- **Gap:** Missing comprehensive edge-case tests

---

### 3. src/shared/python/model_generation

**Priority:** P0 — URDF and robotics model generation

**Responsibility:**
- URDF XML generation from parameters
- Mesh generation and validation
- Kinematics chain construction
- Used by UpstreamDrift for robot model export

**Current Coverage:** 0.0% (needs measurement)
**Target Coverage:** 80%
**Reason for 80% threshold:** Downstream dependency (UpstreamDrift), export critical path

**Key modules to test:**
- `model_generation/__init__.py` — Main API
- `model_generation/urdf_generator.py` — URDF XML generation
- `model_generation/mesh_tools.py` — 3D geometry validation
- `model_generation/kinematics.py` — Chain construction

---

### 4. src/shared/python/upstream_drift_tools

**Priority:** P0 — Shared library (broad scope)

**Responsibility:**
- Core utilities and data structures
- Process calculator base classes
- Exception hierarchy and contracts
- Data processing pipeline

**Current Coverage:** 23.2% (225/970 lines)
**Target Coverage:** 80% (overall package)
**Reason for 80% threshold:** Foundation for all downstream code

**Key sub-modules (prioritize by downstream use):**

1. **data_processing** (P0) — ETL and feature engineering
   - Current: 0% (requires pandas/complex dependencies)
   - Target: 80%
   - Reason: Async pipeline, data integrity critical

2. **process_calculators** (P0) — Base classes for all calculators
   - Current: Not separately measured
   - Target: 80%
   - Reason: Inherited by pressure_drop, flow_rate, other calculators

3. **exceptions** (P0) — Exception hierarchy
   - Current: Included in upstream_drift_tools (23.2%)
   - Target: 100%
   - Reason: Contracts enforcement, error handling

4. **constants** (P1) — Physical constants, unit conversions
   - Current: Not separately measured
   - Target: 100%
   - Reason: Immutable data, simple validation

---

## Coverage Status Summary (2026-04-30)

| Module | Current | Target | Priority | Gap |
|--------|---------|--------|----------|-----|
| pressure_drop_calculator | Not measured | 80% | P0 | Unknown |
| rotation_converter | 10.17% | 80% | P0 | 69.83% |
| model_generation | 0.0% | 80% | P0 | 80.0% |
| upstream_drift_tools | 23.2% | 80% | P0 | 56.8% |
| notes | 49.3% | N/A (tracked: 49%) | P1 | Maintain |
| safe_eval | 100.0% | N/A | P2 | Maintain |
| contracts | 78.95% | 90% | P2 | 11.05% |

---

## Strategies for Reaching 80%

### For rotation_converter (currently 10.17% → 80%)

**Current test coverage:**
- Basic roundtrip tests (quat → euler → quat)
- Gimbal lock detection
- Some axis-angle conversions

**Gaps to fill:**
1. **Branch coverage:** Many conditional paths untested (gimbal lock logic, edge cases)
2. **Edge cases:** Zero rotations, 180-degree rotations, boundary conditions
3. **Error handling:** Invalid input validation, numerical precision limits
4. **Performance paths:** Vectorized operations (if present)

**Estimated effort:** 3-5 days of test expansion

### For model_generation (currently 0% → 80%)

**Challenges:**
1. Complex dependencies (trimesh, scipy, XML generation)
2. Visualization code (harder to test without display)
3. Integration with external tools (CAD formats)

**Recommendations:**
1. Start with URDF validation logic (testable in isolation)
2. Mock mesh generation for unit tests
3. Add integration tests with sample URDF files
4. Defer GUI visualization tests to Phase 2

**Estimated effort:** 5-7 days

### For pressure_drop_calculator (currently unmeasured → 80%)

**Strategy:**
1. Establish baseline with current tests
2. Identify uncovered calculation paths
3. Add tests for edge cases (laminar/turbulent transition, extreme Reynolds numbers)
4. Validate correlations against reference data

**Estimated effort:** 4-6 days

### For upstream_drift_tools (currently 23.2% → 80%)

**Phased approach:**
1. **Phase 2a:** Focus on exceptions + contracts (→ 50%)
2. **Phase 2b:** Add data_processing tests (→ 60%)
3. **Phase 2c:** Process calculator base class tests (→ 75%)
4. **Phase 2d:** Remaining utilities (→ 80%)

**Estimated effort:** 8-10 days

---

## Phase 2 Roadmap

### Sprint 1 (Week 1-2)
- [ ] rotation_converter: 10% → 40% (gimbal lock edge cases + axis-angle comprehensive)
- [ ] upstream_drift_tools: 23% → 35% (exception hierarchy + basic process calc tests)

### Sprint 2 (Week 3-4)
- [ ] rotation_converter: 40% → 70% (vectorized ops, numerical precision)
- [ ] pressure_drop_calculator: baseline → 50% (laminar/turbulent paths)

### Sprint 3 (Week 5-6)
- [ ] rotation_converter: 70% → 80% (remaining edge cases, branch coverage)
- [ ] model_generation: 0% → 40% (URDF validation, basic mesh ops)

### Sprint 4 (Week 7-8)
- [ ] upstream_drift_tools: 35% → 60% (data processing pipeline)
- [ ] model_generation: 40% → 70% (integration tests)

### Sprint 5 (Week 9-10)
- [ ] All hot-path modules reach 80%
- [ ] Baseline ratchet to 35% (from 6.25%)

---

## Downstream Impact Assessment

### UpstreamDrift Dependencies

**Critical modules:**
1. model_generation — URDF export (high-priority)
2. upstream_drift_tools — shared utilities (high-priority)
3. rotation_converter — motion capture (medium-priority)

**Test verification needed:**
- Sync with UpstreamDrift CI to validate contract tests pass after changes

### Gasification_Model Dependencies

**Critical modules:**
1. upstream_drift_tools — data processing (high-priority)
2. process_calculators — base classes (high-priority)

**Test verification needed:**
- Ensure thermodynamic calculator contracts unchanged

---

## Measurement and Tracking

### How to Track Progress

**Command to measure hot-path coverage:**
```bash
python3 -m pytest tests/ --cov=src/pressure_drop_calculator \
  --cov=src/rotation_converter \
  --cov=src/shared/python/model_generation \
  --cov=src/shared/python/upstream_drift_tools \
  --cov-report=term-missing --cov-fail-under=0
```

**CI Integration:**
- Current baseline tracked in `config/coverage_baseline.json`
- Per-module tracking in `assessments/coverage_baseline.json`
- Policy enforcement in `config/coverage_policy.json`

---

## References

- **Issue:** #2406 — Comprehensive test coverage measurement and ratcheting
- **Coverage baseline:** `assessments/coverage_baseline.json`
- **Policy enforcement:** `config/coverage_policy.json`
- **Setup documentation:** `COVERAGE_SETUP.md`
