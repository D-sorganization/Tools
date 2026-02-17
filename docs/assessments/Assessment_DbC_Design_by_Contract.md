# Assessment: Design by Contract Adherence

**Assessment Date:** 2026-02-02
**Grade:** 6.5/10 (Good Infrastructure, Poor Adoption)

---

## Executive Summary

This codebase demonstrates **excellent infrastructure for Design by Contract (DbC)** with fully-featured decorator libraries, comprehensive validation utilities, and a well-designed exception hierarchy. However, the infrastructure remains **largely unused in production code**, creating a significant gap between capability and practice.

### Key Findings

| Metric                             | Value                  | Assessment           |
| ---------------------------------- | ---------------------- | -------------------- |
| Contract decorator implementations | 2 complete libraries   | Excellent            |
| Production files using decorators  | 1 (documentation only) | Poor                 |
| Total raise statements             | 439                    | Good coverage        |
| ValueError usage                   | 272 (62%)              | Dominant pattern     |
| Assertion statements (non-test)    | 46                     | Appropriately sparse |
| Custom contract exceptions         | 28                     | Well-designed        |

---

## 1. Contract Infrastructure Analysis

### 1.1 Decorator Libraries

Two comprehensive contract decorator implementations exist:

#### `model_generation/core/contracts.py` (424 lines)

| Feature                   | Status      | Description                                                 |
| ------------------------- | ----------- | ----------------------------------------------------------- |
| `@precondition`           | Implemented | Validates inputs before method execution                    |
| `@postcondition`          | Implemented | Validates outputs after method execution                    |
| `@contract`               | Implemented | Combined pre/post decorator                                 |
| `@invariant`              | Implemented | Class decorator for state validation                        |
| `set_contracts_enabled()` | Implemented | Global toggle for performance                               |
| Convenience functions     | Implemented | `require_positive`, `require_finite`, `require_unit_vector` |

**Location:** `src/shared/python/model_generation/core/contracts.py:1-424`

#### `humanoid_character_builder/contracts.py` (191 lines)

| Feature                  | Status      | Description                                    |
| ------------------------ | ----------- | ---------------------------------------------- |
| `@precondition`          | Implemented | With argument binding support                  |
| `@postcondition`         | Implemented | Result validation                              |
| `@invariant`             | Implemented | Checks after `__init__` and all public methods |
| `ContractViolationError` | Implemented | Extends `AssertionError`                       |

**Location:** `src/shared/python/humanoid_character_builder/contracts.py:1-191`

### 1.2 Exception Hierarchy

The codebase has a well-structured exception hierarchy for contract violations:

```
Exception
├── ContractViolation (dataclass)
│   ├── PreconditionError
│   ├── PostconditionError
│   └── InvariantError
└── ContractViolationError (AssertionError)
```

**Strengths:**

- Semantic exception types distinguish contract types
- Detailed error messages include function names and arguments
- Dataclass-based exceptions provide structured error details

---

## 2. Precondition Analysis

### 2.1 Current Implementation Patterns

**Pattern A: Tuple Return Validation** (Preferred in validation.py)

```python
# src/python/src/utils/validation.py:15-55
def validate_path(
    path: Path | str,
    must_exist: bool = True,
    must_be_within: Path | str | None = None,
) -> tuple[bool, str | None]:
    """Returns (is_valid, error_message)"""
```

**Pattern B: Exception-Based Validation** (Used in security)

```python
# src/data_processing/data_processor/python/data_processor/security_utils.py
def validate_python_expression(expr: str, allowed_names: set[str] | None = None) -> None:
    # Raises ExpressionValidationError on violation
```

**Pattern C: Contract Decorators** (Available but unused)

```python
# Available in contracts.py but not deployed
@precondition(lambda x: x > 0, "x must be positive")
def sqrt(x: float) -> float:
    return x ** 0.5
```

### 2.2 Precondition Coverage by Module

| Module              | Preconditions | Pattern       | Quality |
| ------------------- | ------------- | ------------- | ------- |
| `security_utils.py` | Strong        | Exceptions    | 9/10    |
| `validation.py`     | Strong        | Tuple returns | 9/10    |
| `file_utils.py`     | Moderate      | Mixed         | 7/10    |
| `base_builder.py`   | Weak          | None          | 4/10    |
| `types.py`          | Weak          | None          | 4/10    |

### 2.3 Missing Preconditions

**`base_builder.py:101-121`** - No input validation:

```python
def __init__(self, robot_name: str = "robot"):
    # Missing: validate robot_name is non-empty, valid XML name
    self._robot_name = robot_name

@robot_name.setter
def robot_name(self, name: str) -> None:
    # Missing: validate name is non-empty, valid XML identifier
    self._robot_name = name
```

**`types.py` from_dict methods** - No validation on dictionary structure or value ranges.

---

## 3. Postcondition Analysis

### 3.1 Current Implementation

Postconditions are primarily implemented through `ValidationResult` objects:

```python
# src/shared/python/model_generation/core/validation.py
@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
```

**Issue:** Postcondition checking is **optional** - callers can ignore `ValidationResult.is_valid`:

```python
result = builder.build()
# Caller may proceed without checking result.validation.is_valid
```

### 3.2 Postcondition Decorator Usage

The `@postcondition` decorator exists but is **only used in tests**:

```python
# src/shared/python/model_generation/tests/test_contracts.py:61-80
@postcondition(lambda result: result >= 0, "result must be non-negative")
def abs_value(x: float) -> float:
    return abs(x)
```

No production code uses `@postcondition`.

---

## 4. Invariant Analysis

### 4.1 Infrastructure

Both contract libraries implement the `@invariant` class decorator that:

- Checks condition after `__init__`
- Checks condition after every public method call
- Provides detailed error messages

### 4.2 Adoption Status

| Class              | Should Have Invariant      | Has Invariant |
| ------------------ | -------------------------- | ------------- |
| `URDFModel`        | Yes (links > 0)            | No            |
| `BaseURDFBuilder`  | Yes (valid state)          | No            |
| `BuildResult`      | Yes (success ⟺ urdf_xml)   | No            |
| `ValidationResult` | Yes (is_valid ⟺ no errors) | No            |

**Example Missing Invariant:**

```python
# base_builder.py should use:
@invariant(
    lambda self: len(self._links) == 0 or self._has_valid_tree(),
    "Model must maintain valid link tree structure"
)
class BaseURDFBuilder(ABC):
    ...
```

---

## 5. Validation Utilities Inventory

### 5.1 Centralized Validators

| File                                  | Functions       | Purpose                                      |
| ------------------------------------- | --------------- | -------------------------------------------- |
| `src/python/src/utils/validation.py`  | 6               | Path, extension, version, null, empty, range |
| `model_generation/core/validation.py` | Validator class | Model structure validation                   |
| `security_utils.py`                   | 3               | Security-focused validation                  |

### 5.2 Scattered Validation

Validation logic is also scattered across:

- `file_utils.py` - Format-specific checks
- `conversion/service.py` - Unit validation
- Various `__init__.py` files - Import checks

**Recommendation:** Consolidate into a single `contracts/` or `validation/` package.

---

## 6. Defensive Programming Patterns

### 6.1 Strengths

**Error Handling Decorators** (`error_handling.py`):

```python
@handle_file_errors(default=None, log_error=True, reraise=False)
def read_config() -> Config | None:
    # Safe file operations
```

**Import Guards** (throughout codebase):

```python
try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
```

**Security Validation** (AST-based expression checking):

```python
def validate_python_expression(expr: str, allowed_names: set[str] | None = None):
    tree = ast.parse(expr, mode="eval")
    # Validates against allowed operations
```

### 6.2 Weaknesses

- No systematic null checking on function entry
- Missing bounds validation on numeric inputs
- Inconsistent handling of edge cases (empty lists, zero values)

---

## 7. Test Coverage for Contracts

### 7.1 Test Quality

Contract decorators have **excellent test coverage**:

| Test File                                            | Lines | Coverage           |
| ---------------------------------------------------- | ----- | ------------------ |
| `model_generation/tests/test_contracts.py`           | 227   | 100% of decorators |
| `humanoid_character_builder/tests/test_contracts.py` | 99    | 100% of decorators |

### 7.2 Test Patterns

Tests verify:

- Precondition violations raise `PreconditionError`
- Postcondition violations raise `PostconditionError`
- Invariant violations raise `InvariantError`
- Global disable toggle works correctly
- Multiple arguments are properly bound

---

## 8. Recommendations

### Phase 1: Immediate (High Impact, Low Effort)

#### 8.1 Deploy Contract Decorators on Critical APIs

**Target files:**

- `src/shared/python/model_generation/builders/base_builder.py`
- `src/shared/python/humanoid_character_builder/interfaces/api.py`
- `src/shared/python/model_generation/core/types.py`

**Example implementation:**

```python
from model_generation.core.contracts import precondition, require_positive

class BaseURDFBuilder(ABC):
    @precondition(
        lambda self, robot_name: robot_name and robot_name.strip(),
        "robot_name must be non-empty"
    )
    def __init__(self, robot_name: str = "robot"):
        self._robot_name = robot_name
```

#### 8.2 Create DbC Style Guide

Create `docs/development/DESIGN_BY_CONTRACT.md` covering:

- When to use assertions vs. exceptions
- Precondition vs. postcondition patterns
- Invariant-checking class design
- Performance considerations (`set_contracts_enabled`)

### Phase 2: Short-term (Medium Effort)

#### 8.3 Standardize Validation Pattern

**Current inconsistency:**

- Some validators return `tuple[bool, str | None]`
- Others raise exceptions immediately
- Others return `ValidationResult`

**Recommendation:**

- Preconditions: Raise exceptions (fail-fast)
- Postconditions: Return `ValidationResult` (caller decides severity)
- Invariants: Use decorator infrastructure

#### 8.4 Add Invariants to Core Classes

| Class              | Proposed Invariant                                      |
| ------------------ | ------------------------------------------------------- |
| `BaseURDFBuilder`  | `len(self._links) >= 0 and all links have unique names` |
| `BuildResult`      | `self.success == (self.urdf_xml is not None)`           |
| `ValidationResult` | `self.is_valid == (len(self.errors) == 0)`              |

### Phase 3: Long-term (Ecosystem Improvement)

#### 8.5 Unified Validation Package

Create `src/shared/python/contracts/`:

```
contracts/
├── __init__.py
├── decorators.py     # @precondition, @postcondition, @invariant
├── validators.py     # Reusable validation functions
├── exceptions.py     # Contract exception hierarchy
└── result.py         # ValidationResult and related
```

#### 8.6 CI Integration

- Run tests with `CONTRACTS_ENABLED=True`
- Add contract coverage metrics to CI reports
- Consider property-based testing with Hypothesis

---

## 9. Maturity Assessment

### 9.1 DbC Maturity Model

| Level | Description             | Current Status |
| ----- | ----------------------- | -------------- |
| 1     | Ad-hoc validation       | Surpassed      |
| 2     | Validation utilities    | Achieved       |
| 3     | Contract infrastructure | Achieved       |
| **4** | **Systematic adoption** | **Gap**        |
| 5     | Formal verification     | Future goal    |

### 9.2 Component Scores

| Component      | Score      | Notes                            |
| -------------- | ---------- | -------------------------------- |
| Infrastructure | 9/10       | Two complete decorator libraries |
| Documentation  | 5/10       | Mentions DbC but no guide        |
| Adoption       | 3/10       | Only tests use decorators        |
| Consistency    | 5/10       | Multiple validation patterns     |
| Test Coverage  | 9/10       | Thorough contract tests          |
| **Overall**    | **6.5/10** | Good foundation, poor adoption   |

---

## 10. Files Requiring Attention

### Priority 1 (High-Risk, No Contracts)

| File                       | Risk           | Action                       |
| -------------------------- | -------------- | ---------------------------- |
| `builders/base_builder.py` | Public API     | Add preconditions to setters |
| `core/types.py`            | Data integrity | Validate from_dict methods   |
| `interfaces/api.py`        | External input | Validate BodyParameters      |

### Priority 2 (Important, Inconsistent)

| File                    | Issue           | Action                      |
| ----------------------- | --------------- | --------------------------- |
| `validation.py`         | Tuple returns   | Consider exception-based    |
| `file_utils.py`         | Silent failures | Add explicit error handling |
| `conversion/service.py` | Warnings only   | Add strict mode             |

### Priority 3 (Enhancement)

| File                    | Opportunity                      |
| ----------------------- | -------------------------------- |
| `physics_validation.py` | Add @postcondition to validators |
| `Validator` class       | Add @invariant for state         |
| All builders            | Use @invariant for valid state   |

---

## Conclusion

The Tools repository has **invested significantly in DbC infrastructure** with two complete decorator libraries, a well-designed exception hierarchy, and comprehensive validation utilities. However, this infrastructure is **not being deployed in production code**.

**Key Takeaway:** The patterns are defined; they need to be deployed. Adopting the existing contract decorators on public APIs would significantly improve code reliability with minimal refactoring required.

**Recommended Next Steps:**

1. Add `@precondition` to 3 high-risk public APIs (2 hours)
2. Create DbC style guide document (1 hour)
3. Add `@invariant` to 2 core classes (2 hours)
4. Update PR checklist to include contract review (30 minutes)

---

_Assessment conducted following Pragmatic Programmer principles for Design by Contract evaluation._
