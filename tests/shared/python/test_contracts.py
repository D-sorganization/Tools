"""Tests for contracts module — Design by Contract enforcement.

Covers:
- ContractLevel enum and resolution
- Core primitives: require, ensure, invariant
- Precondition / postcondition decorators
- Contract combinator decorator
- Class invariant decorator and ContractChecker mixin
- Domain helpers: check_positive, check_non_negative, check_range, etc.
- Backward-compat set_contracts_enabled
"""

from __future__ import annotations

from typing import Any

import pytest
from contracts import (
    ContractChecker,
    ContractLevel,
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
    check_non_negative,
    check_positive,
    check_pressure,
    check_range,
    check_temperature,
    class_invariant,
    contract,
    ensure,
    get_contract_level,
    invariant,
    invariant_checked,
    postcondition,
    precondition,
    require,
    set_contract_level,
    set_contracts_enabled,
)

# ── Contract Level ────────────────────────────────────────────────────────


class TestContractLevel:
    """Test ContractLevel enum and resolution."""

    def test_enum_values(self) -> None:
        assert ContractLevel.OFF.value == "off"
        assert ContractLevel.WARN.value == "warn"
        assert ContractLevel.ENFORCE.value == "enforce"

    def test_set_and_get_level(self) -> None:
        original = get_contract_level()
        try:
            set_contract_level(ContractLevel.OFF)
            assert get_contract_level() == ContractLevel.OFF

            set_contract_level(ContractLevel.WARN)
            assert get_contract_level() == ContractLevel.WARN

            set_contract_level(ContractLevel.ENFORCE)
            assert get_contract_level() == ContractLevel.ENFORCE
        finally:
            set_contract_level(original)

    def test_set_contracts_enabled_true(self) -> None:
        original = get_contract_level()
        try:
            set_contracts_enabled(True)
            assert get_contract_level() == ContractLevel.ENFORCE
        finally:
            set_contract_level(original)

    def test_set_contracts_enabled_false(self) -> None:
        original = get_contract_level()
        try:
            set_contracts_enabled(False)
            assert get_contract_level() == ContractLevel.OFF
        finally:
            set_contract_level(original)


# ── Exception Hierarchy ──────────────────────────────────────────────────


class TestContractExceptions:
    """Test the exception hierarchy."""

    def test_precondition_error(self) -> None:
        err = PreconditionError("must be positive", value=-1)
        assert "pre-condition" in str(err)
        assert "must be positive" in str(err)
        assert err.value == -1

    def test_postcondition_error(self) -> None:
        err = PostconditionError("result must be non-negative")
        assert "post-condition" in str(err)
        assert err.value is None

    def test_invariant_error(self) -> None:
        err = InvariantError("count must be >= 0", value=-5)
        assert "invariant" in str(err)
        assert err.value == -5

    def test_hierarchy(self) -> None:
        assert issubclass(PreconditionError, ContractViolationError)
        assert issubclass(PostconditionError, ContractViolationError)
        assert issubclass(InvariantError, ContractViolationError)


# ── Core Primitives ──────────────────────────────────────────────────────


class TestCorePrimitives:
    """Test require, ensure, invariant functions."""

    def setup_method(self) -> None:
        self._orig = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._orig)

    def test_require_passes(self) -> None:
        # Should not raise
        require(True, "ok")

    def test_require_fails(self) -> None:
        with pytest.raises(PreconditionError, match="must be positive"):
            require(False, "must be positive", value=-1)

    def test_ensure_passes(self) -> None:
        ensure(True, "ok")

    def test_ensure_fails(self) -> None:
        with pytest.raises(PostconditionError):
            ensure(False, "result too large")

    def test_invariant_passes(self) -> None:
        invariant(True, "ok")

    def test_invariant_fails(self) -> None:
        with pytest.raises(InvariantError):
            invariant(False, "broken invariant")

    def test_warn_mode_does_not_raise(self) -> None:
        set_contract_level(ContractLevel.WARN)
        # Should log warning but not raise
        require(False, "only a warning")
        ensure(False, "only a warning")
        invariant(False, "only a warning")

    def test_off_mode_does_nothing(self) -> None:
        set_contract_level(ContractLevel.OFF)
        require(False, "disabled")
        ensure(False, "disabled")
        invariant(False, "disabled")


# ── Decorator-Based Contracts ────────────────────────────────────────────


class TestPreconditionDecorator:
    """Test the @precondition decorator."""

    def setup_method(self) -> None:
        self._orig = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._orig)

    def test_passes_when_condition_met(self) -> None:
        @precondition(lambda x: x > 0, message="x must be positive")
        def sqrt(x: float) -> float:
            return x**0.5

        assert sqrt(4.0) == pytest.approx(2.0)

    def test_fails_when_condition_violated(self) -> None:
        @precondition(lambda x: x > 0, message="x must be positive")
        def sqrt(x: float) -> float:
            return x**0.5

        with pytest.raises(PreconditionError):
            sqrt(-1.0)


class TestPostconditionDecorator:
    """Test the @postcondition decorator."""

    def setup_method(self) -> None:
        self._orig = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._orig)

    def test_passes_when_condition_met(self) -> None:
        @postcondition(
            lambda result: result >= 0, message="result must be non-negative"
        )
        def abs_val(x: float) -> float:
            return abs(x)

        assert abs_val(-5.0) == 5.0

    def test_fails_when_condition_violated(self) -> None:
        @postcondition(lambda result: result > 0, message="must be positive")
        def negate(x: float) -> float:
            return -x

        with pytest.raises(PostconditionError):
            negate(5.0)


class TestContractDecorator:
    """Test the @contract combined decorator."""

    def setup_method(self) -> None:
        self._orig = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._orig)

    def test_both_conditions_pass(self) -> None:
        @contract(
            pre=lambda x: x >= 0,
            post=lambda result: result >= 0,
            pre_msg="x must be non-negative",
            post_msg="result must be non-negative",
        )
        def sqrt(x: float) -> float:
            return x**0.5

        assert sqrt(9.0) == pytest.approx(3.0)

    def test_pre_fails(self) -> None:
        @contract(pre=lambda x: x >= 0, pre_msg="non-negative")
        def sqrt(x: float) -> float:
            return x**0.5

        with pytest.raises(PreconditionError):
            sqrt(-1.0)

    def test_post_fails(self) -> None:
        @contract(post=lambda r: r > 0, post_msg="positive result")
        def zero() -> int:
            return 0

        with pytest.raises(PostconditionError):
            zero()


# ── Class Invariant ──────────────────────────────────────────────────────


class TestClassInvariant:
    """Test the @class_invariant decorator."""

    def setup_method(self) -> None:
        self._orig = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._orig)

    def test_invariant_holds_after_init(self) -> None:
        @class_invariant(
            lambda self: self.count >= 0,
            message="count must be non-negative",
        )
        class Counter:
            def __init__(self) -> None:
                self.count = 0

            def increment(self) -> None:
                self.count += 1

        c = Counter()
        assert c.count == 0
        c.increment()
        assert c.count == 1

    def test_invariant_violated_after_method(self) -> None:
        @class_invariant(
            lambda self: self.count >= 0,
            message="count must be non-negative",
        )
        class Counter:
            def __init__(self) -> None:
                self.count = 0

            def force_negative(self) -> None:
                self.count = -1

        c = Counter()
        with pytest.raises(InvariantError):
            c.force_negative()


# ── ContractChecker Mixin ────────────────────────────────────────────────


class TestContractCheckerMixin:
    """Test the ContractChecker mixin class."""

    def setup_method(self) -> None:
        self._orig = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._orig)

    def test_verify_invariants_passes(self) -> None:
        class MyClass(ContractChecker):
            def __init__(self) -> None:
                self.value = 10

            def _get_invariants(self) -> list[tuple[Any, str]]:
                return [(lambda: self.value > 0, "value must be positive")]

        obj = MyClass()
        obj.verify_invariants()

    def test_verify_invariants_fails(self) -> None:
        class MyClass(ContractChecker):
            def __init__(self) -> None:
                self.value = -1

            def _get_invariants(self) -> list[tuple[Any, str]]:
                return [(lambda: self.value > 0, "value must be positive")]

        obj = MyClass()
        with pytest.raises(InvariantError):
            obj.verify_invariants()

    def test_invariant_checked_decorator(self) -> None:
        class MyClass(ContractChecker):
            def __init__(self) -> None:
                self.value = 10

            def _get_invariants(self) -> list[tuple[Any, str]]:
                return [(lambda: self.value > 0, "must be positive")]

            @invariant_checked
            def set_value(self, v: int) -> None:
                self.value = v

        obj = MyClass()
        obj.set_value(5)  # Should pass

        with pytest.raises(InvariantError):
            obj.set_value(-1)  # Should violate invariant


# ── Domain Helpers ───────────────────────────────────────────────────────


class TestDomainHelpers:
    """Test domain-specific validation helpers."""

    def setup_method(self) -> None:
        self._orig = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)

    def teardown_method(self) -> None:
        set_contract_level(self._orig)

    def test_check_positive_passes(self) -> None:
        check_positive(1.0, "x")

    def test_check_positive_fails_zero(self) -> None:
        with pytest.raises(PreconditionError):
            check_positive(0.0, "x")

    def test_check_positive_fails_negative(self) -> None:
        with pytest.raises(PreconditionError):
            check_positive(-1.0, "x")

    def test_check_non_negative_passes(self) -> None:
        check_non_negative(0.0, "x")
        check_non_negative(1.0, "x")

    def test_check_non_negative_fails(self) -> None:
        with pytest.raises(PreconditionError):
            check_non_negative(-0.01, "x")

    def test_check_range_passes(self) -> None:
        check_range(5.0, 0.0, 10.0, "x")

    def test_check_range_at_bounds(self) -> None:
        check_range(0.0, 0.0, 10.0, "x")
        check_range(10.0, 0.0, 10.0, "x")

    def test_check_range_fails(self) -> None:
        with pytest.raises(PreconditionError):
            check_range(11.0, 0.0, 10.0, "x")

    def test_check_temperature_passes(self) -> None:
        check_temperature(300.0)

    def test_check_temperature_fails(self) -> None:
        with pytest.raises(PreconditionError):
            check_temperature(-10.0)

    def test_check_pressure_passes(self) -> None:
        check_pressure(101325.0)

    def test_check_pressure_fails(self) -> None:
        with pytest.raises(PreconditionError):
            check_pressure(-1.0)
