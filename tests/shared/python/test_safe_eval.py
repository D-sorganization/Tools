"""Tests for shared.python.safe_eval, including DoS-complexity guards (issue #3290).

These verify that the AST-allowlist evaluator both evaluates legitimate math and
fails fast on the known denial-of-service vectors (pow bombs, repetition bombs,
oversized / over-complex expressions) that previously passed validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.safe_eval import (
    MAX_AST_NODES,
    MAX_EXPRESSION_LENGTH,
    MAX_POW_EXPONENT,
    safe_eval,
    safe_eval_math,
    validate_expression,
)

# --------------------------- happy path -----------------------------------


def test_basic_arithmetic() -> None:
    assert safe_eval("2 + 3 * 4", {}) == 14


def test_empty_and_invalid_syntax_rejected() -> None:
    with pytest.raises(ValueError, match="Expression must not be empty"):
        validate_expression("  ")
    with pytest.raises(ValueError, match="Invalid syntax"):
        validate_expression("2 * +")


def test_namespace_variables() -> None:
    assert safe_eval("a * b + 1", {"a": 3, "b": 4}) == 13


def test_safe_eval_math_functions() -> None:
    result = safe_eval_math("sqrt(x) + 1", {"x": 4.0}, use_numpy=False)
    assert result == pytest.approx(3.0)


def test_reasonable_power_allowed() -> None:
    assert safe_eval("2 ** 10", {}) == 1024


def test_modest_pow_chain_allowed() -> None:
    # depth 2 is allowed: 2 ** 3 ** 2 == 2 ** 9 == 512
    assert safe_eval("2 ** 3 ** 2", {}) == 512


# --------------------------- DoS guards (#3290) ----------------------------


def test_pow_bomb_rejected() -> None:
    """The canonical bignum exponentiation bomb must fail fast, not hang."""
    with pytest.raises(ValueError):
        validate_expression("9**9**9**9")
    with pytest.raises(ValueError):
        safe_eval("9**9**9**9", {})


def test_large_constant_exponent_rejected() -> None:
    with pytest.raises(ValueError, match="Exponent too large"):
        validate_expression(f"2 ** {MAX_POW_EXPONENT + 1}")


def test_computed_constant_exponent_rejected() -> None:
    with pytest.raises(ValueError, match="Exponent too large"):
        validate_expression(f"2 ** ({MAX_POW_EXPONENT} + 1)")


def test_pow_call_large_exponent_rejected() -> None:
    with pytest.raises(ValueError, match="Exponent too large"):
        validate_expression(f"pow(2, {MAX_POW_EXPONENT + 1})", {"pow"})


def test_pow_call_computed_exponent_rejected() -> None:
    with pytest.raises(ValueError, match="Exponent too large"):
        safe_eval_math(f"pow(2, {MAX_POW_EXPONENT} + 1)")


def test_np_power_alias_large_exponent_rejected() -> None:
    with pytest.raises(ValueError, match="Exponent too large"):
        validate_expression(f"np_power(2, {MAX_POW_EXPONENT + 1})", {"np_power"})


def test_deep_pow_chain_rejected() -> None:
    with pytest.raises(ValueError, match="nested too deeply"):
        validate_expression("2 ** 2 ** 2 ** 2")


def test_string_constant_repetition_bomb_rejected() -> None:
    """String constants (repetition-bomb material) over the bound are rejected."""
    big_literal = '"' + "x" * 1000 + '"'
    with pytest.raises(ValueError, match="String/bytes constant"):
        validate_expression(big_literal)


def test_expression_length_limit() -> None:
    too_long = "1+" * MAX_EXPRESSION_LENGTH + "1"
    with pytest.raises(ValueError, match="too long"):
        validate_expression(too_long)


def test_node_count_limit() -> None:
    # Build a long flat sum that blows the node budget. Each "+1" adds AST nodes.
    expr = "1" + "+1" * MAX_AST_NODES
    with pytest.raises(ValueError):
        validate_expression(expr)


def test_attribute_access_rejected() -> None:
    with pytest.raises(ValueError):
        validate_expression("(1).__class__")


def test_lambda_rejected() -> None:
    with pytest.raises(ValueError):
        validate_expression("lambda: 1")


def test_unknown_name_rejected_with_allowlist() -> None:
    with pytest.raises(ValueError):
        validate_expression("os", allowed_names={"x"})


def test_unknown_function_and_attribute_calls_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown function: cos"):
        validate_expression("cos(x)", {"sin", "x"})
    with pytest.raises(ValueError, match="Attribute-based function calls"):
        validate_expression("math.sin(x)", {"math", "x"})


def test_keyword_unpacking_in_call_rejected_but_keywords_allowed() -> None:
    with pytest.raises(ValueError, match="Keyword unpacking"):
        validate_expression("func(**kwargs)", {"func", "kwargs"})

    assert safe_eval("func(x=2)", {"func": lambda *, x: x}) == 2


def test_non_string_expression_raises_contract_error() -> None:
    with pytest.raises(TypeError, match="expression must be a string"):
        validate_expression(1)  # type: ignore[arg-type]


def test_numpy_two_argument_min_max_are_elementwise() -> None:
    x = np.array([1, 4, 2])
    y = np.array([3, 2, 5])

    np.testing.assert_array_equal(
        safe_eval_math("min(x, y)", {"x": x, "y": y}), [1, 2, 2]
    )
    np.testing.assert_array_equal(
        safe_eval_math("max(x, y)", {"x": x, "y": y}), [3, 4, 5]
    )
    assert safe_eval_math("min(4, 2)") == 2
    assert safe_eval_math("max(4, 2)") == 4


def test_numpy_min_max_single_argument_and_arity_contracts() -> None:
    values = np.array([3, 1, 2])

    assert safe_eval_math("min(values)", {"values": values}) == 1
    assert safe_eval_math("max(values)", {"values": values}) == 3
    with pytest.raises(TypeError, match="min expected at least 1 argument"):
        safe_eval_math("min()")
    with pytest.raises(TypeError, match="max expected at least 1 argument"):
        safe_eval_math("max()")


def test_runtime_power_wrappers_enforce_exponent_contracts() -> None:
    assert safe_eval_math("power(2, 3)") == 8
    assert safe_eval_math("pow(2, 3)", use_numpy=False) == 8
    assert safe_eval_math("pow(2, 3, 5)", use_numpy=False) == 3

    with pytest.raises(ValueError, match="Exponent too large"):
        safe_eval_math("power(2, exponent)", {"exponent": MAX_POW_EXPONENT + 1})
    with pytest.raises(ValueError, match="Invalid exponent"):
        safe_eval_math("power(2, exponent)", {"exponent": np.inf})
    with pytest.raises(ValueError, match="Exponent must be numeric"):
        safe_eval_math("power(2, exponent)", {"exponent": "not numeric"})


def test_power_validation_allows_runtime_exponents_and_incomplete_pow_calls() -> None:
    validate_expression("2 ** x", {"x"})
    validate_expression("pow(2)", {"pow"})
    validate_expression("pow(2, x)", {"pow", "x"})


def test_power_validation_only_rejects_large_positive_integer_exponents() -> None:
    assert safe_eval("2 ** -5000", {}) == 0.0
    validate_expression(f"2 ** {float(MAX_POW_EXPONENT + 1)}")
    validate_expression(f"2 ** -{MAX_POW_EXPONENT + 1}")

    assert safe_eval_math(
        "power(2.0, exponent)",
        {"exponent": -(MAX_POW_EXPONENT + 1)},
    ) == pytest.approx(2.0 ** -(MAX_POW_EXPONENT + 1))
    safe_eval_math("power(2.0, exponent)", {"exponent": MAX_POW_EXPONENT + 0.5})


def test_safe_eval_defaults_allowed_names_to_namespace() -> None:
    assert safe_eval("x + 1", {"x": 2}) == 3


def test_constant_exponent_helper_edge_cases() -> None:
    validate_expression("'short string literal'")
    assert safe_eval("2 ** +3", {}) == 8
    assert safe_eval("2 ** -3", {}) == 0.125
    validate_expression("2 ** +'literal'")
    validate_expression("2 ** ~3")
    validate_expression("2 ** (x ** 2)", {"x"})
    validate_expression("2 ** (1 << 2)")
    validate_expression("2 ** (x + 1)", {"x"})
    validate_expression("2 ** (1 // 0)")

    with pytest.raises(ValueError, match="Invalid exponent"):
        validate_expression("2 ** 1e309")


def test_pow_bomb_does_not_block_for_long() -> None:
    """Validation of the pow bomb returns quickly (well under a second)."""
    import time

    start = time.perf_counter()
    with pytest.raises(ValueError):
        safe_eval("9**9**9**9", {})
    assert time.perf_counter() - start < 1.0
    # sanity: a normal expression with the same functions still works
    assert safe_eval_math("cos(0)", use_numpy=False) == math.cos(0)
