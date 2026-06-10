"""Tests for shared.python.safe_eval, including DoS-complexity guards (issue #3290).

These verify that the AST-allowlist evaluator both evaluates legitimate math and
fails fast on the known denial-of-service vectors (pow bombs, repetition bombs,
oversized / over-complex expressions) that previously passed validation.
"""

from __future__ import annotations

import math

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


def test_pow_bomb_does_not_block_for_long() -> None:
    """Validation of the pow bomb returns quickly (well under a second)."""
    import time

    start = time.perf_counter()
    with pytest.raises(ValueError):
        safe_eval("9**9**9**9", {})
    assert time.perf_counter() - start < 1.0
    # sanity: a normal expression with the same functions still works
    assert safe_eval_math("cos(0)", use_numpy=False) == math.cos(0)
