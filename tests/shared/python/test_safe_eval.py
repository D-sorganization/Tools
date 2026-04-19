"""Tests for the safe_eval module.

Covers:
- Basic arithmetic evaluation
- Math function evaluation (numpy and scalar)
- Variable injection
- Security: rejects dangerous operations (imports, exec, attribute access)
- AST validation
- Convenience wrapper safe_eval_math
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
from numpy.testing import assert_allclose
from safe_eval import (
    NUMPY_MATH_NAMESPACE,
    SCALAR_MATH_NAMESPACE,
    safe_eval,
    safe_eval_math,
    validate_expression,
)

# ── Basic Arithmetic ─────────────────────────────────────────────────────


class TestBasicArithmetic:
    """Test basic arithmetic operations in safe_eval."""

    def test_addition(self) -> None:
        assert safe_eval("1 + 2", {}) == 3

    def test_subtraction(self) -> None:
        assert safe_eval("10 - 4", {}) == 6

    def test_multiplication(self) -> None:
        assert safe_eval("3 * 7", {}) == 21

    def test_division(self) -> None:
        assert_allclose(safe_eval("10 / 3", {}), 10 / 3)

    def test_integer_division(self) -> None:
        assert safe_eval("10 // 3", {}) == 3

    def test_modulo(self) -> None:
        assert safe_eval("10 % 3", {}) == 1

    def test_exponentiation(self) -> None:
        assert safe_eval("2 ** 10", {}) == 1024

    def test_negative_number(self) -> None:
        assert safe_eval("-5", {}) == -5

    def test_nested_expressions(self) -> None:
        assert safe_eval("(1 + 2) * (3 + 4)", {}) == 21

    def test_precedence(self) -> None:
        assert safe_eval("2 + 3 * 4", {}) == 14

    def test_float_literal(self) -> None:
        assert_allclose(safe_eval("3.14", {}), 3.14)

    def test_scientific_notation(self) -> None:
        assert_allclose(safe_eval("1e-3", {}), 0.001)


# ── Variable Injection ───────────────────────────────────────────────────


class TestVariableInjection:
    """Test that user-supplied variables are available in expressions."""

    def test_single_variable(self) -> None:
        assert safe_eval("x + 1", {"x": 10}) == 11

    def test_multiple_variables(self) -> None:
        ns = {"a": 3, "b": 4}
        assert safe_eval("a * b", ns) == 12

    def test_array_variable(self) -> None:
        ns = {"data": np.array([1, 2, 3])}
        ns.update(NUMPY_MATH_NAMESPACE)
        result = safe_eval("sum(data)", ns)
        assert result == 6

    def test_variable_name_shadows_disallowed(self) -> None:
        """Variables named after disallowed constructs should still work."""
        assert safe_eval("x", {"x": 42}) == 42


# ── Math Functions (Numpy) ───────────────────────────────────────────────


class TestNumpyMathNamespace:
    """Test NUMPY_MATH_NAMESPACE functions via safe_eval."""

    def test_sqrt(self) -> None:
        result = safe_eval("sqrt(4)", NUMPY_MATH_NAMESPACE)
        assert_allclose(result, 2.0)

    def test_sin_zero(self) -> None:
        result = safe_eval("sin(0)", NUMPY_MATH_NAMESPACE)
        assert_allclose(result, 0.0, atol=1e-15)

    def test_cos_zero(self) -> None:
        result = safe_eval("cos(0)", NUMPY_MATH_NAMESPACE)
        assert_allclose(result, 1.0)

    def test_exp_zero(self) -> None:
        result = safe_eval("exp(0)", NUMPY_MATH_NAMESPACE)
        assert_allclose(result, 1.0)

    def test_log_e(self) -> None:
        result = safe_eval("log(e)", NUMPY_MATH_NAMESPACE)
        assert_allclose(result, 1.0)

    def test_pi_constant(self) -> None:
        result = safe_eval("pi", NUMPY_MATH_NAMESPACE)
        assert_allclose(result, np.pi)

    def test_e_constant(self) -> None:
        result = safe_eval("e", NUMPY_MATH_NAMESPACE)
        assert_allclose(result, np.e)

    def test_mean_array(self) -> None:
        ns = dict(NUMPY_MATH_NAMESPACE)
        ns["data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_eval("mean(data)", ns)
        assert_allclose(result, 3.0)

    def test_std_array(self) -> None:
        ns = dict(NUMPY_MATH_NAMESPACE)
        ns["data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_eval("std(data)", ns)
        assert result > 0

    def test_abs_negative(self) -> None:
        result = safe_eval("abs(-42)", NUMPY_MATH_NAMESPACE)
        assert result == 42


# ── Math Functions (Scalar) ──────────────────────────────────────────────


class TestScalarMathNamespace:
    """Test SCALAR_MATH_NAMESPACE functions via safe_eval."""

    def test_sqrt(self) -> None:
        result = safe_eval("sqrt(9)", SCALAR_MATH_NAMESPACE)
        assert_allclose(result, 3.0)

    def test_sin_pi(self) -> None:
        result = safe_eval("sin(pi)", SCALAR_MATH_NAMESPACE)
        assert_allclose(result, 0.0, atol=1e-10)

    def test_log10(self) -> None:
        result = safe_eval("log10(100)", SCALAR_MATH_NAMESPACE)
        assert_allclose(result, 2.0)

    def test_pow(self) -> None:
        result = safe_eval("pow(2, 8)", SCALAR_MATH_NAMESPACE)
        assert result == 256

    def test_round(self) -> None:
        result = safe_eval("round(3.14159)", SCALAR_MATH_NAMESPACE)
        assert result == 3


# ── Security Tests ───────────────────────────────────────────────────────


class TestSecurityRejections:
    """Test that dangerous operations are rejected."""

    def test_import_rejected(self) -> None:
        """__import__ is an unknown function when allowed_names is set."""
        with pytest.raises(ValueError, match="Unknown function"):
            validate_expression("__import__('os')", allowed_names={"x"})

    def test_attribute_access_rejected(self) -> None:
        with pytest.raises(ValueError, match="Attribute"):
            validate_expression("os.system('echo hi')", allowed_names={"os"})

    def test_exec_rejected(self) -> None:
        """exec is an unknown function when allowed_names is set."""
        with pytest.raises(ValueError, match="Unknown function"):
            validate_expression("exec('x=1')", allowed_names={"x"})

    def test_lambda_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_expression("(lambda: 1)()")

    def test_list_comprehension_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsafe operation"):
            validate_expression("[x for x in range(10)]")

    def test_docstring_string_rejected(self) -> None:
        """Standalone statements are rejected (mode=eval expects expression)."""
        with pytest.raises(ValueError):
            validate_expression("'hello'; 1+1")

    def test_builtins_always_empty(self) -> None:
        """Even if user tries to reference builtins, they should fail."""
        with pytest.raises((NameError, ValueError)):
            safe_eval("print('hello')", {})

    def test_empty_expression_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_expression("")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_expression("   ")

    def test_syntax_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid syntax"):
            validate_expression("1 +")

    def test_unknown_variable_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown variable"):
            validate_expression("x + 1", allowed_names={"y"})

    def test_unknown_function_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown function"):
            validate_expression("open('file')", allowed_names={"x"})


# ── safe_eval_math Convenience Wrapper ───────────────────────────────────


class TestSafeEvalMath:
    """Test the safe_eval_math convenience function."""

    def test_simple_math(self) -> None:
        result = safe_eval_math("2 + 2")
        assert result == 4

    def test_with_variables(self) -> None:
        result = safe_eval_math("x * 2 + y", {"x": 5, "y": 3})
        assert result == 13

    def test_numpy_mode_default(self) -> None:
        """Default is use_numpy=True."""
        result = safe_eval_math("sqrt(16)")
        assert_allclose(result, 4.0)

    def test_scalar_mode(self) -> None:
        result = safe_eval_math("sqrt(25)", use_numpy=False)
        assert_allclose(result, 5.0)

    def test_trig_identity(self) -> None:
        """sin²(x) + cos²(x) = 1."""
        result = safe_eval_math("sin(pi/4)**2 + cos(pi/4)**2")
        assert_allclose(result, 1.0, atol=1e-10)

    def test_array_operations(self) -> None:
        data = np.array([10, 20, 30])
        result = safe_eval_math("mean(data)", {"data": data})
        assert_allclose(result, 20.0)

    def test_complex_formula(self) -> None:
        result = safe_eval_math("sqrt(x**2 + y**2)", {"x": 3.0, "y": 4.0})
        assert_allclose(result, 5.0)


# ── validate_expression ──────────────────────────────────────────────────


class TestValidateExpression:
    """Test the validate_expression function."""

    def test_returns_ast_for_valid(self) -> None:
        import ast

        tree = validate_expression("1 + 2")
        assert isinstance(tree, ast.Expression)

    def test_allows_comparison(self) -> None:
        tree = validate_expression("x > 5", allowed_names={"x"})
        assert tree is not None

    def test_allows_ternary(self) -> None:
        tree = validate_expression(
            "x if x > 0 else -x",
            allowed_names={"x"},
        )
        assert tree is not None

    def test_allows_subscript(self) -> None:
        tree = validate_expression(
            "data[0]",
            allowed_names={"data"},
        )
        assert tree is not None
