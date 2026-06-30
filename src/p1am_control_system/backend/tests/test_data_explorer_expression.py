"""Tests for the safe expression evaluator (data_explorer_expression)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from data_explorer_expression import (  # noqa: E402
    ExpressionError,
    evaluate_expression,
)


def _vars(**kw: object) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float) for k, v in kw.items()}


# --------------------------------------------------------------------------- #
# Arithmetic / column references                                              #
# --------------------------------------------------------------------------- #
def test_add_columns() -> None:
    out = evaluate_expression("a + b", _vars(a=[1, 2, 3], b=[4, 5, 6]))
    np.testing.assert_allclose(out, [5.0, 7.0, 9.0])


def test_full_arithmetic_chain() -> None:
    v = _vars(a=[2.0, 4.0], b=[1.0, 2.0])
    out = evaluate_expression("(a - b) * 2 + a / b", v)
    # (2-1)*2 + 2/1 = 4 ; (4-2)*2 + 4/2 = 6
    np.testing.assert_allclose(out, [4.0, 6.0])


def test_power_mod_floordiv() -> None:
    v = _vars(a=[3.0, 5.0])
    np.testing.assert_allclose(evaluate_expression("a ** 2", v), [9.0, 25.0])
    np.testing.assert_allclose(evaluate_expression("a % 2", v), [1.0, 1.0])
    np.testing.assert_allclose(evaluate_expression("a // 2", v), [1.0, 2.0])


def test_unary_negation() -> None:
    out = evaluate_expression("-a", _vars(a=[1.0, -2.0]))
    np.testing.assert_allclose(out, [-1.0, 2.0])


def test_scalar_constant_in_expr() -> None:
    out = evaluate_expression("a * 0 + 5", _vars(a=[1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, [5.0, 5.0, 5.0])


# --------------------------------------------------------------------------- #
# Functions and constants                                                     #
# --------------------------------------------------------------------------- #
def test_sqrt_and_abs() -> None:
    out = evaluate_expression("sqrt(abs(a))", _vars(a=[-4.0, 9.0]))
    np.testing.assert_allclose(out, [2.0, 3.0])


def test_trig_with_pi_constant() -> None:
    out = evaluate_expression("sin(pi * a)", _vars(a=[0.0, 0.5]))
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-12)


def test_exp_log_roundtrip() -> None:
    v = _vars(a=[1.0, 2.0, 3.0])
    np.testing.assert_allclose(evaluate_expression("log(exp(a))", v), [1, 2, 3])


def test_clip_min_max_elementwise() -> None:
    v = _vars(a=[-1.0, 0.5, 2.0])
    np.testing.assert_allclose(evaluate_expression("clip(a, 0, 1)", v), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(evaluate_expression("min(a, 0)", v), [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(evaluate_expression("max(a, 0)", v), [0.0, 0.5, 2.0])


def test_mean_reduces_then_broadcasts() -> None:
    out = evaluate_expression("mean(a)", _vars(a=[1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(out, [2.5, 2.5, 2.5, 2.5])


def test_min_reducer_single_arg() -> None:
    out = evaluate_expression("min(a)", _vars(a=[3.0, 1.0, 2.0]))
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0])


def test_e_constant() -> None:
    out = evaluate_expression("log(e)", _vars(a=[1.0, 2.0]))
    np.testing.assert_allclose(out, [1.0, 1.0])


# --------------------------------------------------------------------------- #
# Broadcasting / scalar results                                               #
# --------------------------------------------------------------------------- #
def test_scalar_result_broadcasts_to_first_var_length() -> None:
    out = evaluate_expression("pi", _vars(a=[1.0, 2.0, 3.0], b=[9.0]))
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [np.pi, np.pi, np.pi])


def test_returns_1d_float_array() -> None:
    out = evaluate_expression("a + 1", _vars(a=[1, 2]))
    assert out.dtype == np.float64
    assert out.ndim == 1


# --------------------------------------------------------------------------- #
# Division semantics                                                          #
# --------------------------------------------------------------------------- #
def test_division_by_zero_yields_inf_no_raise() -> None:
    out = evaluate_expression("a / b", _vars(a=[1.0, 0.0], b=[0.0, 0.0]))
    assert np.isinf(out[0])
    assert np.isnan(out[1])


# --------------------------------------------------------------------------- #
# DbC: type / value errors                                                    #
# --------------------------------------------------------------------------- #
def test_expr_not_str_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        evaluate_expression(123, _vars(a=[1.0]))  # type: ignore[arg-type]


def test_variables_not_mapping_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        evaluate_expression("a", [1, 2, 3])  # type: ignore[arg-type]


def test_variable_value_not_ndarray_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        evaluate_expression("a", {"a": [1.0, 2.0]})  # type: ignore[dict-item]


def test_empty_variables_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        evaluate_expression("1 + 1", {})


# --------------------------------------------------------------------------- #
# Rejection of disallowed constructs                                          #
# --------------------------------------------------------------------------- #
def test_unknown_name_rejected_and_named() -> None:
    with pytest.raises(ExpressionError) as exc:
        evaluate_expression("a + zzz", _vars(a=[1.0]))
    assert "zzz" in str(exc.value)


def test_attribute_access_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("a.real", _vars(a=[1.0]))


def test_subscript_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("a[0]", _vars(a=[1.0, 2.0]))


def test_compare_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("a > 0", _vars(a=[1.0]))


def test_boolop_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("a and b", _vars(a=[1.0], b=[1.0]))


def test_lambda_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("lambda x: x", _vars(a=[1.0]))


def test_comprehension_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("[x for x in a]", _vars(a=[1.0]))


def test_unknown_function_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("eval(a)", _vars(a=[1.0]))


def test_import_like_string_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("__import__('os')", _vars(a=[1.0]))


def test_attribute_call_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("np.sin(a)", _vars(a=[1.0]))


def test_keyword_argument_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("clip(a, a_min=0, a_max=1)", _vars(a=[1.0]))


def test_syntax_error_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("a +", _vars(a=[1.0]))


def test_string_constant_rejected() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("'hello'", _vars(a=[1.0]))
