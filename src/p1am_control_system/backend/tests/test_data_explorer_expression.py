"""Tests for the safe expression evaluator (data_explorer_expression)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from data_explorer_expression import (  # noqa: E402
    MAX_AST_NODES,
    MAX_EXPRESSION_LENGTH,
    MAX_NESTING_DEPTH,
    MAX_POW_CHAIN_DEPTH,
    MAX_POW_EXPONENT,
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


# --------------------------------------------------------------------------- #
# Regression: review findings (arity, in-place mutation, overflow)            #
# --------------------------------------------------------------------------- #
def test_min_three_args_rejected_not_mutating() -> None:
    # A 3-arg min() must be rejected, never dispatched to np.minimum(out=...).
    orig = np.asarray([5.0, 5.0, 5.0])
    variables = {"x": orig, "y": np.asarray([1.0, 2.0, 9.0])}
    with pytest.raises(ExpressionError):
        evaluate_expression("min(x, y, x)", variables)


def test_expression_never_mutates_caller_columns() -> None:
    orig = np.asarray([3.0, 4.0, 5.0])
    before = orig.copy()
    evaluate_expression("min(x, 1) + max(x, 9)", {"x": orig})
    np.testing.assert_array_equal(orig, before)


def test_min_two_args_elementwise() -> None:
    out = evaluate_expression("min(a, b)", _vars(a=[1, 5, 3], b=[4, 2, 6]))
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_min_one_arg_reduces() -> None:
    out = evaluate_expression("min(a)", _vars(a=[3, 1, 2]))
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0])  # scalar broadcast


def test_clip_requires_three_args() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("clip(a, 0)", _vars(a=[1, 2, 3]))


def test_mean_requires_one_arg() -> None:
    with pytest.raises(ExpressionError):
        evaluate_expression("mean(a, b)", _vars(a=[1, 2], b=[3, 4]))


def test_deep_power_chain_is_rejected_as_a_bomb() -> None:
    """A long ``**`` chain is refused statically (#3986 / #3290 limits).

    This assertion replaces an earlier one that expected ``inf``. The original
    intent -- "must NOT raise a bare OverflowError" -- is preserved: the caller
    still sees ``ExpressionError``, the module's clean client error, never a raw
    arithmetic exception. The change is that the chain is now refused *before*
    any numpy work happens, matching the shared evaluator's policy.
    """
    with pytest.raises(ExpressionError, match="power chain"):
        evaluate_expression("2.0 ** (2.0 ** 2.0 ** 2.0 ** 100)", _vars(a=[1.0]))


def test_within_limit_power_chain_still_yields_inf_not_overflow() -> None:
    """A chain at the permitted depth keeps the documented inf behaviour."""
    assert MAX_POW_CHAIN_DEPTH >= 2
    out = evaluate_expression("2.0 ** 2.0 ** 999.0", _vars(a=[1.0]))
    assert np.isinf(out).all()


def test_scalar_divide_by_zero_is_inf_not_exception() -> None:
    out = evaluate_expression("a / 0", _vars(a=[1.0, 2.0]))
    assert np.isinf(out).all()


# --------------------------------------------------------------------------- #
# Complexity / DoS limits (issue #3986; limits shared from #3290)             #
# --------------------------------------------------------------------------- #
def test_limits_are_the_shared_ones_not_a_local_copy() -> None:
    """The limits must be imported from the shared evaluator, not re-declared.

    This is the DRY half of #3986: the p1am evaluator previously had *no*
    complexity limits while ``shared.python.safe_eval`` had four. Asserting
    identity (not just equality) means a future tightening of the shared limits
    cannot silently leave this HTTP-facing endpoint behind.
    """
    from shared.python import safe_eval as shared_safe_eval

    assert MAX_EXPRESSION_LENGTH is shared_safe_eval.MAX_EXPRESSION_LENGTH
    assert MAX_AST_NODES is shared_safe_eval.MAX_AST_NODES
    assert MAX_POW_EXPONENT is shared_safe_eval.MAX_POW_EXPONENT
    assert MAX_POW_CHAIN_DEPTH is shared_safe_eval.MAX_POW_CHAIN_DEPTH


def test_over_long_expression_is_rejected() -> None:
    """Rejected before ``ast.parse``: parsing the string is itself the DoS."""
    expr = "a" + " + a" * MAX_EXPRESSION_LENGTH
    assert len(expr) > MAX_EXPRESSION_LENGTH
    with pytest.raises(ExpressionError, match="too long"):
        evaluate_expression(expr, _vars(a=[1.0]))


def test_expression_at_the_length_limit_is_not_rejected_for_length() -> None:
    """The length guard must be a ``>`` boundary, not ``>=``."""
    expr = "a" * MAX_EXPRESSION_LENGTH
    # Rejected for being an unknown name, i.e. it got past the length gate.
    with pytest.raises(ExpressionError, match="unknown name"):
        evaluate_expression(expr, _vars(a=[1.0]))


def test_too_many_ast_nodes_is_rejected() -> None:
    """A wide (not deep) expression is capped by the node count."""
    expr = " + ".join(["a"] * (MAX_AST_NODES + 10))
    with pytest.raises(ExpressionError, match="too complex"):
        evaluate_expression(expr, _vars(a=[1.0]))


def test_deep_nesting_raises_expression_error_not_recursion_error() -> None:
    """The defect in #3986: deep nesting used to escape as ``RecursionError``.

    ``_eval_node`` recurses, so before this guard a nested expression from the
    HTTP derived-column endpoint produced a ``RecursionError`` -- not an
    ``ExpressionError`` -- and therefore surfaced to the operator as a 500
    instead of a clean 4xx. Parenthesised unary minus nests one level per
    character pair, so this reaches the depth limit well inside the AST node
    budget.
    """
    depth = MAX_NESTING_DEPTH + 20
    expr = "-" * depth + "a"
    with pytest.raises(ExpressionError) as excinfo:
        evaluate_expression(expr, _vars(a=[1.0]))
    assert not isinstance(excinfo.value, RecursionError)
    assert "nested" in str(excinfo.value)


def test_nesting_guard_does_not_reject_ordinary_expressions() -> None:
    """A realistic derived column must still evaluate."""
    out = evaluate_expression(
        "clip(sqrt(abs(a * b)) + mean(a) / 2.0, 0.0, 100.0)",
        _vars(a=[1.0, 4.0], b=[9.0, 16.0]),
    )
    assert np.all(np.isfinite(out))


def test_large_constant_exponent_is_rejected() -> None:
    """A single huge exponent is refused rather than silently returning inf."""
    with pytest.raises(ExpressionError, match="exponent too large"):
        evaluate_expression(f"a ** {MAX_POW_EXPONENT + 1}", _vars(a=[2.0]))


def test_column_exponent_is_still_allowed() -> None:
    """A runtime (non-constant) exponent is not a static bomb; keep it working."""
    out = evaluate_expression("a ** b", _vars(a=[2.0, 3.0], b=[2.0, 2.0]))
    np.testing.assert_allclose(out, [4.0, 9.0])
