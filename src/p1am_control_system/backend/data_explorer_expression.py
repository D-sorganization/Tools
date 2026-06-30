"""Safe arithmetic expression evaluator over named numpy columns.

The Data Explorer lets users define *derived columns* from arithmetic
expressions referencing existing column names (e.g. ``"power = volts * amps"``,
written here as the right-hand side ``"volts * amps"``). Evaluating arbitrary
user strings with :func:`eval` would be a remote-code-execution hole, so this
module parses the expression with :func:`ast.parse` and walks the tree,
permitting only a small, explicitly whitelisted grammar:

* :class:`ast.Expression` root,
* binary arithmetic ``+ - * / ** % //``,
* unary ``+`` / ``-``,
* numeric constants,
* names that resolve to a supplied column **or** to a math constant in
  :data:`_CONSTANTS` (``pi``, ``e``),
* calls to whitelisted numpy functions by *bare name* only
  (no attribute access, no subscripting, no comprehensions, etc.).

Any node outside that grammar — attribute access, subscripting, lambdas,
comprehensions, boolean/compare operators, starred args, keywords — raises
:class:`ExpressionError`.

Preconditions
-------------
* ``expr`` must be a :class:`str`.
* ``variables`` must be a mapping of ``str`` -> :class:`numpy.ndarray`; it must
  be non-empty.

Edge handling
-------------
* Division by zero is **not** raised: numpy yields ``inf`` / ``nan`` and that is
  propagated (documented behaviour — gaps surface downstream).
* If the evaluated result is a scalar (the expression referenced no column, or
  collapsed to one via ``mean``/``sum`` etc.), it is broadcast to a 1-D array
  the length of the **first** variable in ``variables``.
* Otherwise the result is returned as a 1-D float array.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = ["evaluate_expression", "ExpressionError"]


class ExpressionError(ValueError):
    """Raised when an expression is malformed or uses a disallowed construct."""


# Math constants exposed to expressions as bare names.
_CONSTANTS: dict[str, float] = {"pi": float(np.pi), "e": float(np.e)}

# Whitelisted callables mapped to their numpy implementations. ``min``/``max``/
# ``clip`` operate elementwise / with scalar broadcasting.
_FUNCTIONS: dict[str, Callable[..., object]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "floor": np.floor,
    "ceil": np.ceil,
    "sign": np.sign,
    "min": np.minimum,
    "max": np.maximum,
    "mean": np.mean,
    "clip": np.clip,
}

# Reductions taking a single array argument (so a bare ``min(x)`` reduces rather
# than erroring on the binary ``np.minimum`` signature).
_REDUCERS: dict[str, Callable[..., object]] = {
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
}

# Allowed binary operators -> implementing callables.
_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}


def evaluate_expression(expr: str, variables: Mapping[str, np.ndarray]) -> np.ndarray:
    """Safely evaluate ``expr`` against the named columns in ``variables``.

    Parameters
    ----------
    expr:
        Arithmetic expression string (the right-hand side of a derived column).
    variables:
        Mapping of column name to 1-D numpy array. Must be non-empty.

    Returns
    -------
    numpy.ndarray
        1-D float array. Scalar results are broadcast to the length of the
        first variable in ``variables``.

    Raises
    ------
    TypeError
        If ``expr`` is not a ``str`` or ``variables`` is not a mapping of
        ``str`` -> :class:`numpy.ndarray`.
    ValueError
        If ``variables`` is empty.
    ExpressionError
        If the expression is syntactically invalid or uses a disallowed node /
        unknown name.
    """
    if not isinstance(expr, str):
        raise TypeError("expr must be a str")
    if not isinstance(variables, Mapping):
        raise TypeError("variables must be a mapping of str -> ndarray")
    if len(variables) == 0:
        raise ValueError("variables must be non-empty")

    coerced: dict[str, np.ndarray] = {}
    for name, value in variables.items():
        if not isinstance(name, str):
            raise TypeError("variable names must be str")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"variable {name!r} must be a numpy.ndarray")
        coerced[name] = np.asarray(value, dtype=float)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression syntax: {exc}") from exc

    result = _eval_node(tree.body, coerced)

    first_len = len(next(iter(coerced.values())))
    array = np.asarray(result, dtype=float)
    if array.ndim == 0:
        array = np.full(first_len, float(array))
    return np.ravel(array).astype(float, copy=False)


def _eval_node(node: ast.AST, variables: Mapping[str, np.ndarray]) -> object:
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        impl = _BINOPS.get(op_type)
        if impl is None:
            raise ExpressionError(f"operator {op_type.__name__} is not allowed")
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        return impl(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return +operand  # type: ignore[operator]
        if isinstance(node.op, ast.USub):
            return -operand  # type: ignore[operator]
        raise ExpressionError(f"unary operator {type(node.op).__name__} is not allowed")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError("only numeric constants are allowed")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ExpressionError(f"unknown name: {node.id!r}")

    if isinstance(node, ast.Call):
        return _eval_call(node, variables)

    raise ExpressionError(f"disallowed expression element: {type(node).__name__}")


def _eval_call(node: ast.Call, variables: Mapping[str, np.ndarray]) -> object:
    """Evaluate a whitelisted function call by bare name."""
    if not isinstance(node.func, ast.Name):
        raise ExpressionError("only bare function names may be called")
    if node.keywords:
        raise ExpressionError("keyword arguments are not allowed")
    name = node.func.id
    func = _FUNCTIONS.get(name)
    if func is None:
        raise ExpressionError(f"unknown function: {name!r}")

    args = [_eval_node(arg, variables) for arg in node.args]

    # Single-arg reductions: route ``min``/``max``/``mean`` to the reducer.
    if name in _REDUCERS and len(args) == 1:
        return _REDUCERS[name](args[0])
    return func(*args)
