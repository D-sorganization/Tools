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

Complexity limits
-----------------
Expressions arrive from the HTTP derived-column endpoint, so passing the
node-type allowlist is not sufficient: a very long or deeply nested expression
can still exhaust CPU or the recursion stack. The static limits from issue
#3290 are therefore imported from :mod:`shared.python.safe_eval` and enforced
here *before* any evaluation, each as a clean :class:`ExpressionError`:
:data:`MAX_EXPRESSION_LENGTH`, :data:`MAX_AST_NODES`, :data:`MAX_POW_EXPONENT`,
:data:`MAX_POW_CHAIN_DEPTH`, plus the locally defined
:data:`MAX_NESTING_DEPTH` (this evaluator recurses; the shared one compiles).

Relationship to ``shared.python.safe_eval``
-------------------------------------------
This module shares the shared evaluator's **limits** but deliberately keeps its
own **grammar and arithmetic**, because the two have incompatible contracts:

* ``safe_eval`` permits ``Compare``, ``BoolOp``, ``IfExp`` and ``Subscript``. A
  derived *column* must reject them -- a boolean, or a sliced single row, is not
  a column -- so this module's allowlist is strictly narrower.
* This module routes every binary operator through a numpy ufunc so that
  divide-by-zero and overflow yield ``inf``/``nan`` per its documented contract,
  where ``safe_eval`` raises.
* This module enforces per-function arity so a 3-argument ``min``/``max`` cannot
  reach numpy's ``out=`` parameter and write back into a caller's column, and it
  copies every input column for the same reason.

Collapsing the two evaluators into one would mean either loosening the
derived-column grammar or changing ``safe_eval``'s arithmetic for its other
callers. The DRY defect reported in #3986 was the *duplicated security
hardening*, and that is what is now shared; the divergent semantics above are
intentional and are the reason a full merge is not proposed.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

# DRY (#3986): the complexity/DoS limits are *not* re-declared here. They are
# imported from the hardened shared evaluator so there is exactly one place in
# the repo where "how large may a user expression be" is decided, and so a
# future tightening of the #3290 limits applies to this endpoint too.
#
# Only the *limits* are shared -- deliberately not the evaluator itself. See
# "Relationship to shared.python.safe_eval" in the module docstring.
from shared.python.safe_eval import (
    MAX_AST_NODES,
    MAX_EXPRESSION_LENGTH,
    MAX_POW_CHAIN_DEPTH,
    MAX_POW_EXPONENT,
)

__all__ = [
    "MAX_AST_NODES",
    "MAX_EXPRESSION_LENGTH",
    "MAX_NESTING_DEPTH",
    "MAX_POW_CHAIN_DEPTH",
    "MAX_POW_EXPONENT",
    "ExpressionError",
    "evaluate_expression",
]

#: Maximum AST nesting depth. :func:`_eval_node` is recursive, so an expression
#: nested deeper than the interpreter's recursion limit raised ``RecursionError``
#: -- which is not an ``ExpressionError`` and so escaped the derived-column
#: endpoint as a 500 instead of a clean client error (#3986). Enforced
#: statically, before evaluation.
MAX_NESTING_DEPTH = 50


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

# Allowed binary operators -> implementing callables. These use numpy ufuncs
# (not Python operators) so that overflow and divide-by-zero produce ``inf``/
# ``nan`` per the documented contract instead of raising ``OverflowError`` /
# ``ZeroDivisionError`` (which are not ``ExpressionError`` and would 500).
_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
    ast.FloorDiv: np.floor_divide,
}

# Per-function argument arity. ``min``/``max`` accept 1 (reduce) or 2
# (elementwise); ``clip`` needs 3; every other whitelisted function is unary.
# Enforced so a wrong arg count is a clean ExpressionError, never a numpy
# ``out=`` aliasing (which would mutate a caller column) or a raw numpy error.
_VARIADIC_MINMAX = frozenset({"min", "max"})
_TERNARY = frozenset({"clip"})


def _reject_if_too_complex(tree: ast.Expression) -> None:
    """Enforce the static complexity limits before any node is evaluated.

    Every violation is an :class:`ExpressionError`, so the derived-column
    endpoint reports a clean client error rather than a 500. Rejecting up front
    -- instead of discovering the problem partway through -- also means no
    partial numpy work is performed on a hostile expression.
    """
    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise ExpressionError(
                f"expression too complex: more than {MAX_AST_NODES} AST nodes"
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            _reject_bad_power(node)

    _reject_if_too_deep(tree.body)


def _reject_bad_power(node: ast.BinOp) -> None:
    """Reject exponentiation bombs (``2 ** 10**9``, ``2 ** 3 ** 4 ** 5``)."""
    exponent = node.right
    if (
        isinstance(exponent, ast.Constant)
        and isinstance(exponent.value, (int, float))
        and not isinstance(exponent.value, bool)
        and exponent.value > MAX_POW_EXPONENT
    ):
        raise ExpressionError(
            f"exponent too large (> {MAX_POW_EXPONENT}); possible exponentiation bomb"
        )

    depth = 1
    walker: ast.AST = exponent
    while isinstance(walker, ast.BinOp) and isinstance(walker.op, ast.Pow):
        depth += 1
        if depth > MAX_POW_CHAIN_DEPTH:
            raise ExpressionError(
                f"power chain deeper than {MAX_POW_CHAIN_DEPTH}; "
                "possible exponentiation bomb"
            )
        walker = walker.right


def _reject_if_too_deep(node: ast.AST) -> None:
    """Reject nesting deeper than :data:`MAX_NESTING_DEPTH`.

    Iterative on purpose: a recursive depth check would itself exhaust the stack
    on exactly the input it exists to reject.
    """
    stack: list[tuple[ast.AST, int]] = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_NESTING_DEPTH:
            raise ExpressionError(
                f"expression nested deeper than {MAX_NESTING_DEPTH} levels"
            )
        for child in ast.iter_child_nodes(current):
            stack.append((child, depth + 1))


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
    if len(expr) > MAX_EXPRESSION_LENGTH:
        # Checked before ast.parse: parsing a multi-megabyte string is itself
        # the denial of service.
        raise ExpressionError(
            f"expression too long ({len(expr)} chars, limit {MAX_EXPRESSION_LENGTH})"
        )
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
        # Always copy: an expression must never be able to write back into a
        # caller's column (e.g. via a numpy ``out=`` aliasing path).
        coerced[name] = np.array(value, dtype=float, copy=True)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression syntax: {exc}") from exc
    except (RecursionError, MemoryError, ValueError) as exc:
        # CPython's own parser can blow up on pathological input; that must
        # still surface as a clean client error, never a 500.
        raise ExpressionError(f"expression too complex to parse: {exc}") from exc

    _reject_if_too_complex(tree)

    try:
        result = _eval_node(tree.body, coerced)
    except RecursionError as exc:  # pragma: no cover - defence in depth
        # _reject_if_too_deep should already have rejected this; the conversion
        # stays so a limit regression can never become a 500.
        raise ExpressionError("expression nested too deeply") from exc

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
    n = len(args)

    # Strict per-function arity. This both gives clean errors and prevents a
    # 3-arg ``min``/``max`` from reaching numpy's binary ufunc, where the 3rd
    # positional is the ``out=`` destination and would mutate a column.
    if name in _VARIADIC_MINMAX:
        if n == 1:
            chosen: Callable[..., object] = _REDUCERS[name]
        elif n == 2:
            chosen = func  # np.minimum / np.maximum, elementwise
        else:
            raise ExpressionError(
                f"{name}() takes 1 (reduce) or 2 (elementwise) arguments, got {n}"
            )
    elif name == "mean":
        if n != 1:
            raise ExpressionError(f"mean() takes exactly 1 argument, got {n}")
        chosen = func
    elif name in _TERNARY:
        if n != 3:
            raise ExpressionError(
                f"{name}() takes exactly 3 arguments (x, lo, hi), got {n}"
            )
        chosen = func
    else:
        if n != 1:
            raise ExpressionError(f"{name}() takes exactly 1 argument, got {n}")
        chosen = func

    try:
        return chosen(*args)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ExpressionError(f"invalid call to {name}(): {exc}") from exc
