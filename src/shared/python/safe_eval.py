"""Safe mathematical expression evaluator.

Replaces all uses of ``eval()`` with a hardened AST-based evaluator that:

1. Parses the expression into an AST and validates every node type.
2. Compiles the validated AST into a code object (no raw string eval).
3. Executes the compiled code in a namespace that contains **only** the
   caller-supplied names -- ``__builtins__`` is always empty.

This eliminates the class of attacks where ``eval()`` can be abused even
when ``__builtins__`` is set to ``{}``.

Design-by-Contract
-------------------
* **Precondition**: ``expression`` is a non-empty string; ``namespace`` keys
  are all plain identifiers.
* **Postcondition**: the return value is whatever the compiled expression
  produces; no side-effects outside ``namespace``.
* **Invariant**: only the node types listed in ``_ALLOWED_NODE_TYPES`` will
  ever be executed.
"""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np

from shared.python.contracts import require

__all__ = [
    "safe_eval",
    "safe_eval_math",
    "validate_expression",
    "NUMPY_MATH_NAMESPACE",
    "SCALAR_MATH_NAMESPACE",
]

# ── Allowed AST node types ──────────────────────────────────────────────
# These are the *only* node kinds we permit.  Anything else (Import,
# FunctionDef, Attribute access, etc.) is rejected.

_ALLOWED_NODE_TYPES: tuple[type, ...] = (
    ast.Expression,
    ast.Load,
    # Arithmetic / logic
    ast.BinOp,
    ast.UnaryOp,
    ast.operator,
    ast.unaryop,
    ast.cmpop,
    ast.Compare,
    ast.BoolOp,
    ast.boolop,
    # Literals & names
    ast.Constant,
    ast.Name,
    # Function calls (only bare-name calls, no attribute calls)
    ast.Call,
    ast.keyword,
    # Subscript / slice (for array indexing)
    ast.Subscript,
    ast.Slice,
    # IfExp (ternary)
    ast.IfExp,
)


# ── Complexity / DoS guards (issue #3290) ───────────────────────────────
# Without these, expressions that pass the node-type allowlist can still hang
# the process or exhaust memory: ``9**9**9**9`` (bignum pow) or ``"x"*10**9``
# (repetition bomb). These are cheap *static* limits applied during validation.

#: Maximum raw expression length, in characters.
MAX_EXPRESSION_LENGTH = 10_000
#: Maximum number of AST nodes permitted in a single expression.
MAX_AST_NODES = 500
#: Reject a ``Pow`` whose exponent is a constant larger than this.
MAX_POW_EXPONENT = 1_000
#: Reject ``Pow`` chains (``a ** b ** c ...``) deeper than this.
MAX_POW_CHAIN_DEPTH = 2
#: Reject string/bytes constants longer than this (math has no use for them).
MAX_STR_CONSTANT_LENGTH = 256


# ── Pre-built namespaces ────────────────────────────────────────────────

NUMPY_MATH_NAMESPACE: dict[str, Any] = {
    # Standard functions (numpy versions for array support)
    "abs": np.abs,
    "min": np.min,
    "max": np.max,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "sum": np.sum,
    "len": len,
    "round": np.round,
    # Trigonometric
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    # Exponential / logarithmic
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "pow": np.power,
    # Statistical
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    # Constants
    "pi": np.pi,
    "e": np.e,
    # np-prefixed aliases
    "np_sqrt": np.sqrt,
    "np_log": np.log,
    "np_log10": np.log10,
    "np_exp": np.exp,
    "np_sin": np.sin,
    "np_cos": np.cos,
    "np_tan": np.tan,
    "np_abs": np.abs,
    "np_mean": np.mean,
    "np_std": np.std,
    "np_min": np.min,
    "np_max": np.max,
}

SCALAR_MATH_NAMESPACE: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


# ── Core functions ──────────────────────────────────────────────────────


def validate_expression(
    expression: str,
    allowed_names: set[str] | None = None,
) -> ast.Expression:
    """Parse *expression* and validate every AST node.

    Parameters
    ----------
    expression:
        The math expression to validate.
    allowed_names:
        If provided, every ``ast.Name`` node must reference a name in this
        set.  Pass ``None`` to skip name checking (the caller is
        responsible for controlling the execution namespace).

    Returns
    -------
    ast.Expression
        The validated AST, ready to be compiled.

    Raises
    ------
    ValueError
        If the expression contains disallowed constructs.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression must not be empty")

    require(
        isinstance(expression, str),
        "expression must be a string",
    )

    # Cheap length guard before parsing (issue #3290).
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression too long: {len(expression)} > {MAX_EXPRESSION_LENGTH}"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid syntax: {exc}") from exc

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise ValueError(
                f"Expression too complex: more than {MAX_AST_NODES} AST nodes"
            )

        # Check node type is allowed
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"Unsafe operation detected: {type(node).__name__}")

        # Reject string/bytes constants (repetition-bomb material; the math
        # evaluator has no business with large strings). Numbers are fine.
        if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
            if len(node.value) > MAX_STR_CONSTANT_LENGTH:
                raise ValueError("String/bytes constant exceeds allowed length")

        # Bound exponentiation to defeat bignum pow bombs (e.g. 9**9**9**9).
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            _check_pow_safety(node)

        # Validate names
        if isinstance(node, ast.Name):
            if allowed_names is not None and node.id not in allowed_names:
                raise ValueError(f"Unknown variable or function: {node.id}")

        # Only bare-name function calls allowed (no attribute calls like
        # os.system)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                func_id = func.id
                if allowed_names is not None and func_id not in allowed_names:
                    raise ValueError(f"Unknown function: {func_id}")
            else:
                raise ValueError("Attribute-based function calls not allowed")

    return tree


def _check_pow_safety(node: ast.BinOp) -> None:
    """Reject dangerous exponentiation: large constant exponents and deep chains.

    ``9 ** 9 ** 9 ** 9`` parses as right-associated nested ``Pow`` nodes, each
    of which would be evaluated as an unbounded bignum. We reject both a single
    huge constant exponent and any chain of ``Pow`` deeper than
    :data:`MAX_POW_CHAIN_DEPTH`.
    """
    # Constant exponent magnitude check.
    exp = node.right
    if isinstance(exp, ast.Constant) and isinstance(exp.value, int | float):
        try:
            if abs(exp.value) > MAX_POW_EXPONENT:
                raise ValueError(
                    f"Exponent too large (> {MAX_POW_EXPONENT}); "
                    "possible exponentiation bomb"
                )
        except OverflowError as exc:  # pragma: no cover - inf/nan exponents
            raise ValueError("Invalid exponent") from exc

    # Nested-Pow chain depth check (count consecutive Pow on either operand).
    depth = 1
    for child in (node.left, node.right):
        cur = child
        local = 0
        while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Pow):
            local += 1
            cur = cur.right
        depth = max(depth, 1 + local)
    if depth > MAX_POW_CHAIN_DEPTH:
        raise ValueError(
            f"Exponentiation nested too deeply (> {MAX_POW_CHAIN_DEPTH}); "
            "possible exponentiation bomb"
        )


def safe_eval(
    expression: str,
    namespace: dict[str, Any],
    *,
    allowed_names: set[str] | None = None,
) -> Any:
    """Evaluate *expression* safely in *namespace*.

    Parameters
    ----------
    expression:
        Mathematical expression to evaluate.
    namespace:
        Dict of names the expression may reference (variables, functions,
        constants).  ``__builtins__`` is always forced to ``{}``.
    allowed_names:
        Optional explicit allowlist.  Defaults to ``namespace.keys()``.

    Returns
    -------
    Any
        Result of the expression evaluation.
    """
    if allowed_names is None:
        allowed_names = set(namespace.keys())

    tree = validate_expression(expression, allowed_names)
    code = compile(tree, "<safe_eval>", "eval")
    return eval(code, {"__builtins__": {}}, namespace)  # nosec B307


def safe_eval_math(
    expression: str,
    variables: dict[str, Any] | None = None,
    *,
    use_numpy: bool = True,
) -> Any:
    """Convenience wrapper that merges caller variables with math functions.

    Parameters
    ----------
    expression:
        Mathematical expression.
    variables:
        Caller-supplied variables (signal data, parameters, etc.).
    use_numpy:
        If True, use numpy math functions (array-safe).  Otherwise use
        scalar ``math`` module functions.
    """
    base = dict(NUMPY_MATH_NAMESPACE if use_numpy else SCALAR_MATH_NAMESPACE)
    if variables:
        base.update(variables)
    return safe_eval(expression, base)
