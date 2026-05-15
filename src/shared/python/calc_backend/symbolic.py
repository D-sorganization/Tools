"""Optional SymPy-backed symbolic calculator service facade."""

from __future__ import annotations

import html
import importlib
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from .contracts.symbolic import (
    SymbolicLimits,
    SymbolicRenderedValue,
    SymbolicRenderRequest,
    SymbolicRenderResponse,
    SymbolicSolveRequest,
    SymbolicSolveResponse,
    SymbolicWorkflow,
)

_SYMBOL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_EXPRESSION_PATTERN = re.compile(r"^[A-Za-z0-9_+\-*/^=().,\[\] <>\t]+$")
_ALLOWED_NAMES = {
    "E",
    "Eq",
    "Matrix",
    "cos",
    "exp",
    "log",
    "pi",
    "sin",
    "sqrt",
    "tan",
}


def symbolic_workflows() -> tuple[SymbolicWorkflow, ...]:
    """Return static guided workflows shared by API and UI contracts."""
    return (
        SymbolicWorkflow(
            id="equation",
            title="Solve one equation",
            summary="Solve a bounded algebraic expression for one symbol.",
            steps=[
                "Enter an expression or equality.",
                "Choose one target symbol.",
                "Review the text and LaTeX result before reusing it.",
            ],
            example="solve x**2 - 4 for x",
            limits="Runs with expression length, symbol count, and timeout guards.",
        ),
        SymbolicWorkflow(
            id="system",
            title="Solve a small system",
            summary="Solve up to three equations for explicit symbols.",
            steps=[
                "Enter one equation per row.",
                "List each target symbol.",
                "Check each returned solution mapping.",
            ],
            example="solve x + y = 3; x - y = 1 for x, y",
            limits="Systems are capped by max_equations and max_symbols.",
        ),
        SymbolicWorkflow(
            id="substitution",
            title="Solve with substitutions",
            summary="Apply named substitutions before solving or rendering.",
            steps=[
                "Enter symbolic values in the substitution map.",
                "Solve the reduced expression.",
                "Keep the original rendered expression for context.",
            ],
            example="solve a*x - 6 for x with a=3",
            limits="Substitutions use the same parser and complexity limits.",
        ),
    )


class SymbolicMathService:
    """Facade isolating optional SymPy use from Sidekick callers."""

    def solve(
        self,
        payload: SymbolicSolveRequest | dict[str, Any],
    ) -> SymbolicSolveResponse:
        """Solve accepted symbolic equations or report backend unavailability."""
        request = _coerce_solve_request(payload)
        try:
            sympy = _load_sympy()
        except ImportError:
            return SymbolicSolveResponse(
                success=False,
                backend="unavailable",
                message="Install optional dependency sympy to enable symbolic solving.",
                workflows=list(symbolic_workflows()),
            )
        _validate_limits(request.equations, request.symbols, request.limits)
        namespace = _symbol_namespace(sympy, request.symbols)
        equations = [
            _parse_expression(sympy, expression, namespace, request.limits)
            for expression in request.equations
        ]
        substitutions = _parse_substitutions(
            sympy,
            request.substitutions,
            namespace,
            request.limits,
        )
        if substitutions:
            equations = [equation.subs(substitutions) for equation in equations]
        symbols = tuple(namespace[name] for name in request.symbols)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(sympy.solve, equations, symbols, dict=True)
            try:
                raw_solutions = future.result(timeout=request.limits.timeout_seconds)
            except TimeoutError as exc:
                raise TimeoutError("symbolic solve exceeded timeout_seconds") from exc
        rendered = [
            _render_expression(sympy, expression, value)
            for expression, value in zip(request.equations, equations, strict=True)
        ]
        return SymbolicSolveResponse(
            success=True,
            backend="sympy",
            message="Solution computed successfully",
            solutions=_format_solutions(raw_solutions),
            rendered=rendered,
            workflows=list(symbolic_workflows()),
        )

    def render(
        self,
        payload: SymbolicRenderRequest | dict[str, Any],
    ) -> SymbolicRenderResponse:
        """Render accepted expressions as sanitized display text and LaTeX."""
        request = _coerce_render_request(payload)
        try:
            sympy = _load_sympy()
        except ImportError:
            return SymbolicRenderResponse(
                success=False,
                backend="unavailable",
                message="Install optional dependency sympy to enable LaTeX rendering.",
                workflows=list(symbolic_workflows()),
            )
        _validate_limits(request.expressions, (), request.limits)
        namespace = _symbol_namespace(sympy, _discover_symbols(request.expressions))
        rendered = [
            _render_expression(
                sympy,
                expression,
                _parse_expression(sympy, expression, namespace, request.limits),
            )
            for expression in request.expressions
        ]
        return SymbolicRenderResponse(
            success=True,
            backend="sympy",
            message="Rendered successfully",
            rendered=rendered,
            workflows=list(symbolic_workflows()),
        )


def _coerce_solve_request(
    payload: SymbolicSolveRequest | dict[str, Any],
) -> SymbolicSolveRequest:
    if isinstance(payload, SymbolicSolveRequest):
        return payload
    return SymbolicSolveRequest(**payload)


def _coerce_render_request(
    payload: SymbolicRenderRequest | dict[str, Any],
) -> SymbolicRenderRequest:
    if isinstance(payload, SymbolicRenderRequest):
        return payload
    return SymbolicRenderRequest(**payload)


def _load_sympy() -> Any:
    return importlib.import_module("sympy")


def _validate_limits(
    expressions: list[str],
    symbols: tuple[str, ...] | list[str],
    limits: SymbolicLimits,
) -> None:
    if len(expressions) > limits.max_equations:
        raise ValueError("equation count exceeds max_equations")
    if len(symbols) > limits.max_symbols:
        raise ValueError("symbol count exceeds max_symbols")
    for symbol in symbols:
        if not _SYMBOL_PATTERN.match(symbol) or symbol in _ALLOWED_NAMES:
            raise ValueError(f"unsupported symbol name: {symbol}")
    for expression in expressions:
        if len(expression) > limits.max_expression_chars:
            raise ValueError("expression exceeds max_expression_chars")
        if "__" in expression or not _EXPRESSION_PATTERN.match(expression):
            raise ValueError("expression contains unsupported syntax")


def _symbol_namespace(
    sympy: Any,
    symbols: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    namespace = {name: getattr(sympy, name) for name in _ALLOWED_NAMES}
    for symbol in symbols:
        namespace[symbol] = sympy.Symbol(symbol)
    return namespace


def _parse_expression(
    sympy: Any,
    expression: str,
    namespace: dict[str, Any],
    limits: SymbolicLimits,
) -> Any:
    _validate_limits([expression], (), limits)
    parser = importlib.import_module("sympy.parsing.sympy_parser")
    normalized = expression.replace("^", "**")
    if "=" in normalized:
        left, right = normalized.split("=", 1)
        return sympy.Eq(
            parser.parse_expr(
                left,
                local_dict=namespace,
                global_dict=_parser_globals(sympy),
            ),
            parser.parse_expr(
                right,
                local_dict=namespace,
                global_dict=_parser_globals(sympy),
            ),
        )
    return parser.parse_expr(
        normalized,
        local_dict=namespace,
        global_dict=_parser_globals(sympy),
    )


def _parser_globals(sympy: Any) -> dict[str, Any]:
    return {
        "Float": sympy.Float,
        "Integer": sympy.Integer,
        "Rational": sympy.Rational,
        "__builtins__": {},
    }


def _parse_substitutions(
    sympy: Any,
    substitutions: dict[str, str],
    namespace: dict[str, Any],
    limits: SymbolicLimits,
) -> dict[Any, Any]:
    parsed: dict[Any, Any] = {}
    for name, value in substitutions.items():
        if name not in namespace:
            namespace[name] = sympy.Symbol(name)
        parsed[namespace[name]] = _parse_expression(
            sympy,
            str(value),
            namespace,
            limits,
        )
    return parsed


def _discover_symbols(expressions: list[str]) -> tuple[str, ...]:
    names = set()
    for expression in expressions:
        for name in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expression):
            if name not in _ALLOWED_NAMES:
                names.add(name)
    return tuple(sorted(names))


def _render_expression(
    sympy: Any,
    source: str,
    expression: Any,
) -> SymbolicRenderedValue:
    return SymbolicRenderedValue(
        input=source,
        display_text=html.escape(source),
        latex=sympy.latex(expression),
    )


def _format_solutions(raw_solutions: Any) -> list[dict[str, str]]:
    return [
        {str(symbol): str(value) for symbol, value in solution.items()}
        for solution in raw_solutions
    ]
