"""Symbolic solver router with optional SymPy support. See issue #2682."""

from __future__ import annotations

from fastapi import APIRouter
import ast
from pydantic import BaseModel, Field

def _ast_security_gate(expression: str) -> None:
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")
    stripped = expression.strip()
    if not stripped:
        return
    if len(expression) > 1000:
        raise ValueError("Expression exceeds maximum allowed length")

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError:
        return

    _CALC_ALLOWED_AST_NODES = (
        ast.Expression, ast.Call, ast.Name, ast.Load,
        ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.cmpop,
        ast.Constant,
        ast.Subscript, ast.Index, ast.Slice, ast.ExtSlice,
        ast.Compare, ast.BoolOp, ast.boolop,
    ) if hasattr(ast, "Constant") else (
        ast.Expression, ast.Call, ast.Name, ast.Load,
        ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.cmpop,
        ast.Num, ast.Str, ast.Bytes, ast.List, ast.Tuple, ast.Set, ast.Dict,
        ast.Ellipsis, ast.NameConstant, ast.Constant,
        ast.Subscript, ast.Index, ast.Slice, ast.ExtSlice,
        ast.Compare, ast.BoolOp, ast.boolop,
    )

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > 1000:
            raise ValueError("Expression is too complex")
        if not isinstance(node, _CALC_ALLOWED_AST_NODES):
            raise ValueError(
                f"Forbidden expression construct: {type(node).__name__}"
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
            if len(node.value) > 1000:
                raise ValueError("String constant exceeds allowed length")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("Attribute-based function calls are not allowed")

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        parse_expr,
        standard_transformations,
    )

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None


router = APIRouter(prefix="/api/calc/symbolic", tags=["symbolic-solver"])


class SymbolicSolveRequest(BaseModel):
    """Request model for symbolic equation solving."""

    equation: str = Field(..., description="Equation to solve (e.g., 'x**2 - 4 = 0')")
    variable: str = Field(..., description="Variable to solve for (e.g., 'x')")


class SymbolicSolveResponse(BaseModel):
    """Response model for symbolic equation solving."""

    solutions: list[str] = Field(default_factory=list)
    raw_result: str | None = None
    available: bool = Field(default=SYMPY_AVAILABLE)
    error: str | None = None


class SymbolicDerivativeRequest(BaseModel):
    """Request model for symbolic differentiation."""

    expression: str = Field(..., description="Expression to differentiate")
    variable: str = Field(..., description="Variable with respect to")


class SymbolicDerivativeResponse(BaseModel):
    """Response model for symbolic differentiation."""

    derivative: str | None = None
    available: bool = Field(default=SYMPY_AVAILABLE)
    error: str | None = None


class SymbolicSimplifyRequest(BaseModel):
    """Request model for symbolic simplification."""

    expression: str = Field(..., description="Expression to simplify")


class SymbolicSimplifyResponse(BaseModel):
    """Response model for symbolic simplification."""

    simplified: str | None = None
    available: bool = Field(default=SYMPY_AVAILABLE)
    error: str | None = None


class SymbolicHelpResponse(BaseModel):
    """Help metadata for symbolic calculator capabilities."""

    supported_operations: list[str]
    examples: list[dict[str, str]]
    available: bool = Field(default=SYMPY_AVAILABLE)


@router.get("/help", response_model=SymbolicHelpResponse)
def get_symbolic_help() -> SymbolicHelpResponse:
    """Return guided symbolic workflow metadata."""
    return SymbolicHelpResponse(
        supported_operations=[
            "solve_equation",
            "solve_system",
            "derivative",
            "simplify",
            "expand",
            "factor",
        ],
        examples=[
            {
                "operation": "solve_equation",
                "request": '{"equation": "x**2 - 4 = 0", "variable": "x"}',
                "description": "Solve quadratic equation",
            },
            {
                "operation": "derivative",
                "request": '{"expression": "x**3 + 2*x", "variable": "x"}',
                "description": "Compute derivative",
            },
            {
                "operation": "simplify",
                "request": '{"expression": "(x**2 - 1)/(x - 1)"}',
                "description": "Simplify rational expression",
            },
        ],
    )


@router.post("/solve", response_model=SymbolicSolveResponse)
def solve_equation(request: SymbolicSolveRequest) -> SymbolicSolveResponse:
    """Solve a symbolic equation for a specified variable."""
    if not SYMPY_AVAILABLE:
        return SymbolicSolveResponse(
            available=False,
            error="SymPy is not available. Install with: pip install sympy",
        )

    try:
        # Parse the equation
        if "=" in request.equation:
            lhs, rhs = request.equation.split("=", 1)
            _ast_security_gate(lhs)
            lhs_expr = parse_expr(
                lhs.strip(),
                transformations=standard_transformations + (convert_xor,),
            )
            _ast_security_gate(rhs)
            rhs_expr = parse_expr(
                rhs.strip(),
                transformations=standard_transformations + (convert_xor,),
            )
            equation = sp.Eq(lhs_expr, rhs_expr)
        else:
            # Assume expression equals zero
            _ast_security_gate(request.equation)
            expr = parse_expr(
                request.equation,
                transformations=standard_transformations + (convert_xor,),
            )
            equation = sp.Eq(expr, 0)

        # Parse the variable
        variable = sp.Symbol(request.variable)

        # Solve the equation
        solutions = sp.solve(equation, variable)

        # Convert solutions to strings
        solution_strings = [str(sol) for sol in solutions]

        return SymbolicSolveResponse(
            solutions=solution_strings,
            raw_result=str(solutions),
        )
    except Exception as e:  # noqa: BLE001 - user-facing error message
        return SymbolicSolveResponse(error=f"Failed to solve equation: {e!s}")


@router.post("/derivative", response_model=SymbolicDerivativeResponse)
def compute_derivative(
    request: SymbolicDerivativeRequest,
) -> SymbolicDerivativeResponse:
    """Compute the symbolic derivative of an expression."""
    if not SYMPY_AVAILABLE:
        return SymbolicDerivativeResponse(
            available=False,
            error="SymPy is not available. Install with: pip install sympy",
        )

    try:
        _ast_security_gate(request.expression)
        expr = parse_expr(
            request.expression,
            transformations=standard_transformations + (convert_xor,),
        )
        variable = sp.Symbol(request.variable)
        derivative = sp.diff(expr, variable)

        return SymbolicDerivativeResponse(derivative=str(derivative))
    except Exception as e:  # noqa: BLE001 - user-facing error message
        return SymbolicDerivativeResponse(error=f"Failed to compute derivative: {e!s}")


@router.post("/simplify", response_model=SymbolicSimplifyResponse)
def simplify_expression(request: SymbolicSimplifyRequest) -> SymbolicSimplifyResponse:
    """Simplify a symbolic expression."""
    if not SYMPY_AVAILABLE:
        return SymbolicSimplifyResponse(
            available=False,
            error="SymPy is not available. Install with: pip install sympy",
        )

    try:
        _ast_security_gate(request.expression)
        expr = parse_expr(
            request.expression,
            transformations=standard_transformations + (convert_xor,),
        )
        simplified = sp.simplify(expr)

        return SymbolicSimplifyResponse(simplified=str(simplified))
    except Exception as e:  # noqa: BLE001 - user-facing error message
        return SymbolicSimplifyResponse(error=f"Failed to simplify expression: {e!s}")
