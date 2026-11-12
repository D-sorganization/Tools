"""Symbolic solver router with optional SymPy support. See issue #2682."""

from __future__ import annotations

from fastapi import APIRouter

from shared.python.safe_eval import validate_expression

from ..contracts.symbolic_solver import (
    SymbolicDerivativeRequest,
    SymbolicDerivativeResponse,
    SymbolicHelpResponse,
    SymbolicSimplifyRequest,
    SymbolicSimplifyResponse,
    SymbolicSolveRequest,
    SymbolicSolveResponse,
)

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
        available=SYMPY_AVAILABLE,
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
            validate_expression(lhs.strip())
            validate_expression(rhs.strip())
            lhs_expr = parse_expr(
                lhs.strip(),
                transformations=standard_transformations + (convert_xor,),
            )
            rhs_expr = parse_expr(
                rhs.strip(),
                transformations=standard_transformations + (convert_xor,),
            )
            equation = sp.Eq(lhs_expr, rhs_expr)
        else:
            # Assume expression equals zero
            validate_expression(request.equation.strip())
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
            available=SYMPY_AVAILABLE,
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
        validate_expression(request.expression.strip())
        expr = parse_expr(
            request.expression,
            transformations=standard_transformations + (convert_xor,),
        )
        variable = sp.Symbol(request.variable)
        derivative = sp.diff(expr, variable)

        return SymbolicDerivativeResponse(
            derivative=str(derivative),
            available=SYMPY_AVAILABLE,
        )
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
        validate_expression(request.expression.strip())
        expr = parse_expr(
            request.expression,
            transformations=standard_transformations + (convert_xor,),
        )
        simplified = sp.simplify(expr)

        return SymbolicSimplifyResponse(
            simplified=str(simplified),
            available=SYMPY_AVAILABLE,
        )
    except Exception as e:  # noqa: BLE001 - user-facing error message
        return SymbolicSimplifyResponse(error=f"Failed to simplify expression: {e!s}")
