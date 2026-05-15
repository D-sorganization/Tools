"""Symbolic calculator router with optional SymPy handling."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..contracts.symbolic import (
    SymbolicRenderRequest,
    SymbolicRenderResponse,
    SymbolicSolveRequest,
    SymbolicSolveResponse,
)
from ..symbolic import SymbolicMathService

router = APIRouter(prefix="/api/calc/symbolic", tags=["symbolic"])
_service = SymbolicMathService()


@router.post("/solve", response_model=SymbolicSolveResponse)
def solve_symbolic(request: SymbolicSolveRequest) -> SymbolicSolveResponse:
    """Solve bounded symbolic equations."""
    try:
        return _service.solve(request)
    except (TimeoutError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/render", response_model=SymbolicRenderResponse)
def render_symbolic(request: SymbolicRenderRequest) -> SymbolicRenderResponse:
    """Render bounded symbolic expressions as sanitized text and LaTeX."""
    try:
        return _service.render(request)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
