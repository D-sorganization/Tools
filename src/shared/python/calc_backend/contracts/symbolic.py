"""Pydantic contracts for bounded symbolic calculator operations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SymbolicLimits(BaseModel):
    """Runtime and complexity limits for symbolic operations."""

    max_expression_chars: int = Field(default=240, ge=1, le=2000)
    max_equations: int = Field(default=3, ge=1, le=8)
    max_symbols: int = Field(default=4, ge=1, le=8)
    timeout_seconds: float = Field(default=1.0, gt=0.0, le=10.0)


class SymbolicSolveRequest(BaseModel):
    """Request model for bounded symbolic solving."""

    equations: list[str] = Field(..., min_length=1)
    symbols: list[str] = Field(..., min_length=1)
    substitutions: dict[str, str] = Field(default_factory=dict)
    limits: SymbolicLimits = Field(default_factory=SymbolicLimits)


class SymbolicRenderRequest(BaseModel):
    """Request model for LaTeX rendering without solving."""

    expressions: list[str] = Field(..., min_length=1)
    limits: SymbolicLimits = Field(default_factory=SymbolicLimits)


class SymbolicRenderedValue(BaseModel):
    """Sanitized display text and LaTeX for one accepted expression."""

    input: str
    display_text: str
    latex: str


class SymbolicWorkflow(BaseModel):
    """Guided symbolic workflow metadata for Sidekick UI surfaces."""

    id: str
    title: str
    summary: str
    steps: list[str]
    example: str
    limits: str


class SymbolicSolveResponse(BaseModel):
    """Response model for symbolic solving."""

    success: bool
    backend: str
    message: str
    solutions: list[dict[str, str]] = Field(default_factory=list)
    rendered: list[SymbolicRenderedValue] = Field(default_factory=list)
    workflows: list[SymbolicWorkflow] = Field(default_factory=list)


class SymbolicRenderResponse(BaseModel):
    """Response model for LaTeX rendering."""

    success: bool
    backend: str
    message: str
    rendered: list[SymbolicRenderedValue] = Field(default_factory=list)
    workflows: list[SymbolicWorkflow] = Field(default_factory=list)
