"""ODE solver router.  See issue #608."""

from __future__ import annotations

import math
import re

from fastapi import APIRouter, HTTPException

from ..contracts.ode_solver import (
    ODESolverRequest,
    ODESolverResponse,
    ODEVariableSummary,
)

router = APIRouter(prefix="/api/calc/ode-solver", tags=["ode-solver"])


@router.post("", response_model=ODESolverResponse)
def solve_ode(request: ODESolverRequest) -> ODESolverResponse:
    """Solve a system of ODEs using RK4 integration."""
    var_names = list(request.derivatives.keys())

    # Validate initial conditions match derivatives
    for var in var_names:
        if var not in request.initial_conditions:
            raise HTTPException(
                status_code=422,
                detail=f"Missing initial condition for variable '{var}'",
            )

    try:
        result = _rk4_solve(
            var_names=var_names,
            expressions=request.derivatives,
            parameters=request.parameters,
            initial=request.initial_conditions,
            t_start=request.t_start,
            t_end=request.t_end,
            num_points=request.num_points,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result


def _safe_eval(
    expr: str,
    variables: dict[str, float],
    parameters: dict[str, float],
) -> float:
    """Safely evaluate a mathematical expression."""
    ctx = {**parameters, **variables}

    processed = expr
    # Sort by length descending to avoid partial replacements
    for name in sorted(ctx.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(name) + r"\b"
        processed = re.sub(pattern, f"({ctx[name]})", processed)

    # Map math functions
    processed = re.sub(r"\bsin\b", "math.sin", processed)
    processed = re.sub(r"\bcos\b", "math.cos", processed)
    processed = re.sub(r"\bexp\b", "math.exp", processed)
    processed = re.sub(r"\bsqrt\b", "math.sqrt", processed)
    processed = re.sub(r"\babs\b", "abs", processed)
    processed = re.sub(r"\bPI\b", "math.pi", processed)
    # Handle ** operator
    processed = re.sub(
        r"\(([^)]+)\)\s*\*\*\s*(\d+(?:\.\d+)?)",
        r"math.pow(\1,\2)",
        processed,
    )

    # Only allow safe characters
    allowed = set("0123456789.+-*/() ,mathsincosexpqrtabpow")
    stripped = processed.replace("math.", "").replace("abs", "").replace("pow", "")
    for ch in stripped:
        if ch not in allowed and not ch.isspace():
            raise ValueError(f"Unsafe character in expression: '{ch}'")

    return float(eval(processed, {"__builtins__": {}}, {"math": math}))  # noqa: S307


def _rk4_solve(
    var_names: list[str],
    expressions: dict[str, str],
    parameters: dict[str, float],
    initial: dict[str, float],
    t_start: float,
    t_end: float,
    num_points: int,
) -> ODESolverResponse:
    """RK4 integration of the ODE system."""
    dt = (t_end - t_start) / (num_points - 1)
    state = {v: initial[v] for v in var_names}

    times: list[float] = []
    solutions: dict[str, list[float]] = {v: [] for v in var_names}

    def compute_derivs(t: float, s: dict[str, float]) -> dict[str, float]:
        ctx = {**s, "t": t}
        return {v: _safe_eval(expressions[v], ctx, parameters) for v in var_names}

    for i in range(num_points):
        t = t_start + i * dt
        times.append(round(t, 8))
        for v in var_names:
            solutions[v].append(round(state[v], 8))

        if i < num_points - 1:
            k1 = compute_derivs(t, state)

            s2 = {v: state[v] + dt / 2 * k1[v] for v in var_names}
            k2 = compute_derivs(t + dt / 2, s2)

            s3 = {v: state[v] + dt / 2 * k2[v] for v in var_names}
            k3 = compute_derivs(t + dt / 2, s3)

            s4 = {v: state[v] + dt * k3[v] for v in var_names}
            k4 = compute_derivs(t + dt, s4)

            for v in var_names:
                state[v] += dt / 6 * (k1[v] + 2 * k2[v] + 2 * k3[v] + k4[v])

    summaries = []
    for v in var_names:
        vals = solutions[v]
        summaries.append(
            ODEVariableSummary(
                name=v,
                initial_value=vals[0],
                final_value=vals[-1],
                min_value=min(vals),
                max_value=max(vals),
            )
        )

    return ODESolverResponse(
        times=times,
        solutions=solutions,
        variable_summaries=summaries,
    )
