"""Shared Calculation Backend -- FastAPI application.

Provides a unified REST API wrapping all process calculators so React frontends
and other HTTP clients can call validated Python calculation engines.

Usage:
    uvicorn calc_backend.app:app --reload --port 8010

See issue #613.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    acid_gas_dewpoint,
    baghouse,
    financial,
    flare,
    flow_rate,
    ode_solver,
    pressure_drop,
    scrubber,
    syngas_water,
    thermal_profile,
    wgs_reactor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Shared Calculation Backend",
    description=(
        "Unified REST API for process engineering calculators.  "
        "Auto-generated docs at /docs (Swagger UI) and /redoc."
    ),
    version="1.0.0",
)

# Restrict CORS to known local development origins.
# Override with CORS_ORIGINS env var (comma-separated) if needed.
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
_env_origins = os.environ.get("CORS_ORIGINS")
_cors_origins = _env_origins.split(",") if _env_origins else _DEFAULT_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(flare.router)
app.include_router(wgs_reactor.router)
app.include_router(baghouse.router)
app.include_router(scrubber.router)
app.include_router(financial.router)
app.include_router(acid_gas_dewpoint.router)
app.include_router(pressure_drop.router)
app.include_router(flow_rate.router)
app.include_router(syngas_water.router)
app.include_router(thermal_profile.router)
app.include_router(ode_solver.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok"}


@app.get("/api/calc/endpoints")
def list_endpoints() -> dict[str, list[str]]:
    """List all available calculator endpoints."""
    return {
        "calculators": [
            "POST /api/calc/flare",
            "POST /api/calc/wgs-reactor",
            "POST /api/calc/baghouse",
            "POST /api/calc/scrubber",
            "POST /api/calc/financial",
            "POST /api/calc/acid-gas-dewpoint",
            "POST /api/calc/pressure-drop",
            "POST /api/calc/flow-rate",
            "POST /api/calc/syngas-water",
            "POST /api/calc/thermal-profile",
            "POST /api/calc/ode-solver",
        ],
    }
