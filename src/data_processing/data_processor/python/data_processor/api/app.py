"""FastAPI application factory for the Data Processor API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import export_router, files_router, processing_router
from .state import AppState


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Data Processor API",
        description="REST API for signal processing and data analysis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    _configure_cors(app)
    _configure_routes(app)
    _configure_state(app)

    return app


def _configure_cors(app: FastAPI) -> None:
    """Configure CORS middleware for cross-origin requests."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _configure_routes(app: FastAPI) -> None:
    """Register API route handlers."""
    app.include_router(files_router, prefix="/api/v1/files", tags=["files"])
    app.include_router(processing_router, prefix="/api/v1/processing", tags=["processing"])
    app.include_router(export_router, prefix="/api/v1/export", tags=["export"])

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}


def _configure_state(app: FastAPI) -> None:
    """Configure application state for storing loaded data."""
    app.state.app_state = AppState()


def get_app_state(app: FastAPI) -> AppState:
    """Get the application state from a FastAPI app instance."""
    return app.state.app_state
