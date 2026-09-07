# ARCHITECTURE_DEBT resolved — tracked as GitHub issue #1953
# Split into focused submodules: rest_api_types, rest_api_routes,
# rest_api_flask, rest_api_fastapi. Backward-compatibility shim only.
"""REST API for model_generation package (backward-compatibility shim)."""

from __future__ import annotations

# Re-export all public names for backward compatibility
from .rest_api_core import ModelGenerationAPI  # noqa: F401
from .rest_api_fastapi import FastAPIAdapter  # noqa: F401
from .rest_api_flask import FlaskAdapter  # noqa: F401
from .rest_api_types import (  # noqa: F401
    APIRequest,
    APIResponse,
    HTTPMethod,
    Route,
)

__all__ = [
    "HTTPMethod",
    "APIRequest",
    "APIResponse",
    "Route",
    "ModelGenerationAPI",
    "FlaskAdapter",
    "FastAPIAdapter",
]
