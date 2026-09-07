"""Backward-compatibility shim for model_generation REST API (issue #1953).

Re-exports from rest_api_core, rest_api_routes, rest_api_flask, rest_api_fastapi.
"""

from __future__ import annotations

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
