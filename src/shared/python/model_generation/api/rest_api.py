<<<<<<< HEAD
# ARCHITECTURE_DEBT resolved — tracked as GitHub issue #1953
# Split into focused submodules (issue #1953):
#   rest_api_types.py   — HTTPMethod, APIRequest, APIResponse, Route
#   rest_api_routes.py  — ModelGenerationAPI with all handler methods
#   rest_api_flask.py   — FlaskAdapter
#   rest_api_fastapi.py — FastAPIAdapter
# This file is now a backward-compatibility shim only.

"""
REST API for model_generation package.

Provides HTTP endpoints for URDF generation, conversion, editing, and library access.
Can be used with Flask, FastAPI, or other frameworks via adapters.

All symbols are re-exported from the focused submodules for backward compatibility.
"""

from __future__ import annotations

# Re-export all public names for backward compatibility
from .rest_api_fastapi import FastAPIAdapter  # noqa: F401
from .rest_api_flask import FlaskAdapter  # noqa: F401
from .rest_api_routes import ModelGenerationAPI  # noqa: F401
from .rest_api_types import (  # noqa: F401
=======
"""Backward-compatible re-export shim for the split model_generation REST API."""

from model_generation.api.rest_api_contracts import (
>>>>>>> origin/main
    APIRequest,
    APIResponse,
    HTTPMethod,
    Route,
)
<<<<<<< HEAD

__all__ = [
    "HTTPMethod",
    "APIRequest",
    "APIResponse",
    "Route",
    "ModelGenerationAPI",
    "FlaskAdapter",
    "FastAPIAdapter",
=======
from model_generation.api.rest_api_core import ModelGenerationAPI
from model_generation.api.rest_api_fastapi import FastAPIAdapter
from model_generation.api.rest_api_flask import FlaskAdapter

__all__ = [
    "APIRequest",
    "APIResponse",
    "FastAPIAdapter",
    "FlaskAdapter",
    "HTTPMethod",
    "ModelGenerationAPI",
    "Route",
>>>>>>> origin/main
]
