"""Backward-compatible re-export shim for the model_generation REST API."""

from .rest_api_fastapi import FastAPIAdapter
from .rest_api_flask import FlaskAdapter
from .rest_api_routes import (
    APIRequest,
    APIResponse,
    HTTPMethod,
    ModelGenerationAPI,
    Route,
)

__all__ = [
    "APIRequest",
    "APIResponse",
    "FastAPIAdapter",
    "FlaskAdapter",
    "HTTPMethod",
    "ModelGenerationAPI",
    "Route",
]
