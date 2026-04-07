"""Backward-compatible re-export shim for the split model_generation REST API."""

from model_generation.api.rest_api_contracts import (
    APIRequest,
    APIResponse,
    HTTPMethod,
    Route,
)
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
]
