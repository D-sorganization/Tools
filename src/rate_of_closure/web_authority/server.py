"""Environment-only application factory used by the isolated Uvicorn process."""

from __future__ import annotations

import os

from fastapi import FastAPI

from .api import create_authority_app

TOKEN_ENVIRONMENT_VARIABLE = "ROC_AUTHORITY_TOKEN"


def create_app_from_environment() -> FastAPI:
    """Create the local authority without accepting secrets on the command line."""
    token = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE, "")
    return create_authority_app(token=token)
