"""Environment-only application factory used by the isolated Uvicorn process."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from .api import create_authority_app
from .capability import QUALIFIED_EXECUTION_CAPABILITY
from .job_store import AuthorityJobStore
from .jobs import AuthorityJobManager
from .production_runner import run_regional_ground_production_job
from .state_security import bounded_state_path

TOKEN_ENVIRONMENT_VARIABLE = "ROC_AUTHORITY_TOKEN"
STATE_ROOT_ENVIRONMENT_VARIABLE = "ROC_AUTHORITY_STATE_ROOT"
_STORE_FILENAME = "authority.v1.sqlite3"


def _state_store_from_environment() -> AuthorityJobStore:
    """Open the fixed private store path injected by the parent launcher."""
    source = os.environ.get(STATE_ROOT_ENVIRONMENT_VARIABLE, "")
    root = Path(source)
    if not source or not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("authority state root is unavailable or unsafe")
    return AuthorityJobStore(
        bounded_state_path(root, _STORE_FILENAME),
        max_retained_jobs=4,
    )


def create_app_from_environment() -> FastAPI:
    """Create the local authority without accepting secrets on the command line."""
    token = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE, "")
    return create_authority_app(
        token=token,
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        job_manager=AuthorityJobManager(
            runner=run_regional_ground_production_job,
            store=_state_store_from_environment(),
        ),
    )
