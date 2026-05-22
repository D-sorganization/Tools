"""Shared test configuration and fixtures for Tools."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_PYTHON = REPO_ROOT / "src" / "shared" / "python"
if str(SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(SHARED_PYTHON))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate-api-baseline",
        action="store_true",
        default=False,
        help="Regenerate tests/sidekick_api_baseline.json with the current public API.",
    )
