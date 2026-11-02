"""Shared import helper for backend endpoint tests (test-quality #5924)."""

from __future__ import annotations

import importlib

import pytest


def import_main_app():
    """Import the FastAPI ``main`` module or skip when a dependency is absent."""
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    try:
        return importlib.import_module("main")
    except ModuleNotFoundError as exc:
        pytest.skip(f"P1AM backend dependency missing: {exc}")
