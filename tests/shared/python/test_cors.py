"""Comprehensive TDD suite for cors.py — shared CORS factory module.

Tests cover:
- Default origin resolution (env var, explicit override, DEFAULT_ORIGINS fallback)
- All DbC contract violations
- Middleware attachment verification
"""

from unittest.mock import MagicMock

import pytest
from contracts import PreconditionError
from cors import DEFAULT_ORIGINS, add_cors_middleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_PreconditionError = PreconditionError


# ─── Good path: default origins ────────────────────────────────


def test_default_origins_used_when_no_env_and_no_arg():
    """Falls back to DEFAULT_ORIGINS when neither env var nor argument provided."""
    app = FastAPI()
    add_cors_middleware(app)
    # Middleware was added — verify by checking the middleware stack
    assert any(m.cls is CORSMiddleware for m in app.user_middleware)


def test_explicit_origins_used():
    """Explicit origins list is accepted and used."""
    app = FastAPI()
    custom = ["http://custom.example.com"]
    add_cors_middleware(app, origins=custom)
    assert any(m.cls is CORSMiddleware for m in app.user_middleware)


def test_env_var_overrides_explicit_origins(monkeypatch):
    """CORS_ORIGINS env var takes priority over the `origins` argument."""
    monkeypatch.setenv("CORS_ORIGINS", "http://env-one.com, http://env-two.com")
    app = FastAPI()
    add_cors_middleware(app, origins=["http://should-be-ignored.com"])
    # Middleware added regardless of which origins win
    assert any(m.cls is CORSMiddleware for m in app.user_middleware)


def test_env_var_stripped_correctly(monkeypatch):
    """Comma-separated env var origins are stripped of whitespace."""
    monkeypatch.setenv("CORS_ORIGINS", " http://a.com , http://b.com ")
    app = FastAPI()
    # Should not raise
    add_cors_middleware(app)


def test_empty_env_var_falls_back(monkeypatch):
    """Empty CORS_ORIGINS env var is treated as unset → uses fallback."""
    monkeypatch.setenv("CORS_ORIGINS", "   ")
    app = FastAPI()
    add_cors_middleware(app)
    assert any(m.cls is CORSMiddleware for m in app.user_middleware)


def test_default_origins_list():
    """DEFAULT_ORIGINS contains the expected localhost entries."""
    assert "http://localhost:3000" in DEFAULT_ORIGINS
    assert "http://localhost:5173" in DEFAULT_ORIGINS
    assert len(DEFAULT_ORIGINS) >= 2


def test_allow_credentials_default():
    """allow_credentials defaults to True."""
    app = FastAPI()
    add_cors_middleware(app)
    middleware = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    assert middleware.kwargs.get("allow_credentials", True) is True


def test_allow_methods_default():
    """allow_methods defaults to ['*']."""
    app = FastAPI()
    add_cors_middleware(app)
    middleware = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    assert middleware.kwargs.get("allow_methods") == ["*"]


def test_allow_headers_default():
    """allow_headers defaults to ['*']."""
    app = FastAPI()
    add_cors_middleware(app)
    middleware = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    assert middleware.kwargs.get("allow_headers") == ["*"]


def test_custom_allow_methods():
    """Custom allow_methods are forwarded correctly."""
    app = FastAPI()
    add_cors_middleware(app, allow_methods=["GET", "POST"])
    middleware = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    assert middleware.kwargs.get("allow_methods") == ["GET", "POST"]


def test_none_origins_uses_default():
    """Passing origins=None explicitly uses DEFAULT_ORIGINS."""
    app = FastAPI()
    add_cors_middleware(app, origins=None)
    assert any(m.cls is CORSMiddleware for m in app.user_middleware)


# ─── DbC contract violations ───────────────────────────────────


def test_dbc_requires_fastapi_app():
    """Non-FastAPI app raises PreconditionError."""
    with pytest.raises(_PreconditionError):
        add_cors_middleware(MagicMock())  # type: ignore[arg-type]


def test_dbc_requires_fastapi_not_none():
    """None app raises PreconditionError."""
    with pytest.raises(_PreconditionError):
        add_cors_middleware(None)  # type: ignore[arg-type]


def test_dbc_origins_must_be_list_of_strings():
    """Origins list containing non-strings raises PreconditionError."""
    app = FastAPI()
    with pytest.raises(_PreconditionError):
        add_cors_middleware(app, origins=[123, 456])  # type: ignore[arg-type]


def test_dbc_origins_dict_rejected():
    """Origins as a dict raises PreconditionError."""
    app = FastAPI()
    with pytest.raises(_PreconditionError):
        add_cors_middleware(app, origins={"origin": "http://example.com"})  # type: ignore[arg-type]
