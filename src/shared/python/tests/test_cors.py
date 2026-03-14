import os

import pytest
from cors import DEFAULT_ORIGINS, add_cors_middleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.shared.python.contracts import PreconditionError, set_contracts_enabled


@pytest.fixture
def clean_env():
    set_contracts_enabled(True)
    old = os.environ.get("CORS_ORIGINS")
    if "CORS_ORIGINS" in os.environ:
        del os.environ["CORS_ORIGINS"]
    yield
    if old is not None:
        os.environ["CORS_ORIGINS"] = old
    set_contracts_enabled(False)


def get_cors_origins(app: FastAPI) -> list[str]:
    for middleware in app.user_middleware:
        if middleware.cls == CORSMiddleware:
            return middleware.kwargs.get("allow_origins", [])
    return []


def get_cors_kwargs(app: FastAPI) -> dict:
    for middleware in app.user_middleware:
        if middleware.cls == CORSMiddleware:
            return middleware.kwargs
    return {}


def test_add_cors_middleware_default(clean_env):
    app = FastAPI()
    add_cors_middleware(app)
    kwargs = get_cors_kwargs(app)
    assert kwargs["allow_origins"] == DEFAULT_ORIGINS
    assert kwargs["allow_methods"] == ["*"]
    assert kwargs["allow_headers"] == ["*"]
    assert kwargs["allow_credentials"] is True


def test_add_cors_middleware_explicit_origins(clean_env):
    app = FastAPI()
    origins = ["http://my-domain.com"]
    add_cors_middleware(app, origins=origins)
    assert get_cors_origins(app) == origins


def test_add_cors_middleware_env_override(clean_env, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://env1.com, http://env2.com ,")
    app = FastAPI()
    # It should override even if explicit origins are provided
    add_cors_middleware(app, origins=["http://explicit.com"])
    origins = get_cors_origins(app)
    assert origins == ["http://env1.com", "http://env2.com"]


def test_add_cors_middleware_kwargs(clean_env):
    app = FastAPI()
    add_cors_middleware(
        app,
        allow_methods=["GET"],
        allow_headers=["X-Custom"],
        expose_headers=["X-Exposed"],
    )
    kwargs = get_cors_kwargs(app)
    assert kwargs["allow_methods"] == ["GET"]
    assert kwargs["allow_headers"] == ["X-Custom"]
    assert kwargs["expose_headers"] == ["X-Exposed"]


def test_add_cors_middleware_invalid_app(clean_env):
    with pytest.raises(PreconditionError):
        add_cors_middleware("not_an_app")


def test_add_cors_middleware_invalid_origins(clean_env):
    app = FastAPI()
    with pytest.raises(PreconditionError):
        add_cors_middleware(app, origins="not_a_list")

    with pytest.raises(PreconditionError):
        add_cors_middleware(app, origins=[1, 2, 3])
