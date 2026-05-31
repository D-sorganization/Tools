"""Tests for the P1AM backend CORS allowlist configuration (issue #3144).

The backend directory is not an importable package (its modules rely on a
flat sys.path), so the standalone ``cors_config`` module is loaded directly
from its file path. This keeps the test free of fastapi/sqlmodel imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "p1am_control_system"
    / "backend"
    / "cors_config.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "p1am_cors_config_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cors_config = _load_module()


def test_default_is_local_dev_origins_not_wildcard() -> None:
    """With no config, defaults to local dev origins, never '*'."""
    settings = cors_config.resolve_cors_settings(env={})
    assert "*" not in settings.allow_origins
    assert settings.allow_origins == cors_config.DEFAULT_DEV_ORIGINS
    # Credentials default off so the wildcard/credentials trap is impossible.
    assert settings.allow_credentials is False


def test_explicit_allowlist_passes_through() -> None:
    """An allowed origin from the env allowlist is honored."""
    env = {"P1AM_CORS_ORIGINS": "https://hmi.example, https://ops.example"}
    settings = cors_config.resolve_cors_settings(env=env)
    assert settings.allow_origins == (
        "https://hmi.example",
        "https://ops.example",
    )
    assert "https://hmi.example" in settings.allow_origins


def test_disallowed_origin_is_not_present() -> None:
    """A disallowed origin never appears in the resolved allowlist."""
    env = {"P1AM_CORS_ORIGINS": "https://hmi.example"}
    settings = cors_config.resolve_cors_settings(env=env)
    assert "https://evil.example" not in settings.allow_origins


def test_wildcard_with_credentials_is_rejected() -> None:
    """'*' combined with credentials must raise (browser-unsafe)."""
    env = {
        "P1AM_CORS_ORIGINS": "*",
        "P1AM_CORS_ALLOW_CREDENTIALS": "true",
    }
    with pytest.raises(ValueError, match="wildcard|'\\*'"):
        cors_config.resolve_cors_settings(env=env)


def test_cors_settings_invariant_enforced_directly() -> None:
    """The dataclass invariant rejects '*' + credentials at construction."""
    with pytest.raises(ValueError):
        cors_config.CorsSettings(allow_origins=("*",), allow_credentials=True)


def test_production_without_allowlist_fails_closed() -> None:
    """Production with no explicit allowlist must hard-fail."""
    env = {"P1AM_ENV": "production"}
    with pytest.raises(RuntimeError, match="production"):
        cors_config.resolve_cors_settings(env=env)


def test_production_with_allowlist_ok() -> None:
    """Production with an explicit allowlist resolves normally."""
    env = {
        "P1AM_ENV": "production",
        "P1AM_CORS_ORIGINS": "https://hmi.prod.example",
    }
    settings = cors_config.resolve_cors_settings(env=env)
    assert settings.allow_origins == ("https://hmi.prod.example",)


def test_credentials_opt_in() -> None:
    """Credentials stay off unless explicitly enabled."""
    env = {"P1AM_CORS_ORIGINS": "https://hmi.example"}
    assert cors_config.resolve_cors_settings(env=env).allow_credentials is False
    env["P1AM_CORS_ALLOW_CREDENTIALS"] = "1"
    assert cors_config.resolve_cors_settings(env=env).allow_credentials is True
