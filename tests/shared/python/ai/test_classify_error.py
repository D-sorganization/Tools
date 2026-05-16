"""Tests for the shared _classify_error helper on BaseAgentAdapter.

Verifies that:
- Rate-limit strings map to AIRateLimitError for each provider.
- Timeout strings map to AITimeoutError for each provider.
- Connection/network strings map to AIConnectionError for each provider.
- Unknown errors map to AIProviderError for each provider.
- The timeout value is forwarded correctly.
- The provider name is embedded in the raised exception.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure the repo root is on sys.path and stub minimal packages so
# that importing the adapters works in a plain pytest run.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.config", "src/shared/python/config"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.adapters", "src/shared/python/ai/adapters"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        import types

        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub


_log_cfg = sys.modules.get("src.shared.python.logging_pkg.logging_config")
if not isinstance(_log_cfg, types.ModuleType):
    _log_cfg = types.ModuleType("src.shared.python.logging_pkg.logging_config")
    sys.modules["src.shared.python.logging_pkg.logging_config"] = _log_cfg
_log_cfg.get_logger = logging.getLogger  # type: ignore[attr-defined]


# Stub ai.config so adapters can import without a real environment.
_env_stub = sys.modules.get("src.shared.python.config.environment")
if not isinstance(_env_stub, types.ModuleType):
    _env_stub = types.ModuleType("src.shared.python.config.environment")
    sys.modules["src.shared.python.config.environment"] = _env_stub
_env_stub.get_env = lambda key, default=None, required=False: default
_env_stub.get_env_float = lambda key, default=0.0: float(default)

# Stub memory_manager
if "src.shared.python.ai.memory_manager" not in sys.modules:
    _mm = types.ModuleType("src.shared.python.ai.memory_manager")
    _mm.build_memory_prompt_section = lambda **_kw: ""  # type: ignore[attr-defined]
    _mm.load_agents_md = lambda *_a: None  # type: ignore[attr-defined]
    sys.modules["src.shared.python.ai.memory_manager"] = _mm

# Now the real imports can succeed.
from src.shared.python.ai.adapters.anthropic_adapter import (  # noqa: E402
    AnthropicAdapter,
)
from src.shared.python.ai.adapters.cline_adapter import ClineAdapter  # noqa: E402
from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter  # noqa: E402
from src.shared.python.ai.exceptions import (  # noqa: E402
    AIConnectionError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT = 42.0


def _anthropic() -> AnthropicAdapter:
    return AnthropicAdapter(api_key="sk-test", timeout=TIMEOUT)


def _openai() -> OpenAIAdapter:
    return OpenAIAdapter(api_key="sk-test", timeout=TIMEOUT)


def _cline() -> ClineAdapter:
    return ClineAdapter(timeout=TIMEOUT)


ADAPTERS = [
    pytest.param(_anthropic, "anthropic", id="anthropic"),
    pytest.param(_openai, "openai", id="openai"),
    pytest.param(_cline, "cline", id="cline"),
]

# ---------------------------------------------------------------------------
# Parametrized classification tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("factory,provider", ADAPTERS)
@pytest.mark.parametrize(
    "message",
    ["rate limit exceeded", "429 Too Many Requests", "too many requests"],
)
def test_rate_limit_errors(factory, provider, message):
    """String-scan on rate-limit keywords must raise AIRateLimitError."""
    adapter = factory()
    exc = RuntimeError(message)
    with pytest.raises(AIRateLimitError) as exc_info:
        raise adapter._classify_error(exc, provider=provider)
    assert exc_info.value.provider == provider


@pytest.mark.unit
@pytest.mark.parametrize("factory,provider", ADAPTERS)
@pytest.mark.parametrize("message", ["request timeout", "timed out after 30s"])
def test_timeout_errors(factory, provider, message):
    """String-scan on timeout keywords must raise AITimeoutError."""
    adapter = factory()
    exc = RuntimeError(message)
    with pytest.raises(AITimeoutError) as exc_info:
        raise adapter._classify_error(exc, provider=provider, timeout=TIMEOUT)
    assert exc_info.value.provider == provider
    assert exc_info.value.timeout == TIMEOUT


@pytest.mark.unit
@pytest.mark.parametrize("factory,provider", ADAPTERS)
@pytest.mark.parametrize(
    "message",
    ["connection refused", "network error", "host unreachable"],
)
def test_connection_errors(factory, provider, message):
    """String-scan on network keywords must raise AIConnectionError."""
    adapter = factory()
    exc = RuntimeError(message)
    with pytest.raises(AIConnectionError) as exc_info:
        raise adapter._classify_error(exc, provider=provider)
    assert exc_info.value.provider == provider


@pytest.mark.unit
@pytest.mark.parametrize("factory,provider", ADAPTERS)
def test_generic_error(factory, provider):
    """Unrecognised errors must raise the base AIProviderError."""
    adapter = factory()
    exc = RuntimeError("something completely unexpected")
    result = adapter._classify_error(exc, provider=provider)
    print(f"DEBUG AIProviderError is: {AIProviderError!r}")
    print(f"DEBUG result is: {result!r}")
    assert isinstance(result, AIProviderError)
    assert result.provider == provider
    # Must NOT be a subclass (rate-limit / timeout / connection)
    assert type(result) is AIProviderError


@pytest.mark.unit
@pytest.mark.parametrize("factory,provider", ADAPTERS)
def test_classify_error_chaining(factory, provider):
    """raise ... from error must preserve the cause chain."""
    adapter = factory()
    original = ValueError("boom")
    with pytest.raises(AIProviderError) as exc_info:
        raise adapter._classify_error(original, provider=provider) from original
    assert exc_info.value.__cause__ is original


@pytest.mark.unit
@pytest.mark.parametrize("factory,provider", ADAPTERS)
def test_timeout_forwarded_via_handle_error(factory, provider):
    """_handle_error must propagate the timeout value."""
    adapter = factory()
    exc = RuntimeError("connection timed out")
    with pytest.raises(AITimeoutError) as exc_info:
        adapter._handle_error(exc)
    assert exc_info.value.timeout == TIMEOUT
