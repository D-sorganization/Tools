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

import pytest

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
