"""Regression tests for the Gemini adapter (issues #2756 and #2764).

These tests stub out the third-party ``google.generativeai`` module so they
run in any environment, including CI hosts without the SDK installed.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# --------------------------------------------------------------------------- #
# Path / module shims                                                         #
# --------------------------------------------------------------------------- #
# Mirror the pattern used by tests/shared/python/chat/* — the production
# ``src.shared.python.ai.*`` packages reference a ``logging_pkg`` namespace
# that is not present in this worktree. We synthesize a minimal stub so the
# adapter import succeeds.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [str(_ROOT / "src")]  # type: ignore[attr-defined]
sys.modules.setdefault("src", _src_pkg)

_shared_pkg = types.ModuleType("src.shared")
_shared_pkg.__path__ = [str(_ROOT / "src" / "shared")]  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared", _shared_pkg)

_shared_python_pkg = types.ModuleType("src.shared.python")
_shared_python_pkg.__path__ = [  # type: ignore[attr-defined]
    str(_ROOT / "src" / "shared" / "python")
]
sys.modules.setdefault("src.shared.python", _shared_python_pkg)

_ai_pkg = types.ModuleType("src.shared.python.ai")
_ai_pkg.__path__ = [  # type: ignore[attr-defined]
    str(_ROOT / "src" / "shared" / "python" / "ai")
]
sys.modules.setdefault("src.shared.python.ai", _ai_pkg)

_adapters_pkg = types.ModuleType("src.shared.python.ai.adapters")
_adapters_pkg.__path__ = [  # type: ignore[attr-defined]
    str(_ROOT / "src" / "shared" / "python" / "ai" / "adapters")
]
sys.modules.setdefault("src.shared.python.ai.adapters", _adapters_pkg)

_logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
_logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
_logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config.setup_logging = lambda *a, **k: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", _logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", _logging_config)

from src.shared.python.ai.adapters.base import ToolDeclaration  # noqa: E402
from src.shared.python.ai.types import (  # noqa: E402
    ConversationContext,
    Message,
    ProviderCapability,
)


# --------------------------------------------------------------------------- #
# Shared fakes                                                                #
# --------------------------------------------------------------------------- #
class _FakeChat:
    """Minimal stand-in for ``GenerativeModel.start_chat()``."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send_message(self, message: str, stream: bool = False) -> Any:
        self.sent.append(message)
        if stream:

            def _gen() -> Any:
                chunk = MagicMock()
                chunk.text = "hello"
                yield chunk

            return _gen()
        response = MagicMock()
        response.text = "ok"
        return response


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def start_chat(self, history: Any) -> _FakeChat:
        return _FakeChat()

    def generate_content(self, prompt: str) -> Any:
        response = MagicMock()
        response.text = "pong"
        return response


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Install a fake ``google.generativeai`` module before import."""
    fake_module = types.ModuleType("google.generativeai")
    fake_types_module = types.ModuleType("google.generativeai.types")
    fake_types_module.GenerateContentResponse = object  # type: ignore[attr-defined]

    configured: list[str] = []

    def configure(*, api_key: str) -> None:
        configured.append(api_key)

    fake_module.configure = configure  # type: ignore[attr-defined]
    fake_module.GenerativeModel = _FakeModel  # type: ignore[attr-defined]
    fake_module.types = fake_types_module  # type: ignore[attr-defined]

    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_module)
    monkeypatch.setitem(sys.modules, "google.generativeai.types", fake_types_module)

    # Re-import the adapter module so it picks up the fakes. We pop all alias
    # spellings from ``sys.modules`` first to force a fresh import that binds
    # to our fakes.
    for alias in (
        "src.shared.python.ai.adapters.gemini_adapter",
        "shared.python.ai.adapters.gemini_adapter",
        "ai.adapters.gemini_adapter",
    ):
        monkeypatch.delitem(sys.modules, alias, raising=False)

    return types.SimpleNamespace(module=fake_module, configured=configured)


def _adapter_cls(fake_genai: types.SimpleNamespace) -> Any:
    """Import ``GeminiAdapter`` after the fake SDK is installed."""
    # Drop any cached copy on parent packages so the import below truly
    # re-executes the module body and rebinds to the fixture's fake genai.
    for parent_name in (
        "src.shared.python.ai.adapters",
        "shared.python.ai.adapters",
        "ai.adapters",
    ):
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, "gemini_adapter"):
            delattr(parent, "gemini_adapter")

    for alias in (
        "src.shared.python.ai.adapters.gemini_adapter",
        "shared.python.ai.adapters.gemini_adapter",
        "ai.adapters.gemini_adapter",
    ):
        sys.modules.pop(alias, None)

    import importlib

    gemini_adapter = importlib.import_module(
        "src.shared.python.ai.adapters.gemini_adapter"
    )

    # Force the legacy fallback path: our fake does not expose ``Client``.
    gemini_adapter.HAS_GEMINI_CLIENT = False
    gemini_adapter._GenaiClient = None
    # Rebind the module-level ``genai`` symbol to the fixture's fake so
    # ``_with_configured_sdk`` records into the fixture's ``configured`` list.
    gemini_adapter.genai = fake_genai.module
    gemini_adapter.GenerativeModel = fake_genai.module.GenerativeModel
    return gemini_adapter.GeminiAdapter


def _empty_context() -> ConversationContext:
    return ConversationContext(messages=[Message(role="user", content="hi")])


# --------------------------------------------------------------------------- #
# Issue #2756 — per-instance API keys                                         #
# --------------------------------------------------------------------------- #
def test_two_adapters_with_different_keys_both_work(
    fake_genai: types.SimpleNamespace,
) -> None:
    """Two adapters with different keys must each round-trip a message.

    Regression for #2756: previously the second instance silently overrode
    the first instance's API key in the SDK module-global state.
    """
    GeminiAdapter = _adapter_cls(fake_genai)

    a = GeminiAdapter(api_key="key-A", model="gemini-pro")
    b = GeminiAdapter(api_key="key-B", model="gemini-pro")

    resp_a = a.send_message("hello", _empty_context(), tools=[])
    resp_b = b.send_message("hello", _empty_context(), tools=[])

    assert resp_a.content == "ok"
    assert resp_b.content == "ok"

    # Each adapter must have re-configured the SDK with *its own* key at
    # least once (constructor + send_message paths).
    assert "key-A" in fake_genai.configured
    assert "key-B" in fake_genai.configured


def test_send_message_reconfigures_sdk_for_this_instance(
    fake_genai: types.SimpleNamespace,
) -> None:
    """Even after a sibling adapter clobbers the global, our key wins."""
    GeminiAdapter = _adapter_cls(fake_genai)

    a = GeminiAdapter(api_key="key-A")
    GeminiAdapter(api_key="key-B")  # would clobber genai.configure global

    fake_genai.configured.clear()
    a.send_message("hi", _empty_context(), tools=[])

    # The adapter must have re-applied its own key right before the call.
    assert fake_genai.configured[-1] == "key-A"


def test_empty_api_key_is_rejected(fake_genai: types.SimpleNamespace) -> None:
    """DbC: empty/blank API keys must raise ``ValueError``."""
    GeminiAdapter = _adapter_cls(fake_genai)

    with pytest.raises(ValueError, match="api_key"):
        GeminiAdapter(api_key="")


# --------------------------------------------------------------------------- #
# Issue #2764 — honest tools contract                                         #
# --------------------------------------------------------------------------- #
def test_send_message_with_tools_raises_not_implemented(
    fake_genai: types.SimpleNamespace,
) -> None:
    """Non-empty ``tools`` must raise instead of being silently dropped."""
    GeminiAdapter = _adapter_cls(fake_genai)
    adapter = GeminiAdapter(api_key="key-A")

    tool = ToolDeclaration(name="search", description="search the web")

    with pytest.raises(NotImplementedError, match="2764"):
        adapter.send_message("hello", _empty_context(), tools=[tool])


def test_stream_response_with_tools_raises_not_implemented(
    fake_genai: types.SimpleNamespace,
) -> None:
    """Streaming path also rejects non-empty ``tools``."""
    GeminiAdapter = _adapter_cls(fake_genai)
    adapter = GeminiAdapter(api_key="key-A")

    tool = ToolDeclaration(name="search", description="search the web")

    with pytest.raises(NotImplementedError, match="2764"):
        # iterator must be consumed to hit the body, but we expect the
        # rejection to fire eagerly (validated before entering the try)
        list(adapter.stream_response("hello", _empty_context(), tools=[tool]))


def test_capabilities_does_not_advertise_function_calling(
    fake_genai: types.SimpleNamespace,
) -> None:
    """Capabilities must reflect the honest contract."""
    GeminiAdapter = _adapter_cls(fake_genai)
    adapter = GeminiAdapter(api_key="key-A")

    caps = adapter.capabilities
    assert ProviderCapability.FUNCTION_CALLING not in caps.supported
    assert ProviderCapability.STREAMING in caps.supported


def test_send_message_with_empty_tools_works(
    fake_genai: types.SimpleNamespace,
) -> None:
    """Empty/None tool lists must not trigger the rejection."""
    GeminiAdapter = _adapter_cls(fake_genai)
    adapter = GeminiAdapter(api_key="key-A")

    resp = adapter.send_message("hello", _empty_context(), tools=[])
    assert resp.content == "ok"
