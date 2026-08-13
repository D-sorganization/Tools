"""Verify chat-message bubbles label the assistant with its model name.

Contract under test
-------------------

The user reported that the chat dialog referred to the assistant as
generic "AI". The new contract: assistant-side bubbles show the model
name in the role label, formatted as ``Agent (<model>)``. User-side
bubbles still show ``"You"``.

Resolution order for the assistant label, from highest to lowest priority:

1. Explicit ``agent_label`` passed to ``ChatMessageBubble``.
2. ``ChatDockWidget._format_agent_label()`` derived from the live
   ``_current_model`` / ``_current_provider`` state — the call site in
   ``_add_bubble`` passes this so live model switches are reflected on
   subsequent turns.
3. Plain ``"Agent"`` when neither is known (degenerate case — should
   not happen in production because the dock is constructed with a
   default provider/model, but tests cover it for robustness).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Bubble-level contract
# ---------------------------------------------------------------------------


def test_user_bubble_always_labelled_you(qapp) -> None:  # noqa: F811 - qapp is conftest fixture
    from src.shared.python.chat._qt.bubbles import ChatMessageBubble

    bubble = ChatMessageBubble("user", "hi", agent_label="Agent (gpt-4o)")
    # Even with an agent_label passed, a user-role bubble must still say "You".
    assert bubble._role_label.text() == "You"


def test_assistant_bubble_uses_supplied_agent_label(qapp) -> None:
    from src.shared.python.chat._qt.bubbles import ChatMessageBubble

    bubble = ChatMessageBubble("assistant", "hello", agent_label="Agent (llama3.1:8b)")
    assert bubble._role_label.text() == "Agent (llama3.1:8b)"


def test_assistant_bubble_falls_back_to_plain_agent_without_label(qapp) -> None:
    from src.shared.python.chat._qt.bubbles import ChatMessageBubble

    bubble = ChatMessageBubble("assistant", "hello")  # no agent_label
    assert bubble._role_label.text() == "Agent"


def test_assistant_bubble_treats_empty_label_as_no_label(qapp) -> None:
    from src.shared.python.chat._qt.bubbles import ChatMessageBubble

    bubble = ChatMessageBubble("assistant", "hello", agent_label="")
    # Empty string is falsy → plain Agent
    assert bubble._role_label.text() == "Agent"


# ---------------------------------------------------------------------------
# Dock-level _format_agent_label() resolution order
# ---------------------------------------------------------------------------


def _make_dock_with_state(monkeypatch: pytest.MonkeyPatch, qapp) -> object:
    """Construct a barebones ChatDockWidget instance for label testing.

    Bypasses the full UI build by allocating the class without
    ``__init__`` and only seeding the two state fields the formatter
    reads. This keeps the test fast and Windows-stable.
    """
    from src.shared.python.chat._chat_dock_widget_qt import ChatDockWidget

    dock = ChatDockWidget.__new__(ChatDockWidget)
    dock._current_model = ""
    dock._current_provider = ""
    return dock


def test_format_agent_label_prefers_model_over_provider(monkeypatch, qapp) -> None:
    dock = _make_dock_with_state(monkeypatch, qapp)
    dock._current_model = "claude-sonnet-4-7"
    dock._current_provider = "anthropic"

    label = dock._format_agent_label()
    assert label == "Agent (claude-sonnet-4-7)"


def test_format_agent_label_falls_back_to_provider_when_model_blank(
    monkeypatch, qapp
) -> None:
    dock = _make_dock_with_state(monkeypatch, qapp)
    dock._current_model = ""
    dock._current_provider = "claude-code"

    label = dock._format_agent_label()
    assert label == "Agent (claude-code)"


def test_format_agent_label_returns_plain_agent_when_both_blank(
    monkeypatch, qapp
) -> None:
    dock = _make_dock_with_state(monkeypatch, qapp)
    dock._current_model = ""
    dock._current_provider = ""

    label = dock._format_agent_label()
    assert label == "Agent"


def test_format_agent_label_strips_whitespace_only_values(monkeypatch, qapp) -> None:
    """Whitespace-only model/provider should be treated as blank."""
    dock = _make_dock_with_state(monkeypatch, qapp)
    dock._current_model = "   "
    dock._current_provider = "  ollama  "

    label = dock._format_agent_label()
    assert label == "Agent (ollama)"
