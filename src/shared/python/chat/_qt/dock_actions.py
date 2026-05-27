# ruff: noqa: E501
"""Header-chrome dock actions: pop-out, new chat, clear chat.

This module extracts the implementation of the three header buttons added
alongside the existing Tools menu so the parent ``_chat_dock_widget_qt``
module stays under the 1500-line budget.

Each function takes the ``ChatDockWidget`` instance as its first argument
and operates on it in place. The dock retains thin wrapper methods that
delegate here so external code and tests calling
``dock.pop_out()`` / ``dock.new_chat()`` / ``dock.clear_chat()`` still
work unchanged.

Design:
    * **LOD**: helpers only talk to the dock through its public-ish
      attributes (``_message_layout``, ``_queued_messages``,
      ``_ai_model_combo``, ``_send_ws``); they never reach through Qt
      parent chains.
    * **DRY**: button construction is funnelled through
      :func:`make_chrome_button` so the New chat / Clear chat / Pop out
      icons all get identical styling.
    * **DbC**: each public helper documents its preconditions and
      postconditions in the docstring.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import QMessageBox, QPushButton, QSizePolicy

if TYPE_CHECKING:  # pragma: no cover - import for type-check only
    from ..chat_popout_window import ChatPopoutWindow

log = logging.getLogger(__name__)

# Canonical CLI/terminal-style provider ids. Sourced from the
# cli_provider_availability descriptor table; duplicated as a literal
# frozenset here so this module does not need to import that file at
# every call site (and so unit tests do not need the binaries installed
# to assert routing).
CLI_PROVIDER_IDS: frozenset[str] = frozenset(
    {
        "claude-code",
        "codex",
        "cline-cli",
        "gemini-cli",
        "github-cli",
    }
)


def is_cli_provider(provider_id: str) -> bool:
    """Return ``True`` if ``provider_id`` is a CLI/terminal provider.

    DbC:
        Pre: ``provider_id`` is a string.
        Post: returns a bool; never raises for unknown ids.
    """
    if not isinstance(provider_id, str):
        return False
    return provider_id in CLI_PROVIDER_IDS


def make_chrome_button(
    *,
    text: str,
    tooltip: str,
    bg: str,
    fg: str,
    border_hover: str,
    on_clicked: Callable[[], None],
    width: int = 28,
) -> QPushButton:
    """DRY helper for the small icon buttons in the dock header chrome.

    DbC:
        Pre: ``text`` and ``tooltip`` are non-empty strings.
        Pre: ``on_clicked`` is callable.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("make_chrome_button: text must be a non-empty string")
    if not isinstance(tooltip, str) or not tooltip:
        raise ValueError("make_chrome_button: tooltip must be a non-empty string")
    if not callable(on_clicked):
        raise TypeError("make_chrome_button: on_clicked must be callable")
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setFixedWidth(width)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
    btn.setStyleSheet(
        "QPushButton {"
        f"  background-color: {bg}; color: {fg};"
        "  border-radius: 4px; padding: 2px;"
        "}"
        f"QPushButton:hover {{ background-color: {border_hover}; }}"
    )
    btn.clicked.connect(on_clicked)
    return btn


# ─── Pop out ──────────────────────────────────────────────────────────


def pop_out(dock: Any) -> ChatPopoutWindow | None:
    """Pop the chat dock's inner content into a floating window.

    DbC:
        Pre: ``dock`` has been ``_setup_ui``-initialised
              (``dock.widget()`` returns the inner container).
        Post: returns a :class:`ChatPopoutWindow` whose ``content_widget``
              is the dock's former inner widget; the dock's
              ``.widget()`` returns ``None`` until redock fires.

    Idempotent — if a popout window is already live, re-uses it.
    """
    from ..chat_popout_window import ChatPopoutWindow

    existing = getattr(dock, "_popout_window", None)
    if existing is not None:
        try:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing  # type: ignore[no-any-return]
        except RuntimeError:
            dock._popout_window = None

    inner = dock.widget()
    if inner is None:
        log.warning("pop_out: dock has no inner widget; skipping")
        return None

    from .. import _chat_dock_widget_qt as _dock_mod

    session_id = (
        _dock_mod.ChatDockWidget._get_shared_session_id() or "new"  # type: ignore[attr-defined]
    )

    def _redock() -> None:
        dock._popout_window = None
        dock.show()

    # Pass the dock itself; ChatPopoutWindow will extract the inner widget.
    popout = ChatPopoutWindow(
        dock,
        session_id=session_id,
        redock_callback=_redock,
        title="AI Chat",
    )
    dock._popout_window = popout
    popout.resize(560, 720)
    popout.show()
    return popout


# ─── New chat ────────────────────────────────────────────────────────


def _clear_bubbles(dock: Any) -> None:
    """Remove every message bubble from the dock's message layout."""
    layout = dock._message_layout
    # Keep the trailing addStretch sentinel; remove everything before it.
    while layout.count() > 1:
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def new_chat(dock: Any) -> None:
    """Start a fresh server-side chat session.

    Side effects:
        * The visible bubble list is cleared.
        * The queued-message list is reset.
        * The streaming state is exited (cancels stop-state and chunk-flush
          timers and re-enables the Send button).
        * A WS ``new_session`` payload is dispatched so the server rotates
          the conversation context.

    DbC:
        Post: ``dock._queued_messages == []`` and
              ``dock.input_state == "idle"``.
    """
    _clear_bubbles(dock)
    dock._queued_messages.clear()
    if hasattr(dock, "_update_queue_affordance"):
        try:
            dock._update_queue_affordance()
        except Exception:  # noqa: BLE001
            log.debug("new_chat: queue affordance refresh failed", exc_info=True)
    if hasattr(dock, "_exit_thinking_state"):
        try:
            dock._exit_thinking_state()
        except Exception:  # noqa: BLE001
            log.debug("new_chat: exit_thinking_state failed", exc_info=True)
    dock._current_bubble = None
    dock._send_ws({"action": "new_session", "app_context": dock._app_context})
    if hasattr(dock, "_status_label"):
        try:
            dock._status_label.setText("New session started")
        except Exception:  # noqa: BLE001
            pass


# ─── Clear chat ──────────────────────────────────────────────────────


def confirm_clear_chat(dock: Any) -> bool:
    """Show a Yes/No confirmation modal. Returns True if the user accepts."""
    reply = QMessageBox.question(
        dock,
        "Clear chat?",
        "Clear visible messages?\nThe server-side session is preserved.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes


def clear_chat(dock: Any) -> None:
    """Wipe the visible bubble list. The server-side session is preserved.

    DbC:
        Pre: ``dock._message_layout`` exists.
        Post: if the user accepts the confirmation modal, the bubble list
              is empty and ``_send_ws`` is **not** invoked. The shared
              session id is unchanged.
    """
    accepted = dock._confirm_clear_chat()
    if not accepted:
        return
    _clear_bubbles(dock)


# ─── Model-refresh loading placeholder ───────────────────────────────


def set_model_combo_loading(dock: Any) -> None:
    """Put the model combo into an italic "Loading models..." holding state.

    Used while a provider switch is fetching the model list. The combo is
    disabled so the user cannot pick a stale value mid-refresh; callers
    are expected to follow up with ``_refresh_ai_model_combo`` which
    repopulates and re-enables the combo.
    """
    combo = dock._ai_model_combo
    combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItem("Loading models...", "")
    finally:
        combo.blockSignals(False)
    combo.setEnabled(False)


# ─── Terminal-as-chat send path ──────────────────────────────────────


def send_via_terminal_provider(dock: Any, text: str) -> None:
    """Route a chat-mode message through the terminal subprocess flow.

    When the user selects a CLI provider from the AI dropdown but stays in
    the regular "Chat" mode, the chat dock previously dispatched the
    message via ``_send_ws({"action": "send"})`` — which the server's
    chat router has no concept of for CLI providers, so nothing visible
    happened.

    This helper instead starts a terminal session (if necessary) for the
    selected provider and pushes the message text into it. It also adds
    a user bubble for visibility, so the conversation transcript still
    reflects what was typed.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("send_via_terminal_provider: text must be non-empty")
    text = text.strip()
    dock._add_bubble("user", text)
    # If no terminal session exists yet, ask the server to start one
    # using a default shell + the currently selected CLI provider id.
    if not dock._terminal_session_id and not dock._terminal_start_pending:
        provider_id = dock._current_provider
        shells = dock._terminal_registry.providers_for_shell(_pick_default_shell(dock))
        # Validate the chosen provider is registered; otherwise fall back
        # to surfacing a friendly error bubble.
        registered_ids = {p.id for p in shells}
        if provider_id not in registered_ids:
            dock._add_bubble(
                "assistant",
                f"[chat] no terminal provider registered for {provider_id!r}; "
                "please switch to an API provider or use Terminal mode",
            )
            return
        dock._terminal_start_pending = True
        dock._send_ws(
            {
                "action": "terminal_start",
                "project_root": str(dock._project_root),
                "shell_id": _pick_default_shell(dock),
                "provider_id": provider_id,
                "app_context": dock._app_context,
                "pending_input": f"{text}\n",
            }
        )
        return
    # Existing session — just push input.
    dock._send_ws(
        {
            "action": "terminal_input",
            "terminal_session_id": dock._terminal_session_id,
            "text": f"{text}\n",
        }
    )


def _pick_default_shell(dock: Any) -> str:
    """Return a sensible default shell id for the current platform."""
    import sys

    shells = list(dock._terminal_registry.shells())
    if not shells:
        return "bash"
    plat = sys.platform
    for s in shells:
        plats = getattr(s, "platforms", None) or []
        if plat in plats:
            return s.id
    # Fallback to the first registered shell.
    return shells[0].id
