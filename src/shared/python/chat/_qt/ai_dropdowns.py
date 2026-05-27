# ruff: noqa: E501
"""AI provider / model / thinking dropdown helpers (Tools issue #2871).

Extracted from ``_chat_dock_widget_qt`` so the parent module fits the
1500-line budget. The :class:`ChatDockWidget` retains the public methods
(``switch_provider``, ``_build_ai_dropdowns``, etc.) but their bodies
delegate to the free functions defined here.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QSizePolicy

from ..cli_provider_availability import list_available_cli_providers

logger = logging.getLogger(__name__)

VALID_THINKING_NAMES: frozenset[str] = frozenset({"none", "low", "medium", "high"})
VALID_FIELDS: frozenset[str] = frozenset({"provider", "model", "thinking"})
DEFAULT_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("Ollama", "ollama"),
    ("OpenAI", "openai"),
    ("Anthropic", "anthropic"),
    ("Gemini", "gemini"),
    ("Cline", "cline"),
)


def build_header_combobox(
    *,
    label: str,
    items: list[tuple[str, str]],
) -> QComboBox:
    """Build a header combo box used by the AI Provider/Model/Thinking row.

    DRY helper used for all three header dropdowns (issue #2871).

    Args:
        label: Short label (e.g. ``"provider"``) used for tool-tip; must
            be non-empty.
        items: Sequence of ``(display_text, user_data)`` pairs;
            must be non-empty.

    Raises:
        ValueError: If ``label`` is empty/whitespace or ``items`` is empty.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError("build_header_combobox: label must be non-empty")
    if not items:
        raise ValueError("build_header_combobox: items must be non-empty")
    combo = QComboBox()
    combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    combo.setMinimumWidth(0)
    for display, data in items:
        combo.addItem(display, data)
    combo.setToolTip(f"Select AI {label}")
    return combo


def build_available_cli_provider_items() -> list[tuple[str, str]]:
    """Return ``(display_name, provider_id)`` pairs for installed CLI agents.

    Probes the local ``PATH`` so the dropdown never shows unavailable
    entries.
    """
    return [
        (entry.display_name, entry.provider_id)
        for entry in list_available_cli_providers()
    ]


def build_ai_dropdowns(dock: Any, mode_row: QHBoxLayout) -> None:
    """Construct + wire the three AI header dropdowns on ``dock``.

    Side-effect only: instantiates ``_ai_provider_combo``, ``_ai_model_combo``,
    ``_ai_thinking_combo`` on ``dock`` and inserts them into ``mode_row``.
    """
    api_items = list(DEFAULT_PROVIDERS)
    cli_items = build_available_cli_provider_items()
    all_provider_items = api_items + cli_items
    dock._ai_provider_combo = build_header_combobox(
        label="provider", items=all_provider_items
    )
    mode_row.addWidget(dock._ai_provider_combo)

    dock._ai_model_combo = build_header_combobox(
        label="model", items=[("(default)", "default")]
    )
    mode_row.addWidget(dock._ai_model_combo)

    dock._ai_thinking_combo = build_header_combobox(
        label="thinking", items=[("Off", "none")]
    )
    mode_row.addWidget(dock._ai_thinking_combo)

    dock._ai_provider_combo.currentIndexChanged.connect(
        lambda _: dock._on_ai_combo_changed("provider")
    )
    dock._ai_model_combo.currentIndexChanged.connect(
        lambda _: dock._on_ai_combo_changed("model")
    )
    dock._ai_thinking_combo.currentIndexChanged.connect(
        lambda _: dock._on_ai_combo_changed("thinking")
    )
    dock._refresh_ai_model_combo()
    dock._refresh_ai_thinking_combo()
    dock._sync_ai_dropdowns()


def refresh_ai_model_combo(dock: Any) -> None:
    """Repopulate the model combo for the currently selected provider."""
    try:
        adapter = dock._get_active_ai_adapter()
        models = adapter.list_models() if adapter is not None else []
    except Exception:  # noqa: BLE001 - any adapter failure → empty list
        logger.debug("refresh_ai_model_combo: adapter probe failed", exc_info=True)
        models = []
    items = []
    for m in models:
        display = str(getattr(m, "display_name", None) or getattr(m, "name", str(m)))
        data = str(
            getattr(m, "id", None)
            or getattr(m, "model_id", None)
            or getattr(m, "name", str(m))
        )
        items.append((display, data))
    if not items:
        items = [("(default)", "default")]
    dock._ai_model_combo.blockSignals(True)
    try:
        dock._ai_model_combo.clear()
        for display, data in items:
            dock._ai_model_combo.addItem(display, data)
    finally:
        dock._ai_model_combo.blockSignals(False)


def refresh_ai_thinking_combo(dock: Any) -> None:
    """Repopulate the thinking combo for the currently selected adapter."""
    try:
        adapter = dock._get_active_ai_adapter()
        caps = adapter.thinking_capabilities() if adapter is not None else None
    except Exception:  # noqa: BLE001
        logger.debug("refresh_ai_thinking_combo: adapter probe failed", exc_info=True)
        caps = None
    if caps is None:
        items = [("Off", "none")]
    else:
        items = [
            (
                getattr(level, "label", str(level)),
                getattr(level, "name", str(level)),
            )
            for level in getattr(caps, "available_levels", getattr(caps, "levels", []))
        ]
    dock._ai_thinking_combo.blockSignals(True)
    try:
        dock._ai_thinking_combo.clear()
        for display, data in items:
            dock._ai_thinking_combo.addItem(display, data)
    finally:
        dock._ai_thinking_combo.blockSignals(False)


def sync_ai_dropdowns(dock: Any) -> None:
    """Push current state into the three combos with signals blocked."""
    for combo, value in (
        (dock._ai_provider_combo, dock._current_provider),
        (dock._ai_model_combo, dock._current_model),
        (dock._ai_thinking_combo, dock._current_thinking_level),
    ):
        combo.blockSignals(True)
        try:
            idx = combo.findData(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)


def get_active_ai_adapter(provider_name: str) -> Any | None:
    """Return the adapter for ``provider_name`` or ``None``.

    Adapter construction failures are non-fatal (offline mode, missing
    API key, etc.) — callers fall back to a static catalogue.
    """
    try:
        from src.shared.python.ai.adapters.factory import AdapterFactory

        return AdapterFactory.create(provider_name)
    except Exception:  # noqa: BLE001 - missing credentials are normal
        return None


def apply_settings_change(dock: Any, field: str, value: str) -> None:
    """Single change router for the three AI header dropdowns.

    DbC (issue #2871):
        Pre: ``field`` is exactly one of ``"provider"``, ``"model"``, or
             ``"thinking"``.
        Pre: ``value`` is a non-empty / non-whitespace string.
        Post: dependent combos are refreshed; settings are persisted via
              ``dock._persist_ai_settings``.
    """
    if field not in VALID_FIELDS:
        raise ValueError(
            f"apply_settings_change: unknown field {field!r}; expected "
            f"one of {sorted(VALID_FIELDS)!r}"
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"apply_settings_change: value for {field!r} must be non-empty"
        )
    value = value.strip()
    if field == "provider":
        dock._current_provider = value
        # Show a "Loading models..." placeholder + disable the combo so
        # the user cannot pick a stale model mid-refresh. The helper is
        # a no-op on dock objects that do not own the model combo yet
        # (very early init / trimmed test fixtures).
        if hasattr(dock, "_set_model_combo_loading"):
            try:
                dock._set_model_combo_loading()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "apply_settings_change: model-loading placeholder failed",
                    exc_info=True,
                )
        dock._refresh_ai_model_combo()
        # Re-enable the combo after the refresh repopulates it.
        try:
            dock._ai_model_combo.setEnabled(True)
        except AttributeError:
            pass
        dock._refresh_ai_thinking_combo()
    elif field == "model":
        dock._current_model = value
        dock._refresh_ai_thinking_combo()
    else:  # field == "thinking"
        dock._current_thinking_level = value
    dock._persist_ai_settings()


def switch_provider(
    dock: Any,
    name: str,
    model: str,
    thinking_level: str,
) -> None:
    """Switch AI provider / model / thinking-level mid-thread.

    DbC (Tools issue #2871):
        Pre: ``name``/``model`` are non-empty strings after ``.strip()``.
        Pre: ``thinking_level`` ∈ {none, low, medium, high} after ``.strip()``.
        Post: ``dock._current_provider``/``_current_model``/
              ``_current_thinking_level`` reflect the request.
        Post: ``dock._message_history`` is the same list object and same
              contents as before the call (history-immutability invariant).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("switch_provider: name must be non-empty")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("switch_provider: model must be non-empty")
    if not isinstance(thinking_level, str):
        raise ValueError("switch_provider: thinking_level must be a string")
    normalized_level = thinking_level.strip()
    if normalized_level not in VALID_THINKING_NAMES:
        raise ValueError(
            f"switch_provider: thinking_level {thinking_level!r} not in "
            f"{sorted(VALID_THINKING_NAMES)!r}"
        )
    history_before = dock._message_history
    snapshot_before = list(history_before)

    dock._current_provider = name.strip()
    dock._current_model = model.strip()
    dock._current_thinking_level = normalized_level

    if hasattr(dock, "_ai_provider_combo") and dock._ai_provider_combo is not None:
        dock._sync_ai_dropdowns()

    assert dock._message_history is history_before, (
        "switch_provider invariant: _message_history must remain the same list"
    )
    assert dock._message_history == snapshot_before, (
        "switch_provider invariant: _message_history contents must not change"
    )
