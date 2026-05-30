"""Robust settings panel and descriptor for the Sidekick **Chat** tab.

Historically the sidebar's top ``⚙`` button was a dead control while the
Chat tab was active: the Chat :class:`SidebarTabDefinition` declared no
``settings`` descriptor, so :meth:`open_active_tab_settings` returned
``False`` and the button was disabled by ``_refresh_settings_button``.

This module supplies the missing piece — :data:`CHAT_TAB_SETTINGS` — a
:class:`SidebarTabSettingsDescriptor` whose ``widget_factory`` builds a
real configuration surface for the embedded chat (provider, model,
reasoning level, agent mode, auto-condense threshold) plus secure API-key
management via :class:`chat.credentials.CredentialManager`.

Design notes:

* **DRY** — non-secret preferences flow through the already-validated
  :class:`SidebarTabSettingsStore` (the host's ``tab_settings`` /
  ``update_tab_settings`` API); secrets flow through the existing keyring-
  backed credential manager. No new persistence layer is introduced.
* **DbC** — public constructors validate their inputs; tolerant coercion
  (:func:`coerce_chat_settings`) clamps/falls back on stale persisted
  values so a hand-edited or out-of-date state never crashes the panel.
* **LOD** — live application only ever calls the chat dock's own public
  :meth:`switch_provider`; it never reaches through the sidebar into dock
  internals. The dock is located through a single narrow accessor.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

from .qt_compat import QtWidgets
from .settings import SidebarTabSettingsDescriptor, SidebarTabSettingsSchema

__all__ = [
    "CHAT_AGENT_MODES",
    "CHAT_PROVIDERS",
    "CHAT_SETTINGS_SCHEMA_VERSION",
    "CHAT_TAB_ID",
    "CHAT_TAB_SETTINGS",
    "CHAT_TAB_SETTINGS_DEFAULTS",
    "CHAT_THINKING_LEVELS",
    "ChatSettingsPanel",
    "apply_chat_settings_to_dock",
    "build_chat_settings_panel",
    "chat_settings_defaults",
    "coerce_chat_settings",
    "credential_status",
]

_logger = logging.getLogger(__name__)

CHAT_TAB_ID = "chat"
CHAT_SETTINGS_SCHEMA_VERSION = 1

# Allowed value domains. Kept local (rather than imported from the optional
# ``chat`` package) so the descriptor is importable on headless hosts that
# never install the chat extras.
CHAT_PROVIDERS: tuple[str, ...] = (
    "ollama",
    "openai",
    "anthropic",
    "gemini",
    "cline",
    "codex",
)
CHAT_THINKING_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high")
CHAT_AGENT_MODES: tuple[str, ...] = ("agent", "plan", "ask")

# Auto-condense threshold guard rails (approximate tokens).
_MIN_CONDENSE_THRESHOLD = 500
_MAX_CONDENSE_THRESHOLD = 1_000_000
_DEFAULT_CONDENSE_THRESHOLD = 8000

CHAT_TAB_SETTINGS_DEFAULTS: dict[str, Any] = {
    "provider": "ollama",
    "model": "llama3",
    "thinking_level": "none",
    "agent_mode": "agent",
    "auto_condense_threshold": _DEFAULT_CONDENSE_THRESHOLD,
}

_ALLOWED_KEYS = frozenset(CHAT_TAB_SETTINGS_DEFAULTS)


# ─── Pure helpers (headless-testable) ────────────────────────────


def chat_settings_defaults() -> dict[str, Any]:
    """Return a deep copy of the chat settings defaults."""
    return copy.deepcopy(CHAT_TAB_SETTINGS_DEFAULTS)


def _clamp_threshold(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_CONDENSE_THRESHOLD
    return max(_MIN_CONDENSE_THRESHOLD, min(_MAX_CONDENSE_THRESHOLD, number))


def _choose(value: Any, allowed: tuple[str, ...], default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


def coerce_chat_settings(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return validated chat settings merged over the defaults.

    Tolerant by design: unknown keys are dropped and invalid values fall
    back to their default, so stale or hand-edited persisted state never
    raises. This is the single normalization point used on both load and
    save (DRY).

    Args:
        values: Raw settings mapping (or ``None``).

    Returns:
        A complete, JSON-safe settings dict with every known key present.

    Raises:
        TypeError: If ``values`` is provided but is not a mapping.
    """
    if values is None:
        return chat_settings_defaults()
    if not isinstance(values, Mapping):
        raise TypeError("chat settings values must be a mapping or None")

    result = chat_settings_defaults()
    result["provider"] = _choose(
        values.get("provider"), CHAT_PROVIDERS, result["provider"]
    )
    model = values.get("model")
    if isinstance(model, str) and model.strip():
        result["model"] = model.strip()
    result["thinking_level"] = _choose(
        values.get("thinking_level"), CHAT_THINKING_LEVELS, result["thinking_level"]
    )
    result["agent_mode"] = _choose(
        values.get("agent_mode"), CHAT_AGENT_MODES, result["agent_mode"]
    )
    result["auto_condense_threshold"] = _clamp_threshold(
        values.get("auto_condense_threshold")
    )
    return result


def apply_chat_settings_to_dock(dock: Any, values: Mapping[str, Any]) -> bool:
    """Apply provider/model/reasoning selections to a live chat dock.

    Uses only the dock's public :meth:`switch_provider` — no reach into
    dock internals (LOD). Non-provider preferences (agent mode,
    auto-condense threshold) are persisted by the caller and honored by
    the dock on its next build, so they are intentionally not forced here.

    Args:
        dock: The live chat dock widget, or ``None``.
        values: Settings mapping to apply.

    Returns:
        ``True`` if the provider switch was applied, ``False`` otherwise.
    """
    if dock is None:
        return False
    switch = getattr(dock, "switch_provider", None)
    if not callable(switch):
        return False
    settings = coerce_chat_settings(values)
    try:
        switch(
            settings["provider"],
            settings["model"],
            settings["thinking_level"],
        )
    except Exception:  # noqa: BLE001 - live apply is best-effort
        _logger.debug("Live chat provider switch failed", exc_info=True)
        return False
    return True


def credential_status(
    manager: Any,
    providers: tuple[str, ...] = CHAT_PROVIDERS,
) -> dict[str, bool]:
    """Return ``{provider: has_key}`` for ``providers`` via ``manager``.

    Args:
        manager: A credential manager exposing ``has_credentials`` (or
            ``None`` — every provider then reports ``False``).
        providers: Providers to query.

    Returns:
        Mapping of provider name to whether a credential is configured.
    """
    if manager is None:
        return dict.fromkeys(providers, False)
    status: dict[str, bool] = {}
    for provider in providers:
        try:
            status[provider] = bool(manager.has_credentials(provider))
        except Exception:  # noqa: BLE001 - a flaky backend must not break the UI
            _logger.debug("Credential probe failed for %s", provider, exc_info=True)
            status[provider] = False
    return status


def _default_credential_manager() -> Any | None:
    """Return a :class:`CredentialManager`, or ``None`` if chat is absent."""
    try:
        from chat.credentials import CredentialManager
    except Exception as exc:  # noqa: BLE001 - chat extras are optional
        _logger.debug("Chat credentials unavailable: %s", exc)
        return None
    return CredentialManager(service_name="sidekick_chat_ai")


# ─── Settings descriptor ─────────────────────────────────────────

CHAT_TAB_SETTINGS_SCHEMA = SidebarTabSettingsSchema(
    version=CHAT_SETTINGS_SCHEMA_VERSION,
    defaults=copy.deepcopy(CHAT_TAB_SETTINGS_DEFAULTS),
    allowed_keys=_ALLOWED_KEYS,
)


def build_chat_settings_panel(sidebar: Any, tab_id: str) -> QtWidgets.QWidget:
    """Widget factory wired into :data:`CHAT_TAB_SETTINGS`.

    Matches the ``Callable[[sidebar, tab_id], QWidget]`` contract that
    :meth:`open_active_tab_settings` invokes when the gear is clicked.
    """
    return ChatSettingsPanel(sidebar, tab_id)


CHAT_TAB_SETTINGS = SidebarTabSettingsDescriptor(
    schema=CHAT_TAB_SETTINGS_SCHEMA,
    widget_factory=build_chat_settings_panel,
)


# ─── Qt panel ────────────────────────────────────────────────────


class ChatSettingsPanel(QtWidgets.QWidget):
    """Configuration surface for the embedded Sidekick chat.

    Reads current values from the host's validated settings store, lets the
    user edit model/behavior preferences and per-provider API keys, then
    persists preferences back through the store and (best-effort) applies
    the provider selection to the live chat dock.

    Args:
        sidebar: Host sidebar exposing ``tab_settings`` /
            ``update_tab_settings`` and (optionally) ``chat_dock_widget``.
        tab_id: The tab id these settings belong to (normally ``"chat"``).
        credential_manager: Optional injected manager (defaults to a
            keyring-backed :class:`CredentialManager`; ``None`` when the
            chat extras are not installed).
        parent: Optional Qt parent.

    Raises:
        TypeError: If ``sidebar`` is ``None``.
        ValueError: If ``tab_id`` is empty.
    """

    def __init__(
        self,
        sidebar: Any,
        tab_id: str = CHAT_TAB_ID,
        *,
        credential_manager: Any | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if sidebar is None:
            raise TypeError("sidebar must be provided")
        if not isinstance(tab_id, str) or not tab_id.strip():
            raise ValueError("tab_id must be a non-empty string")
        super().__init__(parent)
        self.setObjectName("SidekickChatSettingsPanel")
        self._sidebar = sidebar
        self._tab_id = tab_id
        self._credential_manager = (
            credential_manager
            if credential_manager is not None
            else _default_credential_manager()
        )
        self._build_ui()
        self._load_current()
        self._refresh_key_status()

    # -- construction ------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_behavior_group())
        layout.addWidget(self._build_credentials_group())

        self._status_label = QtWidgets.QLabel("", self)
        self._status_label.setObjectName("SidekickChatSettingsStatus")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        button_row = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset to Defaults", self)
        reset_btn.setObjectName("SidekickChatSettingsReset")
        reset_btn.clicked.connect(self._on_reset)
        button_row.addWidget(reset_btn)
        button_row.addStretch(1)
        save_btn = QtWidgets.QPushButton("Save", self)
        save_btn.setObjectName("SidekickChatSettingsSave")
        save_btn.clicked.connect(self._on_save)
        button_row.addWidget(save_btn)
        layout.addLayout(button_row)

    def _build_behavior_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Model & Behavior", self)
        form = QtWidgets.QFormLayout(group)

        self._provider_combo = QtWidgets.QComboBox(group)
        self._provider_combo.setObjectName("SidekickChatProvider")
        self._provider_combo.addItems(list(CHAT_PROVIDERS))
        form.addRow("Provider", self._provider_combo)

        self._model_input = QtWidgets.QLineEdit(group)
        self._model_input.setObjectName("SidekickChatModel")
        self._model_input.setPlaceholderText("Model identifier (e.g. llama3)")
        form.addRow("Model", self._model_input)

        self._thinking_combo = QtWidgets.QComboBox(group)
        self._thinking_combo.setObjectName("SidekickChatThinking")
        self._thinking_combo.addItems(list(CHAT_THINKING_LEVELS))
        form.addRow("Reasoning level", self._thinking_combo)

        self._agent_combo = QtWidgets.QComboBox(group)
        self._agent_combo.setObjectName("SidekickChatAgentMode")
        self._agent_combo.addItems(list(CHAT_AGENT_MODES))
        form.addRow("Default mode", self._agent_combo)

        self._threshold_spin = QtWidgets.QSpinBox(group)
        self._threshold_spin.setObjectName("SidekickChatCondenseThreshold")
        self._threshold_spin.setRange(_MIN_CONDENSE_THRESHOLD, _MAX_CONDENSE_THRESHOLD)
        self._threshold_spin.setSingleStep(500)
        self._threshold_spin.setSuffix(" tok")
        self._threshold_spin.setToolTip(
            "Auto-condense the thread once its approximate token count "
            "exceeds this value."
        )
        form.addRow("Auto-condense at", self._threshold_spin)

        return group

    def _build_credentials_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("API Keys", self)
        form = QtWidgets.QFormLayout(group)

        self._key_provider_combo = QtWidgets.QComboBox(group)
        self._key_provider_combo.setObjectName("SidekickChatKeyProvider")
        self._key_provider_combo.addItems(list(CHAT_PROVIDERS))
        self._key_provider_combo.currentIndexChanged.connect(self._refresh_key_status)
        form.addRow("Provider", self._key_provider_combo)

        self._key_input = QtWidgets.QLineEdit(group)
        self._key_input.setObjectName("SidekickChatKeyInput")
        self._key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("Paste API key to store in the OS keyring")
        form.addRow("API key", self._key_input)

        key_row = QtWidgets.QHBoxLayout()
        save_key_btn = QtWidgets.QPushButton("Save Key", group)
        save_key_btn.setObjectName("SidekickChatKeySave")
        save_key_btn.clicked.connect(self._on_save_key)
        key_row.addWidget(save_key_btn)
        clear_key_btn = QtWidgets.QPushButton("Clear Key", group)
        clear_key_btn.setObjectName("SidekickChatKeyClear")
        clear_key_btn.clicked.connect(self._on_clear_key)
        key_row.addWidget(clear_key_btn)
        key_row.addStretch(1)
        form.addRow("", self._wrap_row(key_row, group))

        self._key_status_label = QtWidgets.QLabel("", group)
        self._key_status_label.setObjectName("SidekickChatKeyStatus")
        if self._credential_manager is None:
            self._key_status_label.setText(
                "Credential storage unavailable (install the chat extras "
                "and 'keyring' to manage API keys)."
            )
        form.addRow("", self._key_status_label)

        return group

    @staticmethod
    def _wrap_row(
        inner: QtWidgets.QHBoxLayout, parent: QtWidgets.QWidget
    ) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget(parent)
        holder.setLayout(inner)
        return holder

    # -- state <-> widgets ------------------------------------------

    def _load_current(self) -> None:
        values = coerce_chat_settings(self._current_values())
        self._set_combo(self._provider_combo, values["provider"])
        self._model_input.setText(values["model"])
        self._set_combo(self._thinking_combo, values["thinking_level"])
        self._set_combo(self._agent_combo, values["agent_mode"])
        self._threshold_spin.setValue(int(values["auto_condense_threshold"]))

    def _current_values(self) -> Mapping[str, Any]:
        getter = getattr(self._sidebar, "tab_settings", None)
        if not callable(getter):
            return {}
        try:
            payload = getter(self._tab_id)
        except Exception:  # noqa: BLE001 - degrade to defaults on store error
            _logger.debug("Reading chat tab settings failed", exc_info=True)
            return {}
        if isinstance(payload, Mapping):
            raw = payload.get("values", {})
            if isinstance(raw, Mapping):
                return raw
        return {}

    @staticmethod
    def _set_combo(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def collect(self) -> dict[str, Any]:
        """Return the current widget selections as a coerced settings dict."""
        return coerce_chat_settings(
            {
                "provider": self._provider_combo.currentText(),
                "model": self._model_input.text(),
                "thinking_level": self._thinking_combo.currentText(),
                "agent_mode": self._agent_combo.currentText(),
                "auto_condense_threshold": self._threshold_spin.value(),
            }
        )

    # -- dock lookup (single narrow accessor) -----------------------

    def _locate_dock(self) -> Any | None:
        accessor = getattr(self._sidebar, "chat_dock_widget", None)
        if callable(accessor):
            try:
                return accessor()
            except Exception:  # noqa: BLE001 - never let lookup break Save
                _logger.debug("chat_dock_widget() lookup failed", exc_info=True)
        return None

    # -- handlers ----------------------------------------------------

    def _on_save(self) -> None:
        values = self.collect()
        persisted = self._persist(values)
        applied = apply_chat_settings_to_dock(self._locate_dock(), values)
        if not persisted:
            self._status_label.setText("Could not persist chat settings.")
            return
        suffix = " Applied to active chat." if applied else ""
        self._status_label.setText("Chat settings saved." + suffix)

    def _persist(self, values: Mapping[str, Any]) -> bool:
        setter = getattr(self._sidebar, "update_tab_settings", None)
        if not callable(setter):
            return False
        try:
            setter(self._tab_id, dict(values))
        except Exception:  # noqa: BLE001 - report failure to the user
            _logger.debug("Persisting chat tab settings failed", exc_info=True)
            return False
        return True

    def _on_reset(self) -> None:
        defaults = chat_settings_defaults()
        self._set_combo(self._provider_combo, defaults["provider"])
        self._model_input.setText(defaults["model"])
        self._set_combo(self._thinking_combo, defaults["thinking_level"])
        self._set_combo(self._agent_combo, defaults["agent_mode"])
        self._threshold_spin.setValue(int(defaults["auto_condense_threshold"]))
        self._status_label.setText("Reset to defaults — click Save to apply.")

    def _selected_key_provider(self) -> str:
        return str(self._key_provider_combo.currentText())

    def _on_save_key(self) -> None:
        if self._credential_manager is None:
            return
        provider = self._selected_key_provider()
        key = self._key_input.text().strip()
        if not key:
            self._key_status_label.setText("Enter an API key before saving.")
            return
        try:
            stored = bool(self._credential_manager.store_api_key(provider, key))
        except Exception:  # noqa: BLE001 - surface backend failure to the user
            _logger.debug("Storing API key failed", exc_info=True)
            stored = False
        self._key_input.clear()
        if stored:
            self._key_status_label.setText(f"Stored API key for {provider}.")
        else:
            self._key_status_label.setText(f"Could not store API key for {provider}.")

    def _on_clear_key(self) -> None:
        if self._credential_manager is None:
            return
        provider = self._selected_key_provider()
        try:
            self._credential_manager.delete_api_key(provider)
        except Exception:  # noqa: BLE001 - deletion is best-effort
            _logger.debug("Deleting API key failed", exc_info=True)
        self._refresh_key_status(message=f"Cleared stored key for {provider}.")

    def _refresh_key_status(self, *_args: Any, message: str | None = None) -> None:
        if self._credential_manager is None:
            return
        provider = self._selected_key_provider()
        status = credential_status(self._credential_manager, (provider,))
        state = "configured" if status.get(provider) else "not configured"
        prefix = f"{message} " if message else ""
        self._key_status_label.setText(f"{prefix}{provider}: {state}.")
