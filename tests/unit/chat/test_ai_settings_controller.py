# ruff: noqa: E501
"""Headless tests for ``AiSettingsController`` (ADR-0022 / issue #6119).

No ``QApplication`` and no widget instantiation — the controller is driven
through a fake view that records refresh / sync / persist calls and holds the
``current_*`` state, mirroring how ``ChatDockWidget`` exposes it. This sidesteps
the Sidekick multi-widget Qt segfault.
"""

from __future__ import annotations

import pytest

from chat.ai_settings_controller import (
    VALID_FIELDS,
    VALID_THINKING_NAMES,
    AiSettingsController,
)


class _FakeView:
    """Minimal AiSettingsView: holds state + records driven hooks."""

    def __init__(self) -> None:
        self.current_provider = "ollama"
        self.current_model = "llama3"
        self.current_thinking_level = "none"
        self.calls: list[str] = []

    def refresh_model_combo(self) -> None:
        self.calls.append("refresh_model_combo")

    def refresh_thinking_combo(self) -> None:
        self.calls.append("refresh_thinking_combo")

    def sync_ai_view(self) -> None:
        self.calls.append("sync_ai_view")

    def persist_ai_settings(self) -> None:
        self.calls.append("persist_ai_settings")


def test_view_required() -> None:
    with pytest.raises(TypeError):
        AiSettingsController(None)  # type: ignore[arg-type]


def test_vocabulary_constants() -> None:
    assert VALID_FIELDS == {"provider", "model", "thinking"}
    assert VALID_THINKING_NAMES == {"none", "low", "medium", "high"}


def test_apply_provider_change_refreshes_both_and_persists() -> None:
    view = _FakeView()
    AiSettingsController(view).apply_settings_change("provider", "  openai  ")
    assert view.current_provider == "openai"  # stripped
    assert view.calls == [
        "refresh_model_combo",
        "refresh_thinking_combo",
        "persist_ai_settings",
    ]


def test_apply_model_change_refreshes_thinking_only() -> None:
    view = _FakeView()
    AiSettingsController(view).apply_settings_change("model", "gpt-4")
    assert view.current_model == "gpt-4"
    assert view.calls == ["refresh_thinking_combo", "persist_ai_settings"]


def test_apply_thinking_change_persists_only() -> None:
    view = _FakeView()
    AiSettingsController(view).apply_settings_change("thinking", "high")
    assert view.current_thinking_level == "high"
    assert view.calls == ["persist_ai_settings"]


def test_apply_rejects_unknown_field() -> None:
    view = _FakeView()
    with pytest.raises(ValueError, match="unknown field"):
        AiSettingsController(view).apply_settings_change("color", "red")
    assert view.calls == []


@pytest.mark.parametrize("bad", ["", "   ", 123, None])
def test_apply_rejects_empty_value(bad: object) -> None:
    view = _FakeView()
    with pytest.raises(ValueError, match="must be non-empty"):
        AiSettingsController(view).apply_settings_change("provider", bad)  # type: ignore[arg-type]


def test_switch_provider_updates_state_and_syncs() -> None:
    view = _FakeView()
    AiSettingsController(view).switch_provider("anthropic", "claude-3", "medium")
    assert view.current_provider == "anthropic"
    assert view.current_model == "claude-3"
    assert view.current_thinking_level == "medium"
    assert view.calls == ["sync_ai_view"]


def test_switch_provider_strips_inputs() -> None:
    view = _FakeView()
    AiSettingsController(view).switch_provider("  openai ", " gpt-4 ", " low ")
    assert (view.current_provider, view.current_model, view.current_thinking_level) == (
        "openai",
        "gpt-4",
        "low",
    )


@pytest.mark.parametrize(
    ("name", "model", "level", "match"),
    [
        ("", "m", "none", "name must be non-empty"),
        ("p", "  ", "none", "model must be non-empty"),
        ("p", "m", "loud", "not in"),
        ("p", "m", 5, "must be a string"),
    ],
)
def test_switch_provider_rejects_bad_inputs(
    name: object, model: object, level: object, match: str
) -> None:
    view = _FakeView()
    with pytest.raises((ValueError, TypeError), match=match):
        AiSettingsController(view).switch_provider(name, model, level)  # type: ignore[arg-type]
    assert view.calls == []
