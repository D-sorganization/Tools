"""Notification delivery remains stable when theme subscriptions change."""

import pytest
from plot_theme.manager import PlotThemeManager
from plot_theme.themes import PlotTheme


def test_self_unsubscribe_does_not_skip_next_subscriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every initial subscriber receives the current theme exactly once."""
    monkeypatch.setattr(PlotThemeManager, "_load_saved_theme", lambda self: None)
    manager = PlotThemeManager()
    events: list[str] = []

    def one_shot(theme: PlotTheme) -> None:
        events.append("one-shot")
        manager.remove_theme_change_callback(one_shot)

    def persistent(theme: PlotTheme) -> None:
        events.append(theme.name)

    manager.add_theme_change_callback(one_shot)
    manager.add_theme_change_callback(persistent)
    manager.set_theme("vampire_dark", save=False)
    assert events == ["one-shot", "Vampire Dark"]
    manager.set_theme("classic_light", save=False)
    assert events == ["one-shot", "Vampire Dark", "Classic Light"]


def test_new_subscriber_starts_with_next_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback registered during dispatch waits for the next change."""
    monkeypatch.setattr(PlotThemeManager, "_load_saved_theme", lambda self: None)
    manager = PlotThemeManager()
    events: list[str] = []

    def added(theme: PlotTheme) -> None:
        events.append(theme.name)

    def register(theme: PlotTheme) -> None:
        manager.add_theme_change_callback(added)

    manager.add_theme_change_callback(register)
    manager.set_theme("vampire_dark", save=False)
    assert events == []
    manager.set_theme("classic_light", save=False)
    assert events == ["Classic Light"]
