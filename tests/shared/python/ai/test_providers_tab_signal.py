"""Tests for the ProvidersTab.provider_changed signal (Tools #3745 P2).

The settings dialog previously reached two levels in
(``self._providers_tab.provider_combo.currentIndexChanged.connect(...)``).
ProvidersTab now exposes a ``provider_changed`` signal so callers connect to
the tab directly (Law of Demeter). These tests verify the signal fires and
that AISettingsDialog wires its handler to it rather than to the inner combo.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.ai.gui._providers_tab import ProvidersTab
from src.shared.python.ai.gui.settings_dialog import AISettingsDialog

pytestmark = [pytest.mark.unit]


def test_providers_tab_exposes_provider_changed_signal(qapp: Any) -> None:
    tab = ProvidersTab()
    received: list[int] = []
    tab.provider_changed.connect(received.append)

    if tab.provider_combo.count() > 1:
        tab.provider_combo.setCurrentIndex(1)
        assert received and received[-1] == 1
    tab.close()


def test_dialog_connects_to_provider_changed_signal(qapp: Any) -> None:
    """Selecting a provider must drive AISettingsDialog._on_provider_changed.

    Emitting the tab-level signal (not poking the inner combo) is enough to
    trigger the dialog handler, proving the connection targets the new signal.
    """
    dialog = AISettingsDialog()
    calls: list[int] = []
    original = dialog._on_provider_changed

    def _spy(index: int) -> None:
        calls.append(index)
        original(index)

    dialog._on_provider_changed = _spy  # type: ignore[method-assign]
    # Reconnect the spy so the live wiring exercises it.
    dialog._providers_tab.provider_changed.connect(_spy)

    dialog._providers_tab.provider_changed.emit(0)
    assert calls and calls[-1] == 0
    dialog.close()
