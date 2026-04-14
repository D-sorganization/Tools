"""SidePanelTabs — reusable tabbed container for SimulationPanel side panels.

Decouples panel composition from layout: every secondary panel (Setup,
Mass Matrix, Plots, Optimizer, Noise, Perturbation, …) is added through
a single ``add_panel(label, widget, *, tooltip="")`` entry point that:

- validates inputs (Design by Contract)
- wraps the panel in a QScrollArea so panels with tall content never clip
- enforces label uniqueness so saved-state restoration is unambiguous
- tracks insertion order so ``panel_labels()`` is reproducible
- persists the active tab to QSettings under a configurable key

Design by Contract
------------------
- Pre:  ``settings_key`` is non-empty (constructor)
- Pre:  ``label`` is non-empty (after strip), ``widget`` is not None,
        and ``label`` is not already in use (``add_panel``)
- Pre:  ``label`` exists (``set_active_tab`` / ``panel_widget``)
- Inv:  every tab widget is a ``QScrollArea`` wrapping the panel
- Inv:  ``len(self._labels) == self.count()`` after every mutating call
- Post: ``add_panel`` returns ``self.count() - 1`` (a 0-based index)

Law of Demeter
--------------
SimulationPanel only talks to SidePanelTabs through the public API
(``add_panel``, ``set_active_tab``, ``save_state``, ``restore_state``).
It never reaches into the internal QScrollArea or QTabWidget bookkeeping.

DRY
---
A single helper (``_wrap``) creates the QScrollArea wrapper, so the
border, scrollbar policy, and minimum-width handling are defined in
exactly one place.
"""

from __future__ import annotations

import logging
from typing import Final

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QScrollArea, QTabWidget, QWidget

logger = logging.getLogger(__name__)


_SCROLL_STYLE: Final[str] = "QScrollArea { border: none; background: transparent; }"
_QSETTINGS_ORG: Final[str] = "D-sorganization"
_QSETTINGS_APP: Final[str] = "PendulumSimulator"


class SidePanelTabs(QTabWidget):
    """Tabbed container for SimulationPanel side panels.

    Parameters
    ----------
    settings_key : str
        QSettings key (under org=D-sorganization, app=PendulumSimulator)
        used by ``save_state`` / ``restore_state`` to remember the
        active tab between sessions. Must be non-empty.
    parent : QWidget | None
        Optional parent widget.

    Raises
    ------
    ValueError
        If ``settings_key`` is empty or whitespace-only.
    """

    def __init__(
        self,
        settings_key: str,
        parent: QWidget | None = None,
    ) -> None:
        if not settings_key or not settings_key.strip():
            raise ValueError(f"settings_key must be a non-empty string, got {settings_key!r}")
        super().__init__(parent)
        self._settings_key: str = settings_key
        # Insertion-ordered: label → wrapped scroll area
        self._labels: list[str] = []
        self._panels: dict[str, QWidget] = {}  # label → original widget
        self.setDocumentMode(True)
        self.setUsesScrollButtons(True)
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setMovable(False)
        self.setMinimumWidth(280)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def add_panel(
        self,
        label: str,
        widget: QWidget,
        *,
        tooltip: str = "",
    ) -> int:
        """Add a new tab containing ``widget``.

        Parameters
        ----------
        label : str
            Tab title. Must be non-empty (after stripping whitespace) and
            must not duplicate an existing label.
        widget : QWidget
            The panel widget. Will be wrapped in a QScrollArea.
        tooltip : str, optional
            Tooltip shown on the tab header. Defaults to empty.

        Returns
        -------
        int
            The index of the newly added tab (0-based).

        Raises
        ------
        ValueError
            If ``label`` is empty/whitespace, ``widget`` is None, or the
            ``label`` is already in use.
        """
        if label is None or not label.strip():
            raise ValueError(f"label must be a non-empty string, got {label!r}")
        if widget is None:
            raise ValueError("widget must not be None")
        if label in self._panels:
            raise ValueError(f"duplicate label {label!r} — already used by another panel")

        wrapper = self._wrap(widget)
        index = self.addTab(wrapper, label)
        if tooltip:
            self.setTabToolTip(index, tooltip)
        self._labels.append(label)
        self._panels[label] = widget
        # Class invariant
        assert len(self._labels) == self.count()
        return index

    def panel_labels(self) -> list[str]:
        """Return the labels of every added panel, in insertion order."""
        return list(self._labels)

    def panel_widget(self, label: str) -> QWidget:
        """Return the original (un-wrapped) widget for the given label.

        Raises
        ------
        KeyError
            If ``label`` is not a current panel.
        """
        if label not in self._panels:
            raise KeyError(f"no panel labeled {label!r}; known labels: {self._labels}")
        return self._panels[label]

    def active_tab_label(self) -> str:
        """Return the label of the currently selected tab.

        Returns an empty string when no panels have been added.
        """
        idx = self.currentIndex()
        if idx < 0 or idx >= len(self._labels):
            return ""
        return self._labels[idx]

    def set_active_tab(self, label: str) -> None:
        """Switch the active tab to the panel with the given label.

        Raises
        ------
        KeyError
            If ``label`` is not a current panel.
        """
        if label not in self._panels:
            raise KeyError(f"no panel labeled {label!r}; known labels: {self._labels}")
        self.setCurrentIndex(self._labels.index(label))

    def save_state(self) -> None:
        """Persist the active tab label under the configured settings key."""
        active = self.active_tab_label()
        if not active:
            return
        QSettings(_QSETTINGS_ORG, _QSETTINGS_APP).setValue(self._settings_key, active)

    def restore_state(self) -> None:
        """Restore the active tab from QSettings, if a saved value exists.

        Falls back silently to the default selection if:
        - no value has been saved yet, or
        - the saved label no longer matches any current panel
          (e.g. a panel was removed in a newer build).
        """
        saved = QSettings(_QSETTINGS_ORG, _QSETTINGS_APP).value(self._settings_key)
        if saved is None:
            return
        try:
            saved_str = str(saved)
        except (TypeError, ValueError):
            return
        if saved_str in self._panels:
            self.set_active_tab(saved_str)
        else:
            logger.debug(
                "Saved tab %r not found in current panels %s — keeping default",
                saved_str,
                self._labels,
            )

    # ──────────────────────────────────────────────────────────────────
    # Internals (DRY: single source of wrapping behaviour)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(widget: QWidget) -> QScrollArea:
        """Wrap ``widget`` in a styled QScrollArea.

        Centralised so every panel gets identical scroll/border behaviour.
        """
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet(_SCROLL_STYLE)
        return scroll
