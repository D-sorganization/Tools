"""Lifetime and bound-slot regressions for CalculatorStateMixin (#5520)."""

from __future__ import annotations

import gc
import weakref
from unittest.mock import patch

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QMenu, QSplitter, QTextEdit
from sidekick.tests._calculator_state_helpers import MockCalculator


def test_calculator_freed_by_refcount_without_cyclic_gc(qapp) -> None:
    """Regression for #5520: signal lambdas must not pin the calculator.

    With the cyclic collector disabled, a self-capturing lambda connection
    keeps the calculator alive forever. Plain reference counting must free it.
    """
    gc.disable()
    try:
        calc = MockCalculator("LifetimeCalc")
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setParent(calc)
        calc.register_splitter(splitter, "lifetime_splitter")
        text_edit = QTextEdit()
        text_edit.setParent(calc)
        calc.register_copyable_widget(text_edit, "text")
        calc.setup_copy_paste()
        save_btn, load_btn = calc.create_save_load_buttons()
        with patch("PyQt6.QtWidgets.QMenu.exec"):
            calc.show_context_menu(QPoint(0, 0))
            calc.show_widget_context_menu(QPoint(0, 0), calc.copyable_widgets[0])

        calc_ref = weakref.ref(calc)
        del calc, splitter, text_edit, save_btn, load_btn
        assert calc_ref() is None
    finally:
        gc.enable()


def _trigger_menu_action(menu: QMenu, text: str) -> None:
    for action in menu.actions():
        if action.text() == text:
            action.trigger()


def test_context_menu_actions_route_through_bound_slots(qapp) -> None:
    """The per-action slots must reach the right target without lambdas (#5520)."""
    calc = MockCalculator()
    text_edit = QTextEdit()
    text_edit.setParent(calc)
    calc.register_copyable_widget(text_edit, "text")

    def _exec_copy_menu(menu: QMenu, *_args: object) -> None:
        _trigger_menu_action(menu, "Copy")

    def _exec_calculator_menu(menu: QMenu, *_args: object) -> None:
        _trigger_menu_action(menu, "Save State")

    with (
        patch.object(calc, "copy_widget_text") as copy_widget_text,
        patch("PyQt6.QtWidgets.QMenu.exec", new=_exec_copy_menu),
    ):
        calc.show_widget_context_menu(QPoint(0, 0), calc.copyable_widgets[0])
    copy_widget_text.assert_called_once_with(text_edit)

    with (
        patch.object(calc, "save_calculator_state") as save_state,
        patch("PyQt6.QtWidgets.QMenu.exec", new=_exec_calculator_menu),
    ):
        calc.show_context_menu(QPoint(0, 0))
    save_state.assert_called_once_with()
