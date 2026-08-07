"""Compact-window regressions for the native Simulation and Swing workspace."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QScrollArea,
)

from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402
from rate_of_closure.ui.pyqt6.responsive_layout import (  # noqa: E402
    HeightForWidthGroupBox,
)
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COMPACT_SIMULATION_SIZE = (900, 640)
COMPACT_SWING_SIZE = (600, 560)
MINIMUM_PLOT_SIZE = (360, 280)


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = SimulationTab()
    qtbot.addWidget(widget)
    widget.resize(*COMPACT_SIMULATION_SIZE)
    widget.show()
    qtbot.wait(20)
    yield widget
    widget.stop()


def test_control_column_never_uses_nested_horizontal_scrolling(tab) -> None:  # type: ignore[no-untyped-def]
    scroll = tab._controls_scroll
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert scroll.horizontalScrollBar().maximum() == 0
    assert tab._run_status.width() <= scroll.viewport().width()
    assert tab._run_status.height() > 0


def test_wrapped_setup_groups_reserve_readable_editor_heights(tab) -> None:  # type: ignore[no-untyped-def]
    root = tab._controls_scroll.widget()
    assert root is not None
    for group in root.findChildren(HeightForWidthGroupBox):
        layout = group.layout()
        assert layout is not None
        assert group.height() >= layout.heightForWidth(group.width())

    for widget_type in (QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit):
        for widget in root.findChildren(widget_type):
            if widget.isVisible():
                if isinstance(widget, QLineEdit) and isinstance(
                    widget.parentWidget(), QAbstractSpinBox
                ):
                    continue
                assert widget.height() >= 16


def test_swing_controls_collapse_into_readable_panels_at_compact_size(
    tab, qtbot
) -> None:  # type: ignore[no-untyped-def]
    view = tab.view()
    view.resize(*COMPACT_SWING_SIZE)
    qtbot.wait(20)

    assert view._layers_button.text().startswith("Display & Layers")
    assert not view._layers_panel.isVisible()
    assert view._impact_summary.isVisible()
    assert not view._impact_details_scroll.isVisible()
    assert view._canvas.minimumWidth() >= MINIMUM_PLOT_SIZE[0]
    assert view._canvas.minimumHeight() >= MINIMUM_PLOT_SIZE[1]

    view._layers_button.click()
    qtbot.wait(10)
    assert view._layers_panel.isVisible()
    for checkbox in view._layer_checkboxes():
        assert checkbox.width() >= checkbox.minimumSizeHint().width()


def test_engineering_details_are_scrollable_and_key_metrics_remain_visible(
    tab, qtbot
) -> None:  # type: ignore[no-untyped-def]
    assert tab.run_now() is not None
    view = tab.view()
    assert "Contact AoA" in view._impact_summary.text()
    assert "Contact-Point AoA" in view._impact_kinematics_readout.text()
    assert not view._impact_details_scroll.isVisible()

    view._details_button.click()
    qtbot.wait(10)
    assert view._impact_details_scroll.isVisible()
    assert view._impact_details_scroll.maximumHeight() <= 160
    assert view._impact_details_scroll.horizontalScrollBar().maximum() == 0


def test_legend_defaults_outside_and_can_move_or_hide(tab, qtbot) -> None:  # type: ignore[no-untyped-def]
    assert tab.run_now() is not None
    view = tab.view()
    view._layers_button.click()
    qtbot.wait(10)

    legend = view._axes.get_legend()
    assert legend is not None
    assert view._legend_position.currentData() == "outside_right"
    assert legend.get_bbox_to_anchor()._bbox.x0 >= 1.0
    assert view._axes.get_position().x1 <= 0.75

    view._legend_position.setCurrentIndex(
        view._legend_position.findData("inside_lower_left")
    )
    qtbot.wait(10)
    assert view._axes.get_legend() is not None
    assert view._axes.get_legend()._loc == 3  # Matplotlib lower-left location

    view._legend_check.setChecked(False)
    qtbot.wait(10)
    assert view._axes.get_legend() is None


@pytest.mark.parametrize("window_size", [(1269, 731), (1280, 768)])
def test_full_window_simulation_controls_stay_inside_viewport(
    qtbot, window_size: tuple[int, int]
) -> None:  # type: ignore[no-untyped-def]
    window = RateOfClosureMainWindow()
    qtbot.addWidget(window)
    window.resize(*window_size)
    simulation_index = next(
        index
        for index in range(window._tabs.count())
        if window._tabs.tabBar().tabData(index) == "simulation"
    )
    window._tabs.setCurrentIndex(simulation_index)
    window._simulation_tab._display_tabs.setCurrentWidget(window._simulation_tab.view())
    window.show()
    qtbot.wait(30)

    tab = window._simulation_tab
    view = tab.view()
    assert window.size().width() == window_size[0]
    assert window.size().height() == window_size[1]
    assert window.devicePixelRatioF() >= 1.0
    assert tab._controls_scroll.horizontalScrollBar().maximum() == 0
    global_controls_scroll = next(
        scroll
        for scroll in window.findChildren(QScrollArea)
        if scroll.widget() is not None
        and scroll.widget().isAncestorOf(window._controls)
    )
    assert global_controls_scroll.horizontalScrollBar().maximum() == 0
    assert view._canvas.width() >= MINIMUM_PLOT_SIZE[0]
    assert view._canvas.height() >= MINIMUM_PLOT_SIZE[1]
    for widget_type in (QAbstractButton, QComboBox):
        for widget in view.findChildren(widget_type):
            if not widget.isVisible():
                continue
            top_left = widget.mapTo(view, widget.rect().topLeft())
            bottom_right = widget.mapTo(view, widget.rect().bottomRight())
            assert view.rect().contains(top_left)
            assert view.rect().contains(bottom_right)

    window._club_view.stop()
    tab.stop()
