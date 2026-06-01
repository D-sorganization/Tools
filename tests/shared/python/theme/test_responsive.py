"""Regression tests for shared responsive PyQt sizing helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from shared.python.theme.responsive import (
    TextWidthSpec,
    configure_form_layout_for_readability,
    derive_text_candidates,
    readable_text_width,
    set_text_minimum_width,
    wrap_in_scroll_area,
)


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def test_readable_text_width_accounts_for_padding_and_chrome(
    qapp: QApplication,
) -> None:
    metrics = QFontMetrics(qapp.font())
    spec = TextWidthSpec(padding_px=12, chrome_px=18, minimum_px=80)

    width = readable_text_width(metrics, ["Short", "Much longer filter label"], spec)

    expected = metrics.horizontalAdvance("Much longer filter label") + 30
    assert width == max(80, expected)


def test_readable_text_width_rejects_invalid_contract(qapp: QApplication) -> None:
    metrics = QFontMetrics(qapp.font())

    with pytest.raises(ValueError, match="padding_px"):
        readable_text_width(metrics, ["Filter"], TextWidthSpec(padding_px=-1))

    with pytest.raises(ValueError, match="maximum_px"):
        readable_text_width(
            metrics,
            ["Filter"],
            TextWidthSpec(minimum_px=120, maximum_px=80),
        )

    with pytest.raises(ValueError, match="at least one"):
        readable_text_width(metrics, [], TextWidthSpec())


def test_readable_text_width_honors_maximum(qapp: QApplication) -> None:
    metrics = QFontMetrics(qapp.font())

    width = readable_text_width(
        metrics,
        ["A very long label that should be clamped"],
        TextWidthSpec(minimum_px=20, maximum_px=90),
    )

    assert width == 90


def test_set_text_minimum_width_preserves_combo_text(qapp: QApplication) -> None:
    combo = QComboBox()
    combo.addItems(["All", "Proprietary (In-House)", "External Database"])

    width = set_text_minimum_width(combo, TextWidthSpec(padding_px=16, chrome_px=34))

    assert combo.minimumWidth() == width
    assert width >= combo.fontMetrics().horizontalAdvance("Proprietary (In-House)")
    assert combo.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.MinimumExpanding


def test_derive_text_candidates_reads_common_widgets(qapp: QApplication) -> None:
    combo = QComboBox()
    combo.addItems(["One", "Two"])
    edit = QLineEdit()
    edit.setPlaceholderText("Search feedstocks")
    edit_with_text = QLineEdit()
    edit_with_text.setText("actual query")
    button = QPushButton("Load Database")
    generic = QWidget()
    generic.setToolTip("Fallback tooltip")

    assert derive_text_candidates(combo) == ["One", "Two"]
    assert derive_text_candidates(edit) == ["Search feedstocks"]
    assert derive_text_candidates(edit_with_text) == ["actual query"]
    assert derive_text_candidates(button) == ["Load Database"]
    assert derive_text_candidates(generic) == ["Fallback tooltip"]


def test_form_layout_uses_wrapping_growth_policy(qapp: QApplication) -> None:
    layout = QFormLayout()

    configure_form_layout_for_readability(layout)

    assert layout.rowWrapPolicy() == QFormLayout.RowWrapPolicy.WrapLongRows
    assert (
        layout.fieldGrowthPolicy() == QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
    )


def test_wrap_in_scroll_area_keeps_content_resizable(qapp: QApplication) -> None:
    widget = QWidget()

    scroll = wrap_in_scroll_area(widget, minimum_width=320)

    assert scroll.widget() is widget
    assert scroll.widgetResizable()
    assert scroll.minimumWidth() == 320


def test_wrap_in_scroll_area_accepts_zero_minimum_width(qapp: QApplication) -> None:
    widget = QWidget()

    scroll = wrap_in_scroll_area(widget)

    assert scroll.widget() is widget
    assert scroll.widgetResizable()


def test_wrap_in_scroll_area_rejects_negative_minimum_width(qapp: QApplication) -> None:
    with pytest.raises(ValueError, match="minimum_width"):
        wrap_in_scroll_area(QWidget(), minimum_width=-1)
