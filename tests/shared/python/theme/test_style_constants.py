from src.shared.python.theme.style_constants import Styles


def test_status_and_button_constants_keep_expected_qss_fragments() -> None:
    assert Styles.STATUS_SUCCESS == "color: #30D158;"
    assert Styles.STATUS_ERROR_BOLD == "color: #FF375F; font-weight: bold;"
    assert "QPushButton:hover" in Styles.BTN_PRIMARY
    assert "background-color: #0A84FF" in Styles.BTN_SEND
    assert "QPlainTextEdit" in Styles.CONSOLE_DARK
    assert "font-family" in Styles.CONSOLE_DARK


def test_color_swatch_helper_uses_rgb_tuple_and_border() -> None:
    style = Styles.color_swatch(12, 34, 56)

    assert style == "background-color: rgb(12,34,56); border: 1px solid #555;"


def test_status_chip_helper_formats_background_text_and_shape() -> None:
    style = Styles.status_chip("#112233", "#ffffff")

    assert "background-color: #112233" in style
    assert "color: #ffffff" in style
    assert "padding: 2px 6px" in style
    assert "border-radius: 4px" in style


def test_text_helper_styles_remain_composable() -> None:
    assert Styles.colored_bold("#abcdef") == "color: #abcdef; font-weight: bold;"
    assert Styles.no_image_label("#999") == (
        "QLabel { color: #999; font-style: italic; "
        "border: none; background: transparent; }"
    )
