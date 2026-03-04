"""TDD suite for icon_utils.py.

Tests cover check_pil_installed, create_resized_images sizes validation,
convert_png_to_ico path handling, and DbC contract violations.
"""

from unittest.mock import MagicMock

import pytest

from src.shared.python.contracts import PreconditionError
from src.tools.icon_utils import (
    ICO_SIZES,
    check_pil_installed,
    convert_png_to_ico,
    create_resized_images,
)


def test_check_pil_installed_returns_bool():
    result = check_pil_installed()
    assert isinstance(result, bool)


def test_create_resized_images_dbc_empty_sizes():
    """create_resized_images must reject empty size list."""
    mock_img = MagicMock()
    with pytest.raises(PreconditionError):
        create_resized_images(mock_img, [])


def test_create_resized_images_dbc_non_list():
    """create_resized_images must reject non-list sizes."""
    mock_img = MagicMock()
    with pytest.raises(PreconditionError):
        create_resized_images(mock_img, "not_a_list")  # type: ignore[arg-type]


def test_convert_png_to_ico_dbc_non_path_input(tmp_path):
    """convert_png_to_ico must reject string paths."""
    with pytest.raises(PreconditionError):
        convert_png_to_ico("input.png", tmp_path / "out.ico")  # type: ignore[arg-type]


def test_convert_png_to_ico_dbc_non_path_output(tmp_path):
    """convert_png_to_ico must reject string output paths."""
    with pytest.raises(PreconditionError):
        convert_png_to_ico(tmp_path / "in.png", "out.ico")  # type: ignore[arg-type]


def test_convert_png_to_ico_missing_file_returns_false(tmp_path):
    """Missing PNG returns False gracefully (no crash)."""
    result = convert_png_to_ico(tmp_path / "missing.png", tmp_path / "out.ico")
    assert result is False


@pytest.mark.skipif(not check_pil_installed(), reason="PIL not installed")
def test_convert_png_to_ico_success(tmp_path):
    """Integration test: converts a real PNG when PIL is available."""
    from PIL import Image

    # Create a minimal PNG
    png = tmp_path / "test.png"
    ico = tmp_path / "test.ico"
    img = Image.new("RGBA", (256, 256), color=(255, 0, 0, 255))
    img.save(png, format="PNG")

    result = convert_png_to_ico(png, ico, sizes=[(16, 16), (32, 32)])
    assert result is True
    assert ico.exists()


def test_ico_sizes_constant():
    """Verify ICO_SIZES constant contains valid tuples."""
    assert isinstance(ICO_SIZES, list)
    assert len(ICO_SIZES) >= 4
    for size in ICO_SIZES:
        assert len(size) == 2
        assert all(isinstance(dim, int) and dim > 0 for dim in size)
