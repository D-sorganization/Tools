import sys
from pathlib import Path

import pytest

# Ensure tools package is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from tools.icon_utils import convert_png_to_ico, ensure_pil_installed


@pytest.fixture
def source_png() -> Path:
    # Assuming tests are run from repo root
    path = Path("assets/tools_icon.png")
    if not path.exists():
        # Fallback if running from proper location ?
        path = repo_root / "assets" / "tools_icon.png"

    if not path.exists():
        pytest.skip(f"Source icon not found at {path}")
    return path


def test_convert_simple_ico(source_png: Path, tmp_path: Path) -> None:
    """Test converting to a simple 32x32 ICO."""
    output_path = tmp_path / "test_simple.ico"
    simple_size = [(32, 32)]

    ensure_pil_installed()
    result = convert_png_to_ico(source_png, output_path, sizes=simple_size)

    assert result is True
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_convert_multi_size_ico(source_png: Path, tmp_path: Path) -> None:
    """Test converting to a multi-size ICO."""
    output_path = tmp_path / "test_multi.ico"
    alt_sizes = [(256, 256), (64, 64), (32, 32), (16, 16)]

    result = convert_png_to_ico(source_png, output_path, sizes=alt_sizes)

    assert result is True
    assert output_path.exists()
    # Should be larger than simple icon
    assert output_path.stat().st_size > 100
