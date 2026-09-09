"""Issue #3992 regression guard: apps must not hand-copy the Catppuccin palette.

The canonical palette lives in ``shared.python.theme.catppuccin`` (plus
``shared/design_tokens.json``).  Hand-copied hex-literal stylesheets drift the
moment the canonical palette changes, so the sites named by #3992 must source
their colors from the canonical module instead of pasting literals.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# (repo-relative module, minimum palette literals that used to be pasted there)
_GUARDED_SITES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "src/function_generator/python/function_generator/ui/pyqt6/main_window.py",
        ("#1e1e2e", "#cdd6f4", "#89b4fa", "#45475a", "#313244"),
    ),
    (
        "src/pressure_drop_calculator/python/pressure_drop_calculator/ui/pyqt6/main_window.py",
        ("#1e1e2e", "#cdd6f4", "#89b4fa", "#45475a", "#313244"),
    ),
    (
        "src/steam_engine_calculator/python/steam_engine_calculator/ui/pyqt6/main_window.py",
        ("#1e1e2e", "#cdd6f4", "#89b4fa", "#45475a", "#313244"),
    ),
    (
        "src/financial_calculator/python/financial_calculator/ui/pyqt6/main_window.py",
        ("#1e1e2e", "#cdd6f4", "#89b4fa", "#45475a", "#313244"),
    ),
    (
        "src/asteroid_jumper/renderer.py",
        ("#1e1e2e", "#cdd6f4", "#89b4fa", "#45475a", "#313244"),
    ),
    (
        "src/python/src/help/help_system.py",
        ("#1e1e2e", "#cdd6f4", "#89b4fa", "#45475a", "#313244"),
    ),
)


@pytest.mark.parametrize(("relative_path", "forbidden_hexes"), _GUARDED_SITES)
def test_no_hand_copied_catppuccin_literals(
    relative_path: str, forbidden_hexes: tuple[str, ...]
) -> None:
    """Each guarded site must source its palette from the canonical module."""
    module_path = _REPO_ROOT / relative_path
    assert module_path.exists(), (
        f"guarded site missing (re-point this test): {relative_path}"
    )
    source = module_path.read_text(encoding="utf-8")
    pasted = [hex_value for hex_value in forbidden_hexes if hex_value in source]
    assert not pasted, (
        f"{relative_path} hand-copies Catppuccin literals {pasted}; "
        "import from shared.python.theme.catppuccin (issue #3992)"
    )
