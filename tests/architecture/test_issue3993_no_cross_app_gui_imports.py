"""Issue #3993 regression guard: no cross-app GUI imports.

``signal_processing_studio`` must not import another app's GUI internals
(``function_generator``).  The shared FunctionGeneratorWidget lives in
``shared.python.ui`` and is imported explicitly — an import failure must
surface as an error, not as a silently missing tab behind
``except ImportError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SPS_MAIN_WINDOW = (
    _REPO_ROOT
    / "src/signal_processing_studio/python/signal_processing_studio/main_window.py"
)
_SHARED_WIDGET = _REPO_ROOT / "src/shared/python/ui/function_generator_widget.py"


def test_sps_does_not_import_function_generator() -> None:
    """SPS must not reach across the app boundary into function_generator."""
    source = _SPS_MAIN_WINDOW.read_text(encoding="utf-8")
    assert "from function_generator" not in source, (
        "signal_processing_studio imports function_generator GUI internals "
        "(issue #3993); import the shared widget from shared.python.ui instead"
    )
    assert "HAS_FUNC_GEN" not in source, (
        "silent feature-degradation flag must not gate the shared widget "
        "(issue #3993); import it explicitly so a rename fails loudly"
    )


def test_shared_widget_module_exists() -> None:
    """The canonical FunctionGeneratorWidget lives in shared.python.ui."""
    assert _SHARED_WIDGET.exists(), (
        "FunctionGeneratorWidget must live at src/shared/python/ui/"
        "function_generator_widget.py (issue #3993)"
    )
    widget_source = _SHARED_WIDGET.read_text(encoding="utf-8")
    assert "class FunctionGeneratorWidget" in widget_source


@pytest.mark.parametrize(
    ("registration_path",),
    [("src/function_generator/gui_registration.py",)],
)
def test_app_registration_points_at_shared_widget(registration_path: str) -> None:
    """The function_generator app must register the shared widget module."""
    source = (_REPO_ROOT / registration_path).read_text(encoding="utf-8")
    assert "shared.python.ui.function_generator_widget" in source, (
        f"{registration_path} must register the shared widget module (issue #3993)"
    )
    assert "function_generator.python.function_generator.ui.pyqt6" not in source, (
        f"{registration_path} still registers the deleted cross-app module"
    )
