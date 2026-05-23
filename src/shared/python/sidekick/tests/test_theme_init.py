import importlib
import sys
from unittest.mock import patch


def test_theme_init_imports() -> None:
    # We will reload the module to ensure the initialization runs during test
    import sidekick.theme as module_under_test

    importlib.reload(module_under_test)

    assert module_under_test._THEME_AVAILABLE is True
    assert "BUILTIN_THEMES" in module_under_test.__all__


def test_theme_init_no_theme() -> None:
    import sidekick.theme as module_under_test

    # Patch out all theme-related entries (and PyQt6) so the reload
    # sees a fully unavailable theme environment.  Simply setting
    # sys.modules["theme"] = None is not sufficient when sub-entries
    # such as "theme.manager" are already cached.
    theme_keys = {
        k: None for k in list(sys.modules) if k == "theme" or k.startswith("theme.")
    }
    theme_keys["PyQt6"] = None

    with patch.dict("sys.modules", theme_keys):
        importlib.reload(module_under_test)

    assert module_under_test._THEME_AVAILABLE is False
    assert module_under_test._PYQT6_AVAILABLE is False

    # Reload again with restored sys.modules — theme should re-load
    importlib.reload(module_under_test)
    assert module_under_test._THEME_AVAILABLE is True


def test_theme_init_sys_path() -> None:
    import sidekick.theme as module_under_test

    # _shared_python_dir is only set when `theme` was not importable at
    # module load time.  On CI (editable install), this attribute does not
    # exist because the import succeeded without path manipulation.
    if not hasattr(module_under_test, "_shared_python_dir"):
        return  # Nothing to test — theme was importable without path surgery.

    # Remove from sys.path
    path = str(module_under_test._shared_python_dir)
    if path in sys.path:
        sys.path.remove(path)

    # Reload should add it back
    importlib.reload(module_under_test)
    assert path in sys.path
