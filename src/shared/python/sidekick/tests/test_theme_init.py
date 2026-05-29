import builtins
import importlib
import runpy
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch


@contextmanager
def _block_theme_import() -> Iterator[None]:
    """Force ``import theme`` to fail regardless of the harness import hook.

    The test harness installs a meta-path redirector that aliases ``theme`` to
    its loaded sibling modules, so blocking ``sys.modules`` keys is ineffective
    (the redirector re-binds the name). Patching ``builtins.__import__`` instead
    intercepts the ``import`` statement before any finder runs, which reliably
    drives the module's ``except ImportError`` fallback branches.
    """
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "theme" or name.startswith("theme."):
            raise ImportError("theme import blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    with patch("builtins.__import__", side_effect=fake_import):
        yield


def _run_theme_module() -> dict[str, Any]:
    """Re-run the ``sidekick.theme`` module body in a fresh namespace.

    ``importlib.reload`` does not route the re-executed import statements
    through a patched ``builtins.__import__``, so ``runpy.run_path`` is used
    instead: it runs the source in the current interpreter (honouring the
    patch) and leaves the real, already-imported module untouched for other
    tests.
    """
    import sidekick.theme as module_under_test

    return runpy.run_path(module_under_test.__file__)


def test_theme_init_imports() -> None:
    # We will reload the module to ensure the initialization runs during test
    import sidekick.theme as module_under_test

    importlib.reload(module_under_test)

    assert module_under_test._THEME_AVAILABLE is True
    assert "BUILTIN_THEMES" in module_under_test.__all__


def test_theme_init_no_theme() -> None:
    with _block_theme_import():
        namespace = _run_theme_module()

    assert namespace["_THEME_AVAILABLE"] is False
    assert namespace["_PYQT6_AVAILABLE"] is False


def test_theme_init_sys_path() -> None:
    import sidekick.theme as module_under_test

    # Remove the fallback search path so the re-run must add it back.
    path = str(module_under_test._shared_python_dir)
    if path in sys.path:
        sys.path.remove(path)

    try:
        # With ``theme`` unimportable, the fallback branch re-inserts the
        # shared-python directory onto sys.path.
        with _block_theme_import():
            _run_theme_module()
        assert path in sys.path
    finally:
        if path not in sys.path:
            sys.path.insert(0, path)
