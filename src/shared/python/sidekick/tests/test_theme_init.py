import importlib
import os
import subprocess
import sys
import textwrap


def test_theme_init_imports() -> None:
    # We will reload the module to ensure the initialization runs during test
    import sidekick.theme as module_under_test

    importlib.reload(module_under_test)

    assert module_under_test._THEME_AVAILABLE is True
    assert "BUILTIN_THEMES" in module_under_test.__all__


def test_theme_init_no_theme() -> None:
    code = """
        import builtins
        import importlib
        import sys
        from unittest.mock import patch

        import sidekick.theme as module_under_test

        original_import = builtins.__import__

        def fail_theme_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "theme":
                raise ImportError("theme disabled for test")
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch.dict(sys.modules, {"theme": None}),
            patch.object(builtins, "__import__", side_effect=fail_theme_import),
        ):
            importlib.reload(module_under_test)

        assert module_under_test._THEME_AVAILABLE is False
        assert module_under_test._PYQT6_AVAILABLE is False
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=os.getcwd(),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_theme_init_sys_path() -> None:
    code = """
        import importlib
        import sys
        from unittest.mock import patch

        import sidekick.theme as module_under_test

        path = str(
            module_under_test.Path(module_under_test.__file__)
            .resolve()
            .parent
            .parent
            .parent
        )
        if path in sys.path:
            sys.path.remove(path)

        with patch.dict(sys.modules, {"theme": None}):
            importlib.reload(module_under_test)

        assert path in sys.path
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=os.getcwd(),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
