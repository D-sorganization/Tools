"""Headless-import guard for the ``ui`` package (issue #3187).

``ui/auto_complete`` and ``ui/hover_copy_browser`` import PyQt6 at module
top level. Importing the ``ui`` package must not hard-fail in a headless
environment without PyQt6 (mirroring ``theme/__init__``); the Qt widget
names must be present but ``None`` when PyQt6 is absent.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_ui_imports_without_pyqt6() -> None:
    """Importing ``ui`` succeeds with PyQt6 forced absent; widgets are None."""
    script = textwrap.dedent(
        """
        import sys
        import importlib.abc

        # Force PyQt6 (and submodules) to be unimportable, mimicking headless.
        class _BlockPyQt6(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "PyQt6" or name.startswith("PyQt6."):
                    raise ImportError(
                        f"PyQt6 blocked for headless test: {name}"
                    )
                return None

        for _mod in list(sys.modules):
            if _mod == "PyQt6" or _mod.startswith("PyQt6."):
                del sys.modules[_mod]
        sys.meta_path.insert(0, _BlockPyQt6())

        import ui

        assert ui._PYQT6_AVAILABLE is False
        assert ui.AutoCompleteLineEdit is None
        assert ui.HoverCopyTextBrowser is None
        assert "AutoCompleteLineEdit" in ui.__all__
        assert "HoverCopyTextBrowser" in ui.__all__
        print("HEADLESS_UI_IMPORT_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=_repo_src_dir(),
    )
    assert result.returncode == 0, (
        f"headless ui import failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "HEADLESS_UI_IMPORT_OK" in result.stdout


def _repo_src_dir() -> str:
    """Return the ``src/shared/python`` dir so ``import ui`` resolves."""
    from pathlib import Path

    here = Path(__file__).resolve()
    # tests/shared/python/ui/<file> -> repo root is parents[4]
    repo_root = here.parents[4]
    return str(repo_root / "src" / "shared" / "python")
