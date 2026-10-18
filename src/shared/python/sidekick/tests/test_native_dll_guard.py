"""Tests for the shared ezc3d native-DLL guard and its c3d_reader routing.

Refs D-sorganization/UpstreamDrift issue 12065.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from sidekick.lab.bio import c3d_reader as c3d_reader_module
from sidekick.lab.bio.c3d_reader import C3DDataReader
from sidekick.utils import native_dll_guard
from sidekick.utils.native_dll_guard import (
    NativeDllConflictError,
    import_ezc3d,
    is_ezc3d_available,
)

DLL_ERROR = ImportError(
    "DLL load failed while importing _ezc3d: "
    "The specified procedure could not be found."
)


def _raiser(exc: BaseException) -> Callable[[str], ModuleType]:
    def _importer(_name: str) -> ModuleType:
        raise exc

    return _importer


def test_returns_module_when_import_succeeds() -> None:
    sentinel = ModuleType("ezc3d")
    assert import_ezc3d(lambda _name: sentinel) is sentinel


def test_module_not_found_is_unchanged() -> None:
    with pytest.raises(ModuleNotFoundError):
        import_ezc3d(_raiser(ModuleNotFoundError("No module named 'ezc3d'")))


def test_non_dll_import_error_is_unchanged() -> None:
    err = ImportError("some other failure")
    with pytest.raises(ImportError) as info:
        import_ezc3d(_raiser(err))
    assert info.value is err
    assert not isinstance(info.value, NativeDllConflictError)


def test_dll_failure_with_loaded_dll_raises_diagnostic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        native_dll_guard, "loaded_dll_path", lambda _n: "C:/os/ezc3d.dll"
    )
    with caplog.at_level(logging.ERROR), pytest.raises(NativeDllConflictError) as info:
        import_ezc3d(_raiser(DLL_ERROR))
    assert "C:/os/ezc3d.dll" in str(info.value)
    assert "separate process" in str(info.value)
    assert info.value.__cause__ is DLL_ERROR


def test_dll_failure_with_opensim_imported_raises_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_dll_guard, "loaded_dll_path", lambda _n: None)
    monkeypatch.setitem(sys.modules, "opensim", ModuleType("opensim"))
    with pytest.raises(NativeDllConflictError, match="opensim is imported"):
        import_ezc3d(_raiser(DLL_ERROR))


def test_dll_failure_without_native_owner_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_dll_guard, "loaded_dll_path", lambda _n: None)
    monkeypatch.delitem(sys.modules, "opensim", raising=False)
    with pytest.raises(ImportError) as info:
        import_ezc3d(_raiser(DLL_ERROR))
    assert not isinstance(info.value, NativeDllConflictError)


def test_loaded_dll_path_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        native_dll_guard.loaded_dll_path("  ")


def test_c3d_reader_has_no_direct_ezc3d_import() -> None:
    """The reader must obtain ezc3d only through the guard."""
    tree = ast.parse(Path(c3d_reader_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "ezc3d" for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "ezc3d"
    assert "import_ezc3d" in vars(c3d_reader_module)


def test_load_surfaces_native_conflict_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c3d_path = tmp_path / "x.c3d"
    c3d_path.write_bytes(b"\x02\x50")
    conflict = NativeDllConflictError("ezc3d cannot load in this process")
    monkeypatch.setattr(c3d_reader_module, "ezc3d", None)
    monkeypatch.setattr(c3d_reader_module, "_EZC3D_IMPORT_ERROR", conflict)
    with pytest.raises(NativeDllConflictError) as info:
        C3DDataReader(c3d_path)._load()
    assert info.value is conflict


def test_load_plain_missing_keeps_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c3d_path = tmp_path / "x.c3d"
    c3d_path.write_bytes(b"\x02\x50")
    monkeypatch.setattr(c3d_reader_module, "ezc3d", None)
    monkeypatch.setattr(c3d_reader_module, "_EZC3D_IMPORT_ERROR", None)
    with pytest.raises(ImportError, match="pip install ezc3d"):
        C3DDataReader(c3d_path)._load()


def test_is_ezc3d_available_find_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_ezc3d_available uses find_spec and checks opensim in sys.modules."""
    monkeypatch.delitem(sys.modules, "opensim", raising=False)
    # When find_spec returns a mock spec
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "ezc3d" else None,
    )
    assert is_ezc3d_available() is True

    # When opensim is loaded, is_ezc3d_available returns False
    monkeypatch.setitem(sys.modules, "opensim", ModuleType("opensim"))
    assert is_ezc3d_available() is False

    # When opensim is not loaded and find_spec returns None
    monkeypatch.delitem(sys.modules, "opensim", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    assert is_ezc3d_available() is False


def test_import_ezc3d_raises_early_when_opensim_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When opensim is loaded, import_ezc3d raises without calling the importer."""
    monkeypatch.setitem(sys.modules, "opensim", ModuleType("opensim"))
    importer_called = False

    def dummy_importer(_name: str) -> ModuleType:
        nonlocal importer_called
        importer_called = True
        return ModuleType("ezc3d")

    with pytest.raises(NativeDllConflictError, match="opensim is already loaded"):
        import_ezc3d(dummy_importer)
    assert not importer_called


def test_import_sidekick_lab_bio_does_not_eagerly_import_ezc3d() -> None:
    """Subprocess test: importing sidekick.lab.bio does not put ezc3d in sys.modules."""
    shared_python_root = Path(__file__).resolve().parent.parent.parent
    with tempfile.TemporaryDirectory() as td:
        # Create a dummy ezc3d module in the temporary directory
        (Path(td) / "ezc3d.py").write_text("# dummy ezc3d\n", encoding="utf-8")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        paths = [td, str(shared_python_root)]
        if existing_pythonpath:
            paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(paths)

        code = (
            "import sys\n"
            "import sidekick.lab.bio\n"
            "assert 'ezc3d' not in sys.modules, 'ezc3d was eagerly imported'\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, f"Subprocess failed:\n{proc.stderr}\n{proc.stdout}"
