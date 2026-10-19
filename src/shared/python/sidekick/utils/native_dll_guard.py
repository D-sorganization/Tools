"""Single bootstrap for importing ``ezc3d`` next to a native OpenSim SDK (#12065).

On Windows, importing OpenSim loads its bundled ``ezc3d.dll``.  Windows then
reuses that already-loaded, same-named DLL for the Python ``ezc3d`` extension,
which fails with ``ImportError: DLL load failed ... The specified procedure
could not be found`` (0xc0000139) because the two builds are not
ABI-compatible.  Import order therefore decides the winner and the two builds
cannot be reconciled in one process.

Supported runtime identity: a process either hosts the native OpenSim SDK or
reads C3D through ``ezc3d``.  C3D preparation belongs in a separate process
(``scripts/diagnostics/frozen_capture_trc.py``), whose source-bound TRC output
the native process consumes.  This module does not suppress the failure; it
turns it into a typed, diagnosable :class:`NativeDllConflictError`.

Every ``ezc3d`` import site should go through :func:`import_ezc3d`.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from types import ModuleType

logger = logging.getLogger(__name__)

EZC3D_DLL = "ezc3d.dll"
_DLL_FAILURE_MARKER = "DLL load failed"


class NativeDllConflictError(ImportError):
    """``ezc3d`` cannot load because another native build owns ``ezc3d.dll``."""


def is_ezc3d_available() -> bool:
    """Return True if ezc3d is available for import without known DLL conflict.

    Uses ``importlib.util.find_spec`` to check on-disk availability without
    loading native DLLs. Returns False if opensim is already loaded.
    """
    if "opensim" in sys.modules:
        return False
    try:
        return importlib.util.find_spec("ezc3d") is not None
    except (ImportError, ValueError):
        return False


def loaded_dll_path(dll_name: str) -> str | None:
    """Return the full path of ``dll_name`` if loaded in this process, else None.

    Always None off Windows (no per-process DLL name collision applies).

    Raises:
        ValueError: If ``dll_name`` is empty.
    """
    if not dll_name or not dll_name.strip():
        raise ValueError("dll_name must be a non-empty string")
    if os.name != "nt":
        return None
    return _windows_module_path(dll_name)


def _windows_module_path(dll_name: str) -> str | None:
    """Ask kernel32 for the loaded module's path (Windows only)."""
    # WinDLL exists only on Windows; attribute lookup keeps Linux type checking
    # honest while the os.name dispatch above keeps runtime use platform-correct.
    kernel32 = vars(ctypes)["WinDLL"]("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleFileNameW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    handle = kernel32.GetModuleHandleW(dll_name)
    if not handle:
        return None
    buffer = ctypes.create_unicode_buffer(1024)
    if not kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
        return None
    return buffer.value


def import_ezc3d(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Import and return the ``ezc3d`` module, diagnosing native DLL conflicts.

    Raises:
        NativeDllConflictError: ``opensim`` is already loaded in this process,
            or the import failed with a DLL load error while a native SDK owns
            ``ezc3d.dll``.  The message names the conflict and remedy.
        ModuleNotFoundError: ``ezc3d`` is not installed (unchanged).
        ImportError: any other import failure, unchanged.
    """
    opensim_loaded = "opensim" in sys.modules
    owner = loaded_dll_path(EZC3D_DLL)
    if opensim_loaded:
        message = (
            "ezc3d cannot load in this process: opensim is already loaded. "
            f"{EZC3D_DLL} is already loaded from {owner or 'an unknown location'} "
            "(opensim is imported). "
            "The native OpenSim and Python ezc3d builds are not ABI-compatible "
            "and cannot share a process. Read C3D in a separate process "
            "(scripts/diagnostics/frozen_capture_trc.py) and feed the "
            "source-bound TRC to the native process, or import ezc3d in a "
            "process that never imports opensim."
        )
        logger.error(message)
        raise NativeDllConflictError(message)

    try:
        return importer("ezc3d")
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError):
            raise
        if _DLL_FAILURE_MARKER not in str(exc):
            raise
        if owner is None and not opensim_loaded:
            raise
        message = (
            f"ezc3d cannot load in this process: {exc}. "
            f"{EZC3D_DLL} is already loaded from {owner or 'an unknown location'}"
            f"{' (opensim is imported)' if opensim_loaded else ''}. "
            "The native OpenSim and Python ezc3d builds are not ABI-compatible "
            "and cannot share a process. Read C3D in a separate process "
            "(scripts/diagnostics/frozen_capture_trc.py) and feed the "
            "source-bound TRC to the native process, or import ezc3d in a "
            "process that never imports opensim."
        )
        logger.error(message)
        raise NativeDllConflictError(message) from exc
