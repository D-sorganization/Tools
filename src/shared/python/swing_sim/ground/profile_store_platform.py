"""Small platform primitives for the ground profile store."""

from __future__ import annotations

import ctypes
import os
import stat
from pathlib import Path


def is_link_like(path: Path) -> bool:
    """Detect symbolic links and Windows reparse points without following."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse)


def validated_digest(value: str, name: str) -> str:
    """Return one lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def validated_filename(filename: str) -> str:
    """Reject path syntax, aliases, controls, and Win32 special filenames."""
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
    }
    reserved |= {f"{prefix}{index}" for prefix in ("COM", "LPT") for index in "¹²³"}
    stem = filename.split(".", 1)[0].rstrip(" .").upper()
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
        or filename.endswith((".", " "))
        or any(character in '<>:"/\\|?*' for character in filename)
        or any(ord(character) < 0x20 for character in filename)
        or stem in reserved
    ):
        raise ValueError("filename must be a safe plain filename")
    return filename


def atomic_replace(source: Path, destination: Path) -> None:
    """Atomically replace, requesting write-through rename semantics on Windows."""
    if os.name != "nt":
        os.replace(source, destination)
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file = kernel32.MoveFileExW
    move_file.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    move_file.restype = ctypes.c_int
    replace_existing = 0x1
    write_through = 0x8
    if not move_file(str(source), str(destination), replace_existing | write_through):
        error = ctypes.get_last_error()
        raise OSError(error, "MoveFileExW atomic write-through replacement failed")


__all__ = [
    "atomic_replace",
    "is_link_like",
    "validated_digest",
    "validated_filename",
]
