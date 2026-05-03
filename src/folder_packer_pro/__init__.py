"""Folder Packer Pro package.

Enhanced professional project packing tool with AES-256 encryption,
multiple compression levels, and syntax-highlighted previews.
"""

from .encryption import EncryptionManager
from .manifest import PackageManifest

__all__ = [
    "EncryptionManager",
    "FolderPackerPro",
    "PackageManifest",
]


def __getattr__(name: str) -> object:
    """Load the Tkinter app only when callers explicitly request it."""
    if name == "FolderPackerPro":
        from .app import FolderPackerPro

        return FolderPackerPro
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
