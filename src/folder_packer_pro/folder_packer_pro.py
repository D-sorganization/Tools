"""Folder Packer Pro v2.0 - Enhanced Professional Project Packing Tool.

A comprehensive project packaging application with advanced features:
- Modern themed UI with professional design
- Pack/Unpack folders into single encrypted archives
- AES-256 encryption for sensitive projects
- Multiple compression levels (store, fast, best)
- Git integration (preserve repository structure)
- Syntax highlighting in file preview
- Smart file filtering with customizable patterns
- Batch operations with progress tracking
- Export manifests and operation logs
- Professional error handling and validation

This module is a backward-compatible facade. The implementation has been
decomposed into focused submodules:
- constants.py: Configuration constants, themes, and logging setup
- encryption.py: AES-256 encryption/decryption via EncryptionManager
- manifest.py: Package manifest management
- file_ops.py: File scanning, statistics, exclusion patterns, type detection
- pack_engine.py: Core pack/unpack/inspect logic (UI-independent)
- ui_tabs.py: Tab UI creation mixins (Pack, Unpack, Preview, Log)
- dialogs.py: Dialog windows (exclusions, about, user guide)
- app.py: Main FolderPackerPro application class
"""

import logging
import sys
import tkinter as tk
from tkinter import messagebox

# Re-export all public API for backward compatibility
from .app import FolderPackerPro
from .constants import (  # noqa: F401
    CODE_EXTENSIONS,
    COMPRESSION_LEVELS,
    CONFIG_EXTENSIONS,
    DARK_THEME,
    DEFAULT_EXCLUDE_PATTERNS,
    LIGHT_THEME,
    MARKUP_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    PADDING_LARGE,
    PADDING_MEDIUM,
    PADDING_SMALL,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from .encryption import EncryptionManager  # noqa: F401
from .manifest import PackageManifest  # noqa: F401

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for Folder Packer Pro."""
    try:
        # Create root window
        root = tk.Tk()

        # Create application
        FolderPackerPro(root)

        # Start main loop
        root.mainloop()

    except (OSError, RuntimeError, ValueError, ImportError) as e:
        logger.exception("Fatal error in main application")
        messagebox.showerror("Fatal Error", f"Application failed to start:\n\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
