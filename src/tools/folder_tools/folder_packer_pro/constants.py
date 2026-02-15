"""Constants for Folder Packer Pro.

Centralizes all configuration constants including compression levels,
UI dimensions, file extension categories, exclusion patterns, and
color themes.
"""

from __future__ import annotations

import logging
from typing import Final

# Package size limits
MAX_FILE_SIZE_MB: Final[int] = 1024  # 1GB max per file

# Compression configuration
COMPRESSION_LEVELS: Final[dict[str, int]] = {
    "none": 0,
    "fast": 1,
    "balanced": 6,
    "best": 9,
}

# UI Constants
WINDOW_WIDTH: Final[int] = 1100
WINDOW_HEIGHT: Final[int] = 750
MIN_WINDOW_WIDTH: Final[int] = 900
MIN_WINDOW_HEIGHT: Final[int] = 600
PADDING_LARGE: Final[int] = 20
PADDING_MEDIUM: Final[int] = 10
PADDING_SMALL: Final[int] = 5

# File extensions for syntax highlighting (basic categorization)
CODE_EXTENSIONS: Final[set[str]] = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".r",
    ".m",
}

MARKUP_EXTENSIONS: Final[set[str]] = {
    ".html",
    ".xml",
    ".css",
    ".scss",
    ".sass",
    ".vue",
    ".jsx",
    ".tsx",
}

CONFIG_EXTENSIONS: Final[set[str]] = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
}

# Default exclusion patterns
DEFAULT_EXCLUDE_PATTERNS: Final[set[str]] = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "*.egg-info",
    ".idea",
    ".vscode",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "pip-log.txt",
    "pip-delete-this-directory.txt",
    ".coverage",
    "htmlcov",
    ".sass-cache",
    "*.log",
}

# Color schemes
DARK_THEME: Final[dict[str, str]] = {
    "bg": "#2b2b2b",
    "fg": "#ffffff",
    "select_bg": "#404040",
    "entry_bg": "#353535",
    "accent": "#0078d7",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
}

LIGHT_THEME: Final[dict[str, str]] = {
    "bg": "#f0f0f0",
    "fg": "#000000",
    "select_bg": "#0078d7",
    "entry_bg": "#ffffff",
    "accent": "#0078d7",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
}

# Set up professional logging
LOG_FILENAME: Final[str] = "folder_packer_pro.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
