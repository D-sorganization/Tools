"""Utility functions for PDF processing and filename handling."""

import hashlib
import re
import unicodedata
from pathlib import Path

MAX_TITLE_LENGTH = 200
MAX_FILENAME_LENGTH = 200  # Conservative limit for cross-platform compatibility

# Windows reserved filenames
WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

# Minor words for title case
MINOR_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "from", "with", "as"
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Calculates the SHA256 hash of a file.

    Args:
        path: Path to file
        chunk_size: Size of chunks to read (default 1MB)

    Returns:
        Hexadecimal SHA256 hash string
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean_title(s: str) -> str:
    """
    Cleans up a title string by removing extra whitespace and special characters.

    Args:
        s: Input title string

    Returns:
        Cleaned title string
    """
    if not s:
        return ""
    # Normalize unicode (NFD -> NFC)
    s = unicodedata.normalize("NFC", s)
    # Replace multiple whitespace with single space
    s = re.sub(r"\s+", " ", s).strip()
    # Remove leading/trailing non-word characters
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    return s[:MAX_TITLE_LENGTH]


def looks_like_title(s: str) -> bool:
    """
    Heuristic check to see if a string looks like a valid title.

    Args:
        s: String to check

    Returns:
        True if string looks like a title, False otherwise
    """
    if not s or len(s) < 6:
        return False

    # Avoid common non-title strings
    bad = ["arxiv", "doi:", "http", "www.", "copyright", "all rights reserved",
           "page ", "draft", "confidential"]
    if any(b in s.lower() for b in bad):
        return False

    # Avoid section headers
    if s.strip().lower() in {"abstract", "introduction", "references", "appendix"}:
        return False

    # Check if it looks like a page number
    if re.match(r"^\d+$", s.strip()):
        return False

    return True


def sanitize_filename(s: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    """
    Removes characters invalid in filenames and handles edge cases.

    Args:
        s: Input filename string
        max_length: Maximum length for filename

    Returns:
        Sanitized filename string
    """
    if not s:
        return ""

    # Normalize unicode
    s = unicodedata.normalize("NFC", s)

    # Remove invalid filename characters
    s = re.sub(r'[\\/*?:"<>|]', "", s)

    # Remove control characters
    s = "".join(char for char in s if unicodedata.category(char)[0] != "C")

    # Strip leading/trailing whitespace and periods
    s = s.strip().strip(".")

    # Handle Windows reserved names
    name_upper = s.upper().split(".")[0]  # Get name without extension
    if name_upper in WINDOWS_RESERVED:
        s = f"_{s}"  # Prefix with underscore to make it safe

    # Truncate to max length
    if len(s) > max_length:
        s = s[:max_length].strip()

    return s if s else "untitled"


def to_snake_case(s: str) -> str:
    """
    Converts string to snake_case.

    Args:
        s: Input string

    Returns:
        snake_case string
    """
    s = sanitize_filename(s).lower()
    # Replace spaces and hyphens with underscores
    s = re.sub(r"[\s\-]+", "_", s)
    # Remove any remaining non-alphanumeric chars (except underscores)
    s = re.sub(r"[^a-z0-9_]", "", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "untitled"


def to_kebab_case(s: str) -> str:
    """
    Converts string to kebab-case.

    Args:
        s: Input string

    Returns:
        kebab-case string
    """
    s = sanitize_filename(s).lower()
    # Replace spaces and underscores with hyphens
    s = re.sub(r"[\s_]+", "-", s)
    # Remove any remaining non-alphanumeric chars (except hyphens)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    # Collapse multiple hyphens
    s = re.sub(r"\-+", "-", s).strip("-")
    return s if s else "untitled"


def to_title_case(s: str) -> str:
    """
    Converts string to proper Title Case, ignoring minor words.

    Args:
        s: Input string

    Returns:
        Title Case string
    """
    if not s:
        return ""

    words = s.split()
    if not words:
        return ""

    cased_words = []
    for i, word in enumerate(words):
        lower_word = word.lower()
        # Capitalize if it's the first word, last word, or not a minor word
        if i == 0 or i == len(words) - 1 or lower_word not in MINOR_WORDS:
            cased_words.append(word.capitalize())
        else:
            cased_words.append(lower_word)

    return " ".join(cased_words)


def get_last_name(author: str) -> str:
    """
    Extracts the last name from an author string.

    Args:
        author: Full author name

    Returns:
        Last name or empty string
    """
    if not author:
        return ""
    # Simple heuristic: split by space and take the last part
    parts = author.strip().split()
    if not parts:
        return ""
    return parts[-1]
