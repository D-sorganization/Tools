"""Dataset naming utilities for file output.

Provides functions for:
- Auto-generating dataset names with timestamps
- Validating custom names
- Generating unique filenames to avoid overwrites
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def generate_dataset_name(
    base_name: str = "data",
    include_timestamp: bool = True,
    include_filter: bool = False,
    filter_type: str | None = None,
    include_date: bool = True,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> str:
    """Generate a dataset name with optional components.

    Args:
        base_name: Base name for the dataset
        include_timestamp: Include timestamp in name
        include_filter: Include filter type in name
        filter_type: Filter type string
        include_date: Include date in name
        timestamp_format: Format string for timestamp

    Returns:
        Generated dataset name (without extension)
    """
    if not (base_name is not None):
        raise ValueError("base_name must be provided")
    parts = [base_name]

    if include_filter and filter_type:
        # Clean the filter type for filename
        clean_filter = re.sub(r"[^\w\-]", "_", filter_type.lower())
        parts.append(clean_filter)

    if include_date:
        parts.append(datetime.now().strftime("%Y-%m-%d"))

    if include_timestamp:
        parts.append(datetime.now().strftime(timestamp_format))

    return "_".join(parts)


def validate_dataset_name(name: str) -> bool:
    """Validate that a dataset name is valid for use as a filename.

    Args:
        name: Proposed dataset name

    Returns:
        True if valid, False otherwise

    Valid names:
    - Non-empty
    - Only alphanumeric, underscore, hyphen, and dot
    - No path separators
    - No leading/trailing whitespace
    """
    if not name or not name.strip():
        return False

    # Check for invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    if re.search(invalid_chars, name):
        return False

    # Check for path separators
    if "/" in name or "\\" in name:
        return False

    # Check for leading/trailing whitespace
    if name != name.strip():
        return False

    # Check for valid characters only
    valid_pattern = r"^[\w\-. ]+$"
    if not re.match(valid_pattern, name):
        return False

    return True


def sanitize_dataset_name(name: str) -> str:
    """Sanitize a dataset name to make it valid.

    Args:
        name: Input name (potentially invalid)

    Returns:
        Sanitized name safe for use as a filename
    """
    # Remove leading/trailing whitespace
    name = name.strip()

    # Replace invalid characters with underscore
    name = re.sub(r'[<>:"/\\|?*]', "_", name)

    # Replace multiple underscores with single
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscores
    name = name.strip("_")

    # If empty after sanitization, use default
    if not name:
        name = "data"

    return name


def generate_unique_name(
    directory: Path | str,
    base_name: str,
    extension: str,
    max_attempts: int = 1000,
) -> str:
    """Generate a unique filename that doesn't exist in the directory.

    Args:
        directory: Directory to check for existing files
        base_name: Base name for the file (without extension)
        extension: File extension (with or without leading dot)
        max_attempts: Maximum number of suffix attempts

    Returns:
        Unique filename (not full path, just the filename)

    Raises:
        RuntimeError: If unable to find unique name after max_attempts
    """
    directory = Path(directory)

    # Ensure extension has leading dot
    if not extension.startswith("."):
        extension = f".{extension}"

    # Try original name first
    original = f"{base_name}{extension}"
    if not (directory / original).exists():
        return original

    # Try with numeric suffix
    for i in range(1, max_attempts + 1):
        candidate = f"{base_name}_{i}{extension}"
        if not (directory / candidate).exists():
            return candidate

    raise RuntimeError(f"Unable to generate unique name after {max_attempts} attempts")


def generate_timestamped_name(
    base_name: str,
    extension: str = "",
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> str:
    """Generate a filename with embedded timestamp.

    Args:
        base_name: Base name for the file
        extension: Optional file extension
        timestamp_format: Format for the timestamp

    Returns:
        Filename with timestamp
    """
    if not (base_name is not None):
        raise ValueError("base_name must be provided")
    timestamp = datetime.now().strftime(timestamp_format)
    name = f"{base_name}_{timestamp}"

    if extension:
        if not extension.startswith("."):
            extension = f".{extension}"
        name += extension

    return name


def parse_dataset_name(filename: str) -> dict[str, str | None]:
    """Parse a dataset filename to extract components.

    Args:
        filename: Filename to parse

    Returns:
        Dictionary with parsed components:
        - base_name: Original base name
        - timestamp: Extracted timestamp (if present)
        - filter_type: Extracted filter type (if present)
        - extension: File extension
    """
    path = Path(filename)
    name = path.stem
    extension = path.suffix

    result: dict[str, str | None] = {
        "base_name": name,
        "timestamp": None,
        "filter_type": None,
        "extension": extension,
    }

    # Try to extract timestamp (YYYYMMDD_HHMMSS pattern)
    timestamp_pattern = r"(\d{8}_\d{6})"
    timestamp_match = re.search(timestamp_pattern, name)
    if timestamp_match:
        result["timestamp"] = timestamp_match.group(1)
        name = name.replace(timestamp_match.group(1), "").strip("_")

    # Try to extract date (YYYY-MM-DD pattern)
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    date_match = re.search(date_pattern, name)
    if date_match:
        if result["timestamp"]:
            result["timestamp"] = date_match.group(1) + "_" + result["timestamp"]
        else:
            result["timestamp"] = date_match.group(1)
        name = name.replace(date_match.group(1), "").strip("_")

    # Known filter types
    filter_types = [
        "moving_average",
        "butterworth",
        "median",
        "savgol",
        "hampel",
        "zscore",
        "gaussian",
        "fft",
    ]

    for ft in filter_types:
        if ft in name.lower():
            result["filter_type"] = ft
            name = re.sub(ft, "", name, flags=re.IGNORECASE).strip("_")
            break

    result["base_name"] = name.strip("_") or "data"

    return result


__all__ = [
    "generate_dataset_name",
    "validate_dataset_name",
    "sanitize_dataset_name",
    "generate_unique_name",
    "generate_timestamped_name",
    "parse_dataset_name",
]
