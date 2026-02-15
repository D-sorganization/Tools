"""Constants and validation for the Folder Tool application.

This module centralizes all configuration constants, their validation,
documentation metadata, and export functionality. Constants include
file size limits, UI dimensions, progress tracking values, and
dialog layout parameters.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File operation constants
# ---------------------------------------------------------------------------
MAX_LOG_ENTRIES: Final[int] = (
    20  # Maximum number of log entries to display per operation
    # - UI performance limit
)

PROGRESS_INCREMENT: Final[int] = (
    10  # Progress bar increment percentage [%]
    # - standard UI update frequency
)
MAX_FILE_SIZE_MB: Final[int] = (
    1024  # Maximum file size limit [MB]
    # - Windows FAT32 limit per Microsoft docs
)
MIN_FILE_SIZE_BYTES: Final[int] = (
    1  # Minimum file size [bytes]
    # - 1 byte minimum per filesystem standards
)
DEFAULT_CHUNK_SIZE: Final[int] = (
    8192  # File copy chunk size [bytes]
    # - optimal for most systems per Python shutil docs
)
MAX_RETRY_ATTEMPTS: Final[int] = (
    3  # Maximum retry attempts for file operations
    # - industry standard retry limit
)

# ---------------------------------------------------------------------------
# UI / dialog constants
# ---------------------------------------------------------------------------
ICON_SIZES: Final[tuple[int, ...]] = (
    16,
    32,
    48,
    64,
)  # Standard icon sizes [pixels] per Windows Shell API guidelines
MAX_STATUS_LENGTH: Final[int] = (
    200  # Maximum status message length [characters]
    # - prevents UI overflow
)
MAX_UI_UPDATE_FREQUENCY: Final[int] = (
    10  # Update progress every N files
    # - balances responsiveness with performance
)
MAX_ARCHIVE_SIZE_RATIO: Final[float] = (
    0.1  # Minimum extracted size ratio [ratio]
    # - archive size * 0.1 for validation
)
MAX_DIALOG_WIDTH: Final[int] = (
    800  # Maximum dialog width [pixels] - prevents dialog overflow
)
MAX_DIALOG_HEIGHT: Final[int] = (
    600  # Maximum dialog height [pixels] - prevents dialog overflow
)
MIN_DIALOG_WIDTH: Final[int] = 400  # Minimum dialog width [pixels] - ensures usability
MIN_DIALOG_HEIGHT: Final[int] = (
    300  # Minimum dialog height [pixels] - ensures usability
)

# ---------------------------------------------------------------------------
# Content / text constants
# ---------------------------------------------------------------------------
MAX_TEXT_CONTENT_SIZE: Final[int] = (
    1000000  # Maximum text content size [characters]
    # - prevents performance issues
)
MAX_TITLE_LENGTH: Final[int] = (
    100  # Maximum title length [characters]
    # - prevents window title truncation
)
MAX_COUNTER_ATTEMPTS: Final[int] = (
    1000  # Maximum attempts to generate unique filename [attempts]
    # - prevents infinite loops
)
MAX_FALLBACK_CONTENT_SIZE: Final[int] = (
    500  # Maximum content size for fallback display [characters]
    # - prevents UI overflow
)

# ---------------------------------------------------------------------------
# Progress tracking constants
# ---------------------------------------------------------------------------
PROGRESS_BACKUP_PERCENT: Final[int] = (
    20  # Progress percentage allocated to backup operations [%]
)
PROGRESS_MAIN_OP_PERCENT: Final[int] = (
    40  # Progress percentage allocated to main operations [%]
)
PROGRESS_ZIP_PERCENT: Final[int] = (
    10  # Progress percentage allocated to ZIP creation [%]
)
PROGRESS_START_MAIN: Final[int] = (
    30  # Starting progress percentage for main operations [%]
)
PROGRESS_START_ZIP: Final[int] = 85  # Starting progress percentage for ZIP creation [%]

# ---------------------------------------------------------------------------
# Dialog layout constants
# ---------------------------------------------------------------------------
CHARS_PER_DIALOG_LINE: Final[int] = (
    80  # Characters per line for dialog width calculation [characters]
    # - standard text width
)
DIALOG_WIDTH_OFFSET: Final[int] = (
    100  # Additional width offset for dialog borders [pixels]
    # - accounts for scrollbars and margins
)
DIALOG_HEIGHT_OFFSET: Final[int] = (
    100  # Additional height offset for dialog borders [pixels]
    # - accounts for title bar and margins
)
LINE_HEIGHT_PIXELS: Final[int] = (
    20  # Height per line for dialog height calculation [pixels]
)
MAX_TITLE_PREVIEW_LENGTH: Final[int] = (
    50  # Maximum title length for preview in logs [characters]
    # - prevents log overflow
)


def validate_constants() -> None:
    """Validate that all constants meet the required constraints.

    This function ensures all constants are within valid ranges and follow
    logical relationships. It validates file sizes, UI dimensions, progress
    percentages, and other configuration values.

    Raises:
        ValueError: If any constant violates its constraints
    """
    # Validate file size constants
    if MAX_FILE_SIZE_MB <= 0:
        raise ValueError(
            f"MAX_FILE_SIZE_MB must be positive, got {MAX_FILE_SIZE_MB}",
        )
    if MIN_FILE_SIZE_BYTES < 0:
        raise ValueError(
            f"MIN_FILE_SIZE_BYTES must be non-negative, got {MIN_FILE_SIZE_BYTES}",
        )
    if MIN_FILE_SIZE_BYTES >= MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"MIN_FILE_SIZE_BYTES must be less than MAX_FILE_SIZE_MB, "
            f"got {MIN_FILE_SIZE_BYTES}",
        )

    # Validate UI constants
    if MAX_STATUS_LENGTH <= 0:
        raise ValueError(
            f"MAX_STATUS_LENGTH must be positive, got {MAX_STATUS_LENGTH}",
        )
    if MAX_UI_UPDATE_FREQUENCY <= 0:
        raise ValueError(
            f"MAX_UI_UPDATE_FREQUENCY must be positive, got {MAX_UI_UPDATE_FREQUENCY}",
        )
    if MAX_DIALOG_WIDTH <= MIN_DIALOG_WIDTH:
        raise ValueError(
            f"MAX_DIALOG_WIDTH must be greater than MIN_DIALOG_WIDTH, "
            f"got {MAX_DIALOG_WIDTH} <= {MIN_DIALOG_WIDTH}",
        )
    if MAX_DIALOG_HEIGHT <= MIN_DIALOG_HEIGHT:
        raise ValueError(
            "MAX_DIALOG_HEIGHT must be greater than MIN_DIALOG_HEIGHT, "
            f"got {MAX_DIALOG_HEIGHT} <= {MIN_DIALOG_HEIGHT}",
        )

    # Validate archive constants
    if not 0 < MAX_ARCHIVE_SIZE_RATIO < 1:
        raise ValueError(
            f"MAX_ARCHIVE_SIZE_RATIO must be between 0 and 1, "
            f"got {MAX_ARCHIVE_SIZE_RATIO}",
        )

    # Validate retry constants
    if MAX_RETRY_ATTEMPTS <= 0:
        raise ValueError(
            f"MAX_RETRY_ATTEMPTS must be positive, got {MAX_RETRY_ATTEMPTS}",
        )

    # Validate new constants
    if MAX_TEXT_CONTENT_SIZE <= 0:
        raise ValueError(
            f"MAX_TEXT_CONTENT_SIZE must be positive, got {MAX_TEXT_CONTENT_SIZE}",
        )
    if MAX_TITLE_LENGTH <= 0:
        raise ValueError(
            f"MAX_TITLE_LENGTH must be positive, got {MAX_TITLE_LENGTH}",
        )
    if MAX_COUNTER_ATTEMPTS <= 0:
        raise ValueError(
            f"MAX_COUNTER_ATTEMPTS must be positive, got {MAX_COUNTER_ATTEMPTS}",
        )

    # Validate progress constants
    for name, value in [
        ("PROGRESS_BACKUP_PERCENT", PROGRESS_BACKUP_PERCENT),
        ("PROGRESS_MAIN_OP_PERCENT", PROGRESS_MAIN_OP_PERCENT),
        ("PROGRESS_ZIP_PERCENT", PROGRESS_ZIP_PERCENT),
        ("PROGRESS_START_MAIN", PROGRESS_START_MAIN),
        ("PROGRESS_START_ZIP", PROGRESS_START_ZIP),
    ]:
        if value < 0 or value > 100:
            raise ValueError(
                f"{name} must be between 0 and 100, got {value}",
            )

    # Validate progress flow consistency
    total_progress = (
        PROGRESS_BACKUP_PERCENT + PROGRESS_MAIN_OP_PERCENT + PROGRESS_ZIP_PERCENT
    )
    if total_progress > 100:
        raise ValueError(
            f"Total progress allocation exceeds 100%: {total_progress}",
        )

    logger.info("All constants validated successfully")


def get_constants_info() -> dict[str, dict[str, str]]:
    """Return information about all constants for debugging and documentation.

    Returns:
        Dictionary mapping constant names to their metadata (value, units, source).
    """
    return {
        "MAX_LOG_ENTRIES": {
            "value": str(MAX_LOG_ENTRIES),
            "units": "entries",
            "source": "UI performance limit",
        },
        "PROGRESS_INCREMENT": {
            "value": str(PROGRESS_INCREMENT),
            "units": "%",
            "source": "Standard UI update frequency",
        },
        "MAX_FILE_SIZE_MB": {
            "value": str(MAX_FILE_SIZE_MB),
            "units": "MB",
            "source": "Windows FAT32 limit per Microsoft docs",
        },
        "MIN_FILE_SIZE_BYTES": {
            "value": str(MIN_FILE_SIZE_BYTES),
            "units": "bytes",
            "source": "1 byte minimum per filesystem standards",
        },
        "DEFAULT_CHUNK_SIZE": {
            "value": str(DEFAULT_CHUNK_SIZE),
            "units": "bytes",
            "source": "Optimal for most systems per Python shutil docs",
        },
        "MAX_RETRY_ATTEMPTS": {
            "value": str(MAX_RETRY_ATTEMPTS),
            "units": "attempts",
            "source": "Industry standard retry limit",
        },
        "ICON_SIZES": {
            "value": str(ICON_SIZES),
            "units": "pixels",
            "source": "Windows Shell API guidelines",
        },
        "MAX_STATUS_LENGTH": {
            "value": str(MAX_STATUS_LENGTH),
            "units": "characters",
            "source": "Prevents UI overflow",
        },
        "MAX_UI_UPDATE_FREQUENCY": {
            "value": str(MAX_UI_UPDATE_FREQUENCY),
            "units": "files",
            "source": "Balances responsiveness with performance",
        },
        "MAX_ARCHIVE_SIZE_RATIO": {
            "value": str(MAX_ARCHIVE_SIZE_RATIO),
            "units": "ratio",
            "source": "Archive size * 0.1 for validation",
        },
        "MAX_DIALOG_WIDTH": {
            "value": str(MAX_DIALOG_WIDTH),
            "units": "pixels",
            "source": "Prevents dialog overflow",
        },
        "MAX_DIALOG_HEIGHT": {
            "value": str(MAX_DIALOG_HEIGHT),
            "units": "pixels",
            "source": "Prevents dialog overflow",
        },
        "MIN_DIALOG_WIDTH": {
            "value": str(MIN_DIALOG_WIDTH),
            "units": "pixels",
            "source": "Ensures usability",
        },
        "MIN_DIALOG_HEIGHT": {
            "value": str(MIN_DIALOG_HEIGHT),
            "units": "pixels",
            "source": "Ensures usability",
        },
        "MAX_TEXT_CONTENT_SIZE": {
            "value": str(MAX_TEXT_CONTENT_SIZE),
            "units": "characters",
            "source": "Prevents performance issues in text dialogs",
        },
        "MAX_TITLE_LENGTH": {
            "value": str(MAX_TITLE_LENGTH),
            "units": "characters",
            "source": "Prevents window title truncation",
        },
        "MAX_COUNTER_ATTEMPTS": {
            "value": str(MAX_COUNTER_ATTEMPTS),
            "units": "attempts",
            "source": "Prevents infinite loops in filename generation",
        },
        "MAX_FALLBACK_CONTENT_SIZE": {
            "value": str(MAX_FALLBACK_CONTENT_SIZE),
            "units": "characters",
            "source": "Prevents UI overflow in fallback dialogs",
        },
        "PROGRESS_BACKUP_PERCENT": {
            "value": str(PROGRESS_BACKUP_PERCENT),
            "units": "%",
            "source": "UI progress tracking for backup operations",
        },
        "PROGRESS_MAIN_OP_PERCENT": {
            "value": str(PROGRESS_MAIN_OP_PERCENT),
            "units": "%",
            "source": "UI progress tracking for main operations",
        },
        "PROGRESS_ZIP_PERCENT": {
            "value": str(PROGRESS_ZIP_PERCENT),
            "units": "%",
            "source": "UI progress tracking for ZIP creation",
        },
        "PROGRESS_START_MAIN": {
            "value": str(PROGRESS_START_MAIN),
            "units": "%",
            "source": "Starting progress for main operations",
        },
        "PROGRESS_START_ZIP": {
            "value": str(PROGRESS_START_ZIP),
            "units": "%",
            "source": "Starting progress for ZIP creation",
        },
    }


def export_constants_documentation(output_path: str) -> bool:
    """Export constants documentation to a file for reference.

    Args:
        output_path: Path to the output file.

    Returns:
        True if export successful, False otherwise.

    Raises:
        OSError: If file operations fail.
    """
    try:
        constants_info = get_constants_info()

        content = [
            "# Folder Tool Constants Documentation",
            f"Generated: {datetime.now()}",
            "",
            "## Constants Overview",
            "",
        ]

        for const_name, info in constants_info.items():
            content.append(f"### {const_name}")
            content.append(f"- **Value**: {info['value']}")
            content.append(f"- **Units**: {info['units']}")
            content.append(f"- **Source**: {info['source']}")
            content.append("")

        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(content), encoding="utf-8")

        logger.info("Constants documentation exported to: %s", output_path)
        return True

    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        logger.error("Failed to export constants documentation: %s", e)
        return False
