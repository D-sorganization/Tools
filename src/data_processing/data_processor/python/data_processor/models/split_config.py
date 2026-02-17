"""Configuration for data splitting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SplitMethod(Enum):
    """Methods for splitting large files."""

    ROWS = "rows"
    SIZE = "size"
    TIME = "time"
    CUSTOM = "custom"


@dataclass
class SplitConfig:
    """Configuration for file splitting."""

    enabled: bool = False
    method: SplitMethod = SplitMethod.ROWS
    rows_per_file: int = 100000
    max_file_size_mb: float = 100.0
    time_column: str = ""
    time_interval: str = "daily"  # daily, hourly, etc.
    custom_condition: str = ""
    output_directory: str = ""
    filename_pattern: str = "{base_name}_part_{part_number:04d}{extension}"
    compression: str = "snappy"
    include_header: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.rows_per_file <= 0:
            raise ValueError("rows_per_file must be positive")
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")

    def get_file_size_bytes(self) -> float:
        """Convert MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024
