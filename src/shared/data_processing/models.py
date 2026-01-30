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
    time_interval_hours: float = 24.0
    custom_condition: str = ""
    output_directory: str = ""
    filename_pattern: str = "{base_name}_part_{part_number:04d}{extension}"
    compression: str = "snappy"  # For parquet files
    include_header: bool = True  # For CSV files

    def get_file_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return int(self.max_file_size_mb * 1024 * 1024)
