from typing import Any

"""Tests for SplitConfig model."""

from typing import Any

import pytest
from data_processor.models.split_config import SplitConfig, SplitMethod


def test_split_config_defaults() -> Any:
    """Test default values of SplitConfig."""
    config = SplitConfig()
    assert config.enabled is False
    assert config.method == SplitMethod.ROWS
    assert config.rows_per_file == 100000
    assert config.max_file_size_mb == 100.0


def test_split_config_validation() -> Any:
    """Test validation (DbC) in SplitConfig."""
    # Valid config
    config = SplitConfig(rows_per_file=10, max_file_size_mb=1.0)
    assert config.rows_per_file == 10

    # Invalid rows_per_file
    with pytest.raises(ValueError, match="rows_per_file must be positive"):
        SplitConfig(rows_per_file=0)

    with pytest.raises(ValueError, match="rows_per_file must be positive"):
        SplitConfig(rows_per_file=-100)

    # Invalid max_file_size_mb
    with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
        SplitConfig(max_file_size_mb=0)

    with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
        SplitConfig(max_file_size_mb=-1.0)


def test_get_file_size_bytes() -> Any:
    """Test file size conversion."""
    config = SplitConfig(max_file_size_mb=1.0)
    assert config.get_file_size_bytes() == 1024 * 1024

    config = SplitConfig(max_file_size_mb=0.5)
    assert config.get_file_size_bytes() == 512 * 1024
