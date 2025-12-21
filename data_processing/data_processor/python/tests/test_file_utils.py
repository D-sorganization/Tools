import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add python/data_processor to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_processor"))

from file_utils import (  # noqa: PGH003
    OPENPYXL_AVAILABLE,
    PYARROW_AVAILABLE,
    DataReader,
    DataWriter,
    FileFormatDetector,
)


@pytest.fixture  # type: ignore[misc]
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4.1, 5.2, 6.3],  # Non-integer floats to preserve dtype in JSON/Excel
            "C": ["x", "y", "z"],
        }
    )


def test_detect_format() -> None:
    """Test format detection from file extension."""
    assert FileFormatDetector.detect_format("test.csv") == "csv"
    assert FileFormatDetector.detect_format("test.parquet") == "parquet"
    assert FileFormatDetector.detect_format("test.unknown") == "csv"  # Default


def test_csv_conversion(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test CSV read/write conversion."""
    filepath = tmp_path / "test.csv"
    DataWriter.write_file(sample_df, filepath, "csv")
    assert filepath.exists()

    loaded_df = DataReader.read_file(filepath, "csv")
    pd.testing.assert_frame_equal(sample_df, loaded_df)


def test_json_conversion(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test JSON read/write conversion."""
    filepath = tmp_path / "test.json"
    DataWriter.write_file(sample_df, filepath, "json")
    assert filepath.exists()

    loaded_df = DataReader.read_file(filepath, "json")
    pd.testing.assert_frame_equal(sample_df, loaded_df)


def test_pickle_conversion(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test Pickle read/write conversion."""
    filepath = tmp_path / "test.pkl"
    DataWriter.write_file(sample_df, filepath, "pickle")
    assert filepath.exists()

    loaded_df = DataReader.read_file(filepath, "pickle")
    pd.testing.assert_frame_equal(sample_df, loaded_df)


def test_numpy_conversion(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test Numpy read/write conversion."""
    filepath = tmp_path / "test.npy"
    # Numpy saves values only, loses column names and index
    # We filter to numeric only to avoid pickle requirement
    numeric_df = sample_df.select_dtypes(include=[np.number])
    DataWriter.write_file(numeric_df, filepath, "numpy")
    assert filepath.exists()

    loaded_df = DataReader.read_file(filepath, "numpy")
    # Check values match
    numeric_df = sample_df.select_dtypes(include=[np.number])
    np.testing.assert_array_equal(numeric_df.values, loaded_df.values)


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")  # type: ignore[misc]
def test_parquet_conversion(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test Parquet read/write conversion."""
    filepath = tmp_path / "test.parquet"
    DataWriter.write_file(sample_df, filepath, "parquet")
    assert filepath.exists()

    loaded_df = DataReader.read_file(filepath, "parquet")
    pd.testing.assert_frame_equal(sample_df, loaded_df)


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")  # type: ignore[misc]
def test_feather_conversion(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test Feather read/write conversion."""
    filepath = tmp_path / "test.feather"
    DataWriter.write_file(sample_df, filepath, "feather")
    assert filepath.exists()

    loaded_df = DataReader.read_file(filepath, "feather")
    pd.testing.assert_frame_equal(sample_df, loaded_df)


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")  # type: ignore[misc]
def test_excel_conversion(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test Excel read/write conversion."""
    filepath = tmp_path / "test.xlsx"
    DataWriter.write_file(sample_df, filepath, "excel")
    assert filepath.exists()

    loaded_df = DataReader.read_file(filepath, "excel")
    pd.testing.assert_frame_equal(sample_df, loaded_df)
