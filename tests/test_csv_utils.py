"""Tests for csv_utils - CSV file operation utilities.

These tests verify the CSV utility functions using
Design by Contract principles.
"""

import pandas as pd


class TestSafeReadCsvContract:
    """Design by Contract tests for safe_read_csv function."""

    def test_returns_dataframe(self, tmp_path):
        """Postcondition: Returns a DataFrame."""
        from utils.csv_utils import safe_read_csv

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3")

        result = safe_read_csv(csv_file)
        assert isinstance(result, pd.DataFrame)

    def test_returns_default_for_missing_file(self, tmp_path):
        """Postcondition: Returns default for missing file."""
        from utils.csv_utils import safe_read_csv

        result = safe_read_csv(tmp_path / "nonexistent.csv")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestSafeReadCsv:
    """Functional tests for safe_read_csv."""

    def test_reads_valid_csv(self, tmp_path):
        """Test reading valid CSV file."""
        from utils.csv_utils import safe_read_csv

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\nalpha,1\nbeta,2")

        df = safe_read_csv(csv_file)
        assert len(df) == 2
        assert list(df.columns) == ["name", "value"]

    def test_returns_custom_default(self, tmp_path):
        """Test returning custom default DataFrame."""
        from utils.csv_utils import safe_read_csv

        default_df = pd.DataFrame({"x": [0]})
        result = safe_read_csv(tmp_path / "missing.csv", default=default_df)
        assert list(result.columns) == ["x"]

    def test_passes_kwargs_to_pandas(self, tmp_path):
        """Test passing kwargs to pd.read_csv."""
        from utils.csv_utils import safe_read_csv

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a;b;c\n1;2;3")

        df = safe_read_csv(csv_file, sep=";")
        assert list(df.columns) == ["a", "b", "c"]

    def test_accepts_string_path(self, tmp_path):
        """Test accepting string path."""
        from utils.csv_utils import safe_read_csv

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1\nval1")

        df = safe_read_csv(str(csv_file))
        assert not df.empty


class TestSafeWriteCsvContract:
    """Design by Contract tests for safe_write_csv function."""

    def test_returns_bool(self, tmp_path):
        """Postcondition: Returns a boolean."""
        from utils.csv_utils import safe_write_csv

        df = pd.DataFrame({"a": [1]})
        result = safe_write_csv(df, tmp_path / "out.csv")
        assert isinstance(result, bool)


class TestSafeWriteCsv:
    """Functional tests for safe_write_csv."""

    def test_writes_csv_file(self, tmp_path):
        """Test writing CSV file."""
        from utils.csv_utils import safe_write_csv

        df = pd.DataFrame({"name": ["test"], "value": [42]})
        csv_file = tmp_path / "output.csv"

        result = safe_write_csv(df, csv_file, index=False)
        assert result is True
        assert csv_file.exists()

        content = csv_file.read_text()
        assert "name,value" in content
        assert "test,42" in content

    def test_creates_parent_directories(self, tmp_path):
        """Test creating parent directories."""
        from utils.csv_utils import safe_write_csv

        df = pd.DataFrame({"x": [1]})
        csv_file = tmp_path / "subdir" / "nested" / "data.csv"

        result = safe_write_csv(df, csv_file)
        assert result is True
        assert csv_file.exists()

    def test_respects_create_parents_false(self, tmp_path):
        """Test respecting create_parents=False."""
        from utils.csv_utils import safe_write_csv

        df = pd.DataFrame({"x": [1]})
        csv_file = tmp_path / "nonexistent_dir" / "data.csv"

        result = safe_write_csv(df, csv_file, create_parents=False)
        assert result is False

    def test_passes_kwargs_to_to_csv(self, tmp_path):
        """Test passing kwargs to df.to_csv."""
        from utils.csv_utils import safe_write_csv

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_file = tmp_path / "out.csv"

        safe_write_csv(df, csv_file, index=False, sep=";")
        content = csv_file.read_text()
        # With sep=";", columns should be separated by semicolon
        assert "a;b" in content


class TestReadCsvWithValidationContract:
    """Design by Contract tests for read_csv_with_validation function."""

    def test_returns_dataframe_or_none(self, tmp_path):
        """Postcondition: Returns DataFrame or None."""
        from utils.csv_utils import read_csv_with_validation

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2")

        result = read_csv_with_validation(csv_file)
        assert result is None or isinstance(result, pd.DataFrame)


class TestReadCsvWithValidation:
    """Functional tests for read_csv_with_validation."""

    def test_returns_df_when_valid(self, tmp_path):
        """Test returning DataFrame when valid."""
        from utils.csv_utils import read_csv_with_validation

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\ntest,1")

        result = read_csv_with_validation(csv_file, required_columns=["name", "value"])
        assert result is not None
        assert len(result) == 1

    def test_returns_none_for_missing_columns(self, tmp_path):
        """Test returning None for missing required columns."""
        from utils.csv_utils import read_csv_with_validation

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")

        result = read_csv_with_validation(csv_file, required_columns=["x", "y", "z"])
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Test returning None for empty file."""
        from utils.csv_utils import read_csv_with_validation

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        result = read_csv_with_validation(csv_file)
        assert result is None

    def test_no_required_columns(self, tmp_path):
        """Test with no required columns specified."""
        from utils.csv_utils import read_csv_with_validation

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("any,columns\n1,2")

        result = read_csv_with_validation(csv_file)
        assert result is not None


class TestMergeCsvFilesContract:
    """Design by Contract tests for merge_csv_files function."""

    def test_returns_bool(self, tmp_path):
        """Postcondition: Returns a boolean."""
        from utils.csv_utils import merge_csv_files

        result = merge_csv_files([], tmp_path / "merged.csv")
        assert isinstance(result, bool)


class TestMergeCsvFiles:
    """Functional tests for merge_csv_files."""

    def test_merges_multiple_files(self, tmp_path):
        """Test merging multiple CSV files."""
        from utils.csv_utils import merge_csv_files

        # Create source files
        file1 = tmp_path / "file1.csv"
        file1.write_text("name,value\na,1")
        file2 = tmp_path / "file2.csv"
        file2.write_text("name,value\nb,2")

        output = tmp_path / "merged.csv"
        result = merge_csv_files([file1, file2], output)

        assert result is True
        assert output.exists()

        df = pd.read_csv(output)
        assert len(df) == 2

    def test_returns_false_for_no_valid_files(self, tmp_path):
        """Test returning False when no valid files to merge."""
        from utils.csv_utils import merge_csv_files

        output = tmp_path / "merged.csv"
        result = merge_csv_files([tmp_path / "missing1.csv", tmp_path / "missing2.csv"], output)
        assert result is False

    def test_skips_empty_files(self, tmp_path):
        """Test skipping empty files during merge."""
        from utils.csv_utils import merge_csv_files

        file1 = tmp_path / "file1.csv"
        file1.write_text("col\nval")
        file2 = tmp_path / "empty.csv"
        file2.write_text("")

        output = tmp_path / "merged.csv"
        result = merge_csv_files([file1, file2], output)

        assert result is True
        df = pd.read_csv(output)
        assert len(df) == 1
