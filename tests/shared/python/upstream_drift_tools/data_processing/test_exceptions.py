"""Tests for data_processing exceptions.

Covers the exception hierarchy, ColumnNotFoundError detail messages,
and exception inheritance.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.data_processing.exceptions import (
    ColumnNotFoundError,
    DataNotLoadedError,
    DataProcessingError,
    FileIOError,
    FilterError,
    FitError,
    TransformationError,
    UnsupportedOperationError,
)


class TestExceptionHierarchy:
    """Verify all exceptions inherit from DataProcessingError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            DataNotLoadedError,
            ColumnNotFoundError,
            FileIOError,
            TransformationError,
            FilterError,
            FitError,
            UnsupportedOperationError,
        ],
    )
    def test_subclass_of_base(self, exc_class: type) -> None:
        assert issubclass(exc_class, DataProcessingError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            DataNotLoadedError,
            ColumnNotFoundError,
            FileIOError,
            TransformationError,
            FilterError,
            FitError,
            UnsupportedOperationError,
        ],
    )
    def test_subclass_of_exception(self, exc_class: type) -> None:
        assert issubclass(exc_class, Exception)


class TestColumnNotFoundError:
    """Test the ColumnNotFoundError with detail messages."""

    def test_message_includes_column_name(self) -> None:
        err = ColumnNotFoundError("temperature")
        assert "temperature" in str(err)

    def test_message_without_available(self) -> None:
        err = ColumnNotFoundError("pressure")
        assert "Available" not in str(err)

    def test_message_with_available(self) -> None:
        err = ColumnNotFoundError("pressure", available=["temp", "flow", "level"])
        msg = str(err)
        assert "pressure" in msg
        assert "Available" in msg

    def test_column_attribute(self) -> None:
        err = ColumnNotFoundError("col_x")
        assert err.column == "col_x"

    def test_available_attribute_default_empty(self) -> None:
        err = ColumnNotFoundError("col_x")
        assert err.available == []

    def test_available_attribute_set(self) -> None:
        err = ColumnNotFoundError("col_x", available=["a", "b"])
        assert err.available == ["a", "b"]

    def test_catchable_as_base(self) -> None:
        with pytest.raises(DataProcessingError):
            raise ColumnNotFoundError("missing_col")


class TestSimpleExceptions:
    """Test that simple exceptions can be raised and caught."""

    def test_data_not_loaded(self) -> None:
        with pytest.raises(DataNotLoadedError, match="no data"):
            raise DataNotLoadedError("no data loaded")

    def test_file_io_error(self) -> None:
        with pytest.raises(FileIOError, match="read failed"):
            raise FileIOError("read failed")

    def test_transformation_error(self) -> None:
        with pytest.raises(TransformationError):
            raise TransformationError("bad transform")

    def test_filter_error(self) -> None:
        with pytest.raises(FilterError):
            raise FilterError("invalid query")

    def test_fit_error(self) -> None:
        with pytest.raises(FitError):
            raise FitError("curve fit failed")

    def test_unsupported_operation(self) -> None:
        with pytest.raises(UnsupportedOperationError):
            raise UnsupportedOperationError("unknown op")
