"""Pydantic schemas for processing-related API endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FilterType(str, Enum):
    """Supported filter types."""

    MOVING_AVERAGE = "Moving Average"
    BUTTERWORTH_LOWPASS = "Butterworth Low-pass"
    BUTTERWORTH_HIGHPASS = "Butterworth High-pass"
    MEDIAN = "Median Filter"
    HAMPEL = "Hampel Filter"
    ZSCORE = "Z-Score Filter"
    SAVITZKY_GOLAY = "Savitzky-Golay"
    GAUSSIAN = "Gaussian Filter"
    FFT_LOWPASS = "FFT Low-pass"
    FFT_HIGHPASS = "FFT High-pass"
    FFT_BANDPASS = "FFT Band-pass"
    FFT_BANDSTOP = "FFT Band-stop"


class FilterParameters(BaseModel):
    """Filter parameters model."""

    ma_window: int | None = Field(default=None, ge=3, le=10000)
    bw_order: int | None = Field(default=None, ge=1, le=10)
    bw_cutoff: float | None = Field(default=None, ge=0.0001, le=0.9999)
    median_kernel: int | None = Field(default=None, ge=3, le=10001)
    hampel_window: int | None = Field(default=None, ge=3, le=10001)
    hampel_threshold: float | None = Field(default=None, ge=0.0, le=1000.0)
    zscore_threshold: float | None = Field(default=None, ge=0.0, le=1000.0)
    zscore_method: str | None = Field(default=None)
    savgol_window: int | None = Field(default=None, ge=3, le=10001)
    savgol_polyorder: int | None = Field(default=None, ge=1, le=9)
    gaussian_sigma: float | None = Field(default=None, ge=0.0, le=10000.0)
    fft_freq_low: float | None = Field(default=None, ge=0.0)
    fft_freq_high: float | None = Field(default=None, ge=0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class FilterRequest(BaseModel):
    """Request to apply a filter to signals."""

    file_id: str = Field(description="File identifier")
    filter_type: FilterType = Field(description="Type of filter to apply")
    signals: list[str] = Field(
        default_factory=list, description="Signals to filter (empty = all numeric)"
    )
    parameters: FilterParameters = Field(
        default_factory=FilterParameters, description="Filter parameters"
    )

    @field_validator("signals")
    @classmethod
    def validate_signals(cls, v: list[str]) -> list[str]:
        """Validate signal names are non-empty strings."""
        return [s.strip() for s in v if s and s.strip()]


class ProcessingStatus(str, Enum):
    """Status of a processing operation."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FilterResponse(BaseModel):
    """Response after applying a filter."""

    success: bool = Field(description="Whether the operation succeeded")
    status: ProcessingStatus = Field(description="Processing status")
    file_id: str = Field(description="File identifier")
    filter_type: str = Field(description="Filter type applied")
    signals_processed: list[str] = Field(
        default_factory=list, description="Signals that were processed"
    )
    row_count: int = Field(default=0, description="Number of rows processed", ge=0)
    error: str | None = Field(default=None, description="Error message if failed")


class StatisticsRequest(BaseModel):
    """Request to calculate statistics for signals."""

    file_id: str = Field(description="File identifier")
    signals: list[str] = Field(
        default_factory=list, description="Signals to analyze (empty = all numeric)"
    )


class SignalStatistics(BaseModel):
    """Statistics for a single signal."""

    name: str = Field(description="Signal name")
    count: int = Field(description="Number of values", ge=0)
    mean: float | None = Field(default=None, description="Mean value")
    std: float | None = Field(default=None, description="Standard deviation")
    min: float | None = Field(default=None, description="Minimum value")
    max: float | None = Field(default=None, description="Maximum value")
    median: float | None = Field(default=None, description="Median value")
    q25: float | None = Field(default=None, description="25th percentile")
    q75: float | None = Field(default=None, description="75th percentile")


class StatisticsResponse(BaseModel):
    """Response containing statistics for signals."""

    file_id: str = Field(description="File identifier")
    statistics: list[SignalStatistics] = Field(
        default_factory=list, description="Statistics for each signal"
    )
    error: str | None = Field(default=None, description="Error message if failed")


class ExportFormat(str, Enum):
    """Supported export formats."""

    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    FEATHER = "feather"


class ExportRequest(BaseModel):
    """Request to export processed data."""

    file_id: str = Field(description="File identifier")
    format: ExportFormat = Field(default=ExportFormat.CSV, description="Export format")
    signals: list[str] = Field(
        default_factory=list, description="Signals to export (empty = all)"
    )
    filename: str | None = Field(default=None, description="Output filename")


class ExportResponse(BaseModel):
    """Response after exporting data."""

    success: bool = Field(description="Whether the export succeeded")
    filename: str = Field(description="Output filename")
    format: str = Field(description="Export format")
    size_bytes: int = Field(default=0, description="File size in bytes", ge=0)
    error: str | None = Field(default=None, description="Error message if failed")


class DataPreviewRequest(BaseModel):
    """Request to preview data."""

    file_id: str = Field(description="File identifier")
    signals: list[str] = Field(
        default_factory=list, description="Signals to include (empty = all)"
    )
    offset: int = Field(default=0, description="Row offset", ge=0)
    limit: int = Field(default=100, description="Maximum rows to return", ge=1, le=1000)


class DataPreviewResponse(BaseModel):
    """Response containing data preview."""

    file_id: str = Field(description="File identifier")
    columns: list[str] = Field(description="Column names")
    data: list[list[Any]] = Field(description="Data rows")
    total_rows: int = Field(description="Total rows in dataset", ge=0)
    offset: int = Field(description="Current offset", ge=0)
    limit: int = Field(description="Rows returned", ge=0)
