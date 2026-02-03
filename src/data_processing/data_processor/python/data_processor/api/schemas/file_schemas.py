"""Pydantic schemas for file-related API endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """Information about a loaded file."""

    filename: str = Field(description="Name of the file")
    path: str = Field(description="Full path to the file")
    size_bytes: int = Field(description="File size in bytes", ge=0)
    row_count: int = Field(description="Number of rows in the file", ge=0)
    column_count: int = Field(description="Number of columns", ge=0)
    loaded_at: datetime = Field(description="When the file was loaded")


class FileUploadResponse(BaseModel):
    """Response after uploading/loading a file."""

    success: bool = Field(description="Whether the operation succeeded")
    file_id: str = Field(description="Unique identifier for the loaded file")
    file_info: FileInfo | None = Field(
        default=None, description="File information if successful"
    )
    error: str | None = Field(default=None, description="Error message if failed")


class FileListResponse(BaseModel):
    """Response containing list of loaded files."""

    files: list[FileInfo] = Field(default_factory=list, description="List of files")
    total_count: int = Field(description="Total number of files", ge=0)


class SignalInfo(BaseModel):
    """Information about a signal/column."""

    name: str = Field(description="Signal name")
    dtype: str = Field(description="Data type")
    is_numeric: bool = Field(description="Whether the signal is numeric")
    non_null_count: int = Field(description="Number of non-null values", ge=0)
    min_value: float | None = Field(default=None, description="Minimum value")
    max_value: float | None = Field(default=None, description="Maximum value")


class SignalListResponse(BaseModel):
    """Response containing list of signals from a file."""

    file_id: str = Field(description="File identifier")
    signals: list[SignalInfo] = Field(
        default_factory=list, description="List of signals"
    )
    numeric_count: int = Field(description="Number of numeric signals", ge=0)
    total_count: int = Field(description="Total number of signals", ge=0)
