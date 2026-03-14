# mypy: ignore-errors
"""Pipeline Recorder Module.

Records data processing operations into a reproducible pipeline.
"""

from __future__ import annotations

from typing import Any

from data_processor.core.script_generator_types import (
    OperationType,
    ProcessingPipeline,
)


class PipelineRecorder:
    """Records data processing operations into a pipeline."""

    def __init__(self, pipeline_name: str = "Untitled Pipeline") -> None:
        """Initialize the recorder."""
        assert pipeline_name is not None, "pipeline_name must be provided"
        self._pipeline = ProcessingPipeline(name=pipeline_name)
        self._recording = True

    @property
    def pipeline(self) -> ProcessingPipeline:
        """Get the current pipeline."""
        return self._pipeline

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._recording

    def start_recording(self) -> None:
        """Start recording operations."""
        self._recording = True

    def stop_recording(self) -> None:
        """Stop recording operations."""
        self._recording = False

    def clear(self) -> None:
        """Clear all recorded steps."""
        self._pipeline.steps.clear()

    def record_load(
        self,
        file_path: str,
        file_format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Record a file load operation."""
        assert file_path is not None, "file_path must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.LOAD,
            parameters={
                "file_path": file_path,
                "file_format": file_format,
                "options": options or {},
            },
            description=f"Load data from {file_path}",
        )

    def record_filter(
        self,
        filter_type: str,
        parameters: dict[str, Any],
        signals: list[str] | None = None,
    ) -> None:
        """Record a filter operation."""
        assert filter_type is not None, "filter_type must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.FILTER,
            parameters={
                "filter_type": filter_type,
                "filter_params": parameters,
                "signals": signals,
            },
            description=f"Apply {filter_type} filter",
        )

    def record_transform(
        self,
        transform_type: str,
        parameters: dict[str, Any],
    ) -> None:
        """Record a transformation operation."""
        assert transform_type is not None, "transform_type must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.TRANSFORM,
            parameters={
                "transform_type": transform_type,
                **parameters,
            },
            description=f"Apply {transform_type} transformation",
        )

    def record_calculate(
        self,
        column_name: str,
        formula: str,
    ) -> None:
        """Record a calculated column operation."""
        assert column_name is not None, "column_name must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.CALCULATE,
            parameters={
                "column_name": column_name,
                "formula": formula,
            },
            description=f"Calculate {column_name} = {formula}",
        )

    def record_resample(
        self,
        time_column: str,
        rule: str,
        method: str = "mean",
    ) -> None:
        """Record a resampling operation."""
        assert time_column is not None, "time_column must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.RESAMPLE,
            parameters={
                "time_column": time_column,
                "rule": rule,
                "method": method,
            },
            description=f"Resample to {rule} using {method}",
        )

    def record_integrate(
        self,
        time_column: str,
        signals: list[str],
        method: str = "trapezoidal",
    ) -> None:
        """Record an integration operation."""
        assert time_column is not None, "time_column must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.INTEGRATE,
            parameters={
                "time_column": time_column,
                "signals": signals,
                "method": method,
            },
            description=f"Integrate signals using {method}",
        )

    def record_differentiate(
        self,
        time_column: str,
        signals: list[str],
        method: str = "spline",
        orders: list[int] | None = None,
    ) -> None:
        """Record a differentiation operation."""
        assert time_column is not None, "time_column must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.DIFFERENTIATE,
            parameters={
                "time_column": time_column,
                "signals": signals,
                "method": method,
                "orders": orders or [1],
            },
            description=f"Differentiate signals using {method}",
        )

    def record_trim(
        self,
        time_column: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """Record a time range trim operation."""
        assert time_column is not None, "time_column must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.TRIM,
            parameters={
                "time_column": time_column,
                "start_time": start_time,
                "end_time": end_time,
            },
            description=(
                f"Trim time range: {start_time or 'start'} " f"to {end_time or 'end'}"
            ),
        )

    def record_select(
        self,
        columns: list[str],
    ) -> None:
        """Record a column selection operation."""
        assert columns is not None, "columns must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.SELECT,
            parameters={"columns": columns},
            description=f"Select {len(columns)} columns",
        )

    def record_export(
        self,
        file_path: str,
        file_format: str = "csv",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Record an export operation."""
        assert file_path is not None, "file_path must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.EXPORT,
            parameters={
                "file_path": file_path,
                "file_format": file_format,
                "options": options or {},
            },
            description=f"Export to {file_path}",
        )

    def record_custom(
        self,
        operation_name: str,
        parameters: dict[str, Any],
        description: str = "",
    ) -> None:
        """Record a custom operation."""
        assert operation_name is not None, "operation_name must be provided"
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.CUSTOM,
            parameters={
                "operation_name": operation_name,
                **parameters,
            },
            description=description or f"Custom operation: {operation_name}",
        )
