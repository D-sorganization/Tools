# mypy: ignore-errors
"""Pipeline Executor Module.

Executes processing pipelines on data, dispatching each step
to the appropriate processing function.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from data_processor.core.script_generator_types import (
    OperationType,
    ProcessingPipeline,
    ProcessingStep,
)

if TYPE_CHECKING:
    import pandas as pd

    from data_processor.vectorized_filter_engine import VectorizedFilterEngine

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes processing pipelines on data."""

    def __init__(self) -> None:
        """Initialize the executor."""
        self._filter_engine: VectorizedFilterEngine | None = None

    def execute(
        self,
        pipeline: ProcessingPipeline,
        input_data: str | Path | pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Execute a pipeline on input data.

        Args:
            pipeline: Processing pipeline to execute
            input_data: Input file path or DataFrame
            output_path: Optional output file path

        Returns:
            Processed DataFrame
        """
        assert pipeline is not None, "pipeline must be provided"
        import pandas as pd

        # Load data if path provided
        if isinstance(input_data, (str, Path)):
            df = pd.read_csv(input_data)
        else:
            df = input_data.copy()

        # Execute each step
        for i, step in enumerate(pipeline.steps):
            if not step.enabled:
                logger.debug(f"Skipping disabled step {i+1}: {step.description}")
                continue

            logger.info(f"Executing step {i+1}: {step.description}")
            df = self._execute_step(df, step)

        # Export if output path provided
        if output_path:
            output_path = Path(output_path)
            suffix = output_path.suffix.lower()

            if suffix == ".csv":
                df.to_csv(output_path, index=False)
            elif suffix in (".xlsx", ".xls"):
                df.to_excel(output_path, index=False)
            elif suffix == ".parquet":
                df.to_parquet(output_path)
            else:
                df.to_csv(output_path, index=False)

            logger.info(f"Exported results to {output_path}")

        return df

    def _execute_step(self, df: pd.DataFrame, step: ProcessingStep) -> pd.DataFrame:
        """Execute a single processing step."""
        assert df is not None, "df must be provided"
        params = step.parameters

        if step.operation == OperationType.FILTER:
            from data_processor.vectorized_filter_engine import VectorizedFilterEngine

            if self._filter_engine is None:
                self._filter_engine = VectorizedFilterEngine()

            return self._filter_engine.apply_filter_batch(
                df,
                params.get("filter_type"),
                params.get("filter_params", {}),
                signal_names=params.get("signals"),
            )

        elif step.operation == OperationType.CALCULATE:
            from data_processor.core.signal_processing import apply_custom_variable

            return apply_custom_variable(
                df,
                params.get("column_name"),
                params.get("formula"),
            )

        elif step.operation == OperationType.RESAMPLE:
            from data_processor.core.signal_processing import resample_data

            return resample_data(
                df,
                params.get("time_column"),
                params.get("rule"),
                method=params.get("method", "mean"),
            )

        elif step.operation == OperationType.INTEGRATE:
            from data_processor.core.signal_processing import integrate_signals

            return integrate_signals(
                df,
                params.get("time_column"),
                params.get("signals"),
                method=params.get("method", "trapezoidal"),
            )

        elif step.operation == OperationType.DIFFERENTIATE:
            from data_processor.core.signal_processing import differentiate_signals

            return differentiate_signals(
                df,
                params.get("time_column"),
                params.get("signals"),
                method=params.get("method", "spline"),
                orders=params.get("orders", [1]),
            )

        elif step.operation == OperationType.TRIM:
            from data_processor.core.signal_processing import trim_time_range

            return trim_time_range(
                df,
                params.get("time_column"),
                start_time=params.get("start_time"),
                end_time=params.get("end_time"),
            )

        elif step.operation == OperationType.SELECT:
            return df[params.get("columns", [])]

        elif step.operation == OperationType.RENAME:
            return df.rename(columns=params.get("mapping", {}))

        else:
            logger.warning(f"Unknown operation: {step.operation}")
            return df
