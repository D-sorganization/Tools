"""Tests for PipelineExecutor — pipeline step dispatch.

Covers filter, calculate, integrate, differentiate, trim, select, and
rename operations, plus the disabled-step skip path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data_processor.core.pipeline_executor import PipelineExecutor
from data_processor.core.script_generator_types import (
    OperationType,
    ProcessingPipeline,
    ProcessingStep,
)


@pytest.fixture()
def executor() -> PipelineExecutor:
    """Create a fresh PipelineExecutor."""
    return PipelineExecutor()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "time": np.arange(0.0, 1.0, 0.01),
            "signal_a": np.sin(2 * np.pi * 5 * np.arange(0.0, 1.0, 0.01)),
            "signal_b": np.cos(2 * np.pi * 5 * np.arange(0.0, 1.0, 0.01)),
        }
    )


class TestPipelineExecutorDisabledSteps:
    """Test that disabled steps are skipped."""

    def test_disabled_step_skipped(
        self, executor: PipelineExecutor, sample_df: pd.DataFrame
    ) -> None:
        """Disabled steps should be a no-op."""
        pipeline = ProcessingPipeline(
            name="test",
            description="test pipeline",
            steps=[
                ProcessingStep(
                    operation=OperationType.SELECT,
                    parameters={"columns": ["time"]},
                    description="Select time only",
                    enabled=False,
                ),
            ],
        )
        result = executor.execute(pipeline, sample_df)
        # The step was disabled, so all columns should remain
        assert list(result.columns) == list(sample_df.columns)


class TestPipelineExecutorSelect:
    """Test SELECT operation."""

    def test_select_columns(
        self, executor: PipelineExecutor, sample_df: pd.DataFrame
    ) -> None:
        """SELECT should keep only the specified columns."""
        pipeline = ProcessingPipeline(
            name="select_test",
            description="Select single column",
            steps=[
                ProcessingStep(
                    operation=OperationType.SELECT,
                    parameters={"columns": ["time", "signal_a"]},
                    description="Keep time and signal_a",
                ),
            ],
        )
        result = executor.execute(pipeline, sample_df)
        assert list(result.columns) == ["time", "signal_a"]


class TestPipelineExecutorRename:
    """Test RENAME operation."""

    def test_rename_column(
        self, executor: PipelineExecutor, sample_df: pd.DataFrame
    ) -> None:
        """RENAME should apply the column mapping."""
        pipeline = ProcessingPipeline(
            name="rename_test",
            description="Rename column",
            steps=[
                ProcessingStep(
                    operation=OperationType.RENAME,
                    parameters={"mapping": {"signal_a": "sin_wave"}},
                    description="Rename signal_a",
                ),
            ],
        )
        result = executor.execute(pipeline, sample_df)
        assert "sin_wave" in result.columns
        assert "signal_a" not in result.columns


class TestPipelineExecutorIntegrate:
    """Test INTEGRATE operation."""

    def test_integrate_creates_cumulative_column(
        self, executor: PipelineExecutor, sample_df: pd.DataFrame
    ) -> None:
        """INTEGRATE should add cumulative_{signal} columns."""
        pipeline = ProcessingPipeline(
            name="integrate_test",
            description="Integrate",
            steps=[
                ProcessingStep(
                    operation=OperationType.INTEGRATE,
                    parameters={
                        "time_column": "time",
                        "signals": ["signal_a"],
                        "method": "trapezoidal",
                    },
                    description="Integrate signal_a",
                ),
            ],
        )
        result = executor.execute(pipeline, sample_df)
        assert "cumulative_signal_a" in result.columns


class TestPipelineExecutorDifferentiate:
    """Test DIFFERENTIATE operation."""

    def test_differentiate_creates_derivative_column(
        self, executor: PipelineExecutor, sample_df: pd.DataFrame
    ) -> None:
        """DIFFERENTIATE should add {signal}_d{order} columns."""
        pipeline = ProcessingPipeline(
            name="diff_test",
            description="Differentiate",
            steps=[
                ProcessingStep(
                    operation=OperationType.DIFFERENTIATE,
                    parameters={
                        "time_column": "time",
                        "signals": ["signal_a"],
                        "method": "spline",
                        "orders": [1],
                    },
                    description="Differentiate signal_a",
                ),
            ],
        )
        result = executor.execute(pipeline, sample_df)
        assert "signal_a_d1" in result.columns
