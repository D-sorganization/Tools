"""Tests for PipelineRecorder -- records data processing operations.

Covers: init, recording state, record operations, clear, and assertions.
"""

from __future__ import annotations

import pytest

from data_processor.core.pipeline_recorder import PipelineRecorder
from data_processor.core.script_generator_types import OperationType


class TestPipelineRecorderInit:
    """Test PipelineRecorder initialization."""

    def test_default_name(self) -> None:
        recorder = PipelineRecorder()
        assert recorder.pipeline.name == "Untitled Pipeline"

    def test_custom_name(self) -> None:
        recorder = PipelineRecorder("My Pipeline")
        assert recorder.pipeline.name == "My Pipeline"

    def test_starts_recording(self) -> None:
        recorder = PipelineRecorder()
        assert recorder.is_recording is True


class TestRecordingState:
    """Test start/stop/clear recording."""

    def test_stop_recording(self) -> None:
        recorder = PipelineRecorder()
        recorder.stop_recording()
        assert recorder.is_recording is False

    def test_start_recording(self) -> None:
        recorder = PipelineRecorder()
        recorder.stop_recording()
        recorder.start_recording()
        assert recorder.is_recording is True

    def test_clear_removes_all_steps(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_load("test.csv")
        assert len(recorder.pipeline.steps) == 1
        recorder.clear()
        assert len(recorder.pipeline.steps) == 0


class TestRecordOperations:
    """Test recording different operation types."""

    def test_record_load(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_load("data.csv", file_format="csv")
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.LOAD
        assert step.parameters["file_path"] == "data.csv"

    def test_record_filter(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_filter("lowpass", {"cutoff": 10.0})
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.FILTER
        assert step.parameters["filter_type"] == "lowpass"

    def test_record_transform(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_transform("normalize", {"method": "z-score"})
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.TRANSFORM

    def test_record_calculate(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_calculate("delta_T", "T_out - T_in")
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.CALCULATE
        assert step.parameters["formula"] == "T_out - T_in"

    def test_record_resample(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_resample("time", "1s", method="mean")
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.RESAMPLE

    def test_record_trim(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_trim("time", start_time="0:00", end_time="1:00")
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.TRIM

    def test_record_select(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_select(["col_a", "col_b"])
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.SELECT

    def test_record_export(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_export("output.csv", file_format="csv")
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.EXPORT

    def test_record_custom(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_custom("my_op", {"key": "val"}, description="custom op")
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.CUSTOM

    def test_record_integrate(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_integrate("time", ["signal_a"])
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.INTEGRATE

    def test_record_differentiate(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_differentiate("time", ["signal_a"])
        step = recorder.pipeline.steps[0]
        assert step.operation == OperationType.DIFFERENTIATE


class TestRecordingDisabled:
    """Operations are not recorded when recording is stopped."""

    def test_load_not_recorded(self) -> None:
        recorder = PipelineRecorder()
        recorder.stop_recording()
        recorder.record_load("test.csv")
        assert len(recorder.pipeline.steps) == 0

    def test_filter_not_recorded(self) -> None:
        recorder = PipelineRecorder()
        recorder.stop_recording()
        recorder.record_filter("lowpass", {"cutoff": 5.0})
        assert len(recorder.pipeline.steps) == 0


class TestMultipleSteps:
    """Test recording multiple steps in sequence."""

    def test_step_count(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_load("data.csv")
        recorder.record_filter("lowpass", {"cutoff": 10.0})
        recorder.record_export("output.csv")
        assert len(recorder.pipeline.steps) == 3

    def test_step_order_preserved(self) -> None:
        recorder = PipelineRecorder()
        recorder.record_load("data.csv")
        recorder.record_filter("lowpass", {"cutoff": 10.0})
        assert recorder.pipeline.steps[0].operation == OperationType.LOAD
        assert recorder.pipeline.steps[1].operation == OperationType.FILTER
