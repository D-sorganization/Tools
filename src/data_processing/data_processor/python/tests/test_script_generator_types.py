"""Tests for data_processor.core.script_generator_types module."""

from __future__ import annotations

from data_processor.core.script_generator_types import (
    OperationType,
    ProcessingPipeline,
    ProcessingStep,
)


class TestOperationType:
    """Tests for OperationType enum."""

    def test_all_values(self) -> None:
        expected = {
            "load",
            "filter",
            "transform",
            "calculate",
            "resample",
            "integrate",
            "differentiate",
            "trim",
            "merge",
            "select",
            "rename",
            "export",
            "custom",
        }
        actual = {m.value for m in OperationType}
        assert actual == expected

    def test_member_count(self) -> None:
        assert len(OperationType) == 13

    def test_from_value(self) -> None:
        assert OperationType("filter") == OperationType.FILTER


class TestProcessingStep:
    """Tests for ProcessingStep dataclass."""

    def test_construction(self) -> None:
        step = ProcessingStep(
            operation=OperationType.FILTER,
            parameters={"cutoff": 10},
            description="Low-pass filter at 10 Hz",
        )
        assert step.operation == OperationType.FILTER
        assert step.parameters == {"cutoff": 10}
        assert step.description == "Low-pass filter at 10 Hz"
        assert step.enabled is True

    def test_disabled(self) -> None:
        step = ProcessingStep(
            operation=OperationType.TRIM,
            parameters={"start": 0, "end": 100},
            enabled=False,
        )
        assert step.enabled is False

    def test_to_dict(self) -> None:
        step = ProcessingStep(
            operation=OperationType.LOAD,
            parameters={"path": "data.csv"},
            description="Load data",
        )
        d = step.to_dict()
        assert d["operation"] == "load"
        assert d["parameters"] == {"path": "data.csv"}
        assert d["description"] == "Load data"
        assert d["enabled"] is True

    def test_from_dict(self) -> None:
        data = {
            "operation": "filter",
            "parameters": {"cutoff": 5},
            "description": "Apply filter",
            "enabled": False,
        }
        step = ProcessingStep.from_dict(data)
        assert step.operation == OperationType.FILTER
        assert step.parameters == {"cutoff": 5}
        assert step.description == "Apply filter"
        assert step.enabled is False

    def test_roundtrip(self) -> None:
        original = ProcessingStep(
            operation=OperationType.TRANSFORM,
            parameters={"method": "log"},
            description="Log transform",
        )
        restored = ProcessingStep.from_dict(original.to_dict())
        assert restored.operation == original.operation
        assert restored.parameters == original.parameters
        assert restored.description == original.description
        assert restored.enabled == original.enabled


class TestProcessingPipeline:
    """Tests for ProcessingPipeline dataclass."""

    def test_construction(self) -> None:
        pipeline = ProcessingPipeline(name="test_pipeline")
        assert pipeline.name == "test_pipeline"
        assert pipeline.description == ""
        assert pipeline.steps == []
        assert pipeline.input_config == {}
        assert pipeline.output_config == {}
        assert pipeline.metadata == {}

    def test_add_step(self) -> None:
        pipeline = ProcessingPipeline(name="p1")
        step = pipeline.add_step(
            OperationType.LOAD,
            {"path": "data.csv"},
            "Load data",
        )
        assert isinstance(step, ProcessingStep)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].operation == OperationType.LOAD

    def test_remove_step(self) -> None:
        pipeline = ProcessingPipeline(name="p1")
        pipeline.add_step(OperationType.LOAD, {"path": "a.csv"})
        pipeline.add_step(OperationType.FILTER, {"cutoff": 10})
        removed = pipeline.remove_step(0)
        assert removed is not None
        assert removed.operation == OperationType.LOAD
        assert len(pipeline.steps) == 1

    def test_remove_step_invalid_index(self) -> None:
        pipeline = ProcessingPipeline(name="p1")
        assert pipeline.remove_step(0) is None
        assert pipeline.remove_step(-1) is None

    def test_move_step(self) -> None:
        pipeline = ProcessingPipeline(name="p1")
        pipeline.add_step(OperationType.LOAD, {})
        pipeline.add_step(OperationType.FILTER, {})
        pipeline.add_step(OperationType.EXPORT, {})
        assert pipeline.move_step(0, 2) is True
        assert pipeline.steps[0].operation == OperationType.FILTER
        # After removal of index 0, remaining are [FILTER, EXPORT],
        # then insert at 2 → may end up at end

    def test_move_step_invalid_index(self) -> None:
        pipeline = ProcessingPipeline(name="p1")
        assert pipeline.move_step(0, 1) is False

    def test_to_dict(self) -> None:
        pipeline = ProcessingPipeline(
            name="my_pipeline",
            description="Test pipeline",
            metadata={"version": 1},
        )
        pipeline.add_step(OperationType.LOAD, {"path": "data.csv"})
        d = pipeline.to_dict()
        assert d["name"] == "my_pipeline"
        assert d["description"] == "Test pipeline"
        assert d["metadata"] == {"version": 1}
        assert len(d["steps"]) == 1

    def test_from_dict(self) -> None:
        data = {
            "name": "restored",
            "description": "From dict",
            "steps": [
                {
                    "operation": "filter",
                    "parameters": {"cutoff": 5},
                },
            ],
            "input_config": {"format": "csv"},
            "output_config": {"format": "xlsx"},
            "metadata": {"author": "test"},
        }
        pipeline = ProcessingPipeline.from_dict(data)
        assert pipeline.name == "restored"
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].operation == OperationType.FILTER
        assert pipeline.input_config == {"format": "csv"}
        assert pipeline.metadata == {"author": "test"}

    def test_roundtrip(self) -> None:
        pipeline = ProcessingPipeline(
            name="roundtrip",
            description="Test roundtrip",
        )
        pipeline.add_step(OperationType.LOAD, {"path": "data.csv"}, "Load")
        pipeline.add_step(OperationType.FILTER, {"cutoff": 10}, "Filter")
        pipeline.add_step(OperationType.EXPORT, {"path": "out.csv"}, "Export")

        restored = ProcessingPipeline.from_dict(pipeline.to_dict())
        assert restored.name == pipeline.name
        assert len(restored.steps) == len(pipeline.steps)
        for orig, rest in zip(pipeline.steps, restored.steps, strict=True):
            assert orig.operation == rest.operation
            assert orig.parameters == rest.parameters
