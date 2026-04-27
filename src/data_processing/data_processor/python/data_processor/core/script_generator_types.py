# mypy: ignore-errors
"""Script generator types and data models.

Contains the core data structures for the script generation system:
- OperationType enum for categorizing operations
- ProcessingStep dataclass for individual operations
- ProcessingPipeline dataclass for operation sequences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperationType(Enum):
    """Types of data processing operations."""

    LOAD = "load"
    FILTER = "filter"
    TRANSFORM = "transform"
    CALCULATE = "calculate"
    RESAMPLE = "resample"
    INTEGRATE = "integrate"
    DIFFERENTIATE = "differentiate"
    TRIM = "trim"
    MERGE = "merge"
    SELECT = "select"
    RENAME = "rename"
    EXPORT = "export"
    CUSTOM = "custom"


@dataclass
class ProcessingStep:
    """A single processing operation."""

    operation: OperationType
    parameters: dict[str, Any]
    description: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation.value,
            "parameters": self.parameters,
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingStep:
        """Create from dictionary."""
        if not (data is not None):
            raise ValueError("data must be provided")
        return cls(
            operation=OperationType(data["operation"]),
            parameters=data["parameters"],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
        )


@dataclass
class ProcessingPipeline:
    """A complete processing pipeline with multiple steps."""

    name: str
    description: str = ""
    steps: list[ProcessingStep] = field(default_factory=list)
    input_config: dict[str, Any] = field(default_factory=dict)
    output_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        operation: OperationType,
        parameters: dict[str, Any],
        description: str = "",
    ) -> ProcessingStep:
        """Add a processing step to the pipeline."""
        if not (operation is not None):
            raise ValueError("operation must be provided")
        step = ProcessingStep(
            operation=operation,
            parameters=parameters,
            description=description,
        )
        self.steps.append(step)
        return step

    def remove_step(self, index: int) -> ProcessingStep | None:
        """Remove a step by index."""
        if not (index is not None):
            raise ValueError("index must be provided")
        if 0 <= index < len(self.steps):
            return self.steps.pop(index)
        return None

    def move_step(self, from_index: int, to_index: int) -> bool:
        """Move a step from one position to another."""
        if not (from_index is not None):
            raise ValueError("from_index must be provided")
        if 0 <= from_index < len(self.steps) and 0 <= to_index < len(self.steps):
            step = self.steps.pop(from_index)
            self.steps.insert(to_index, step)
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "input_config": self.input_config,
            "output_config": self.output_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingPipeline:
        """Create from dictionary."""
        if not (data is not None):
            raise ValueError("data must be provided")
        steps = [ProcessingStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            input_config=data.get("input_config", {}),
            output_config=data.get("output_config", {}),
            metadata=data.get("metadata", {}),
        )
