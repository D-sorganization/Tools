"""Time step management for time-dependent field data visualization.

This module provides time stepping and seeking functionality for animating
time-dependent FEA field data. It supports:
- Sequential stepping through time steps
- Direct seeking to specific time indices
- Time range queries and property access
- Field data access at current time step

Key classes:
- TimeStepManager: Core time stepping state machine
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TimeStepManager:
    """Manages time stepping for time-dependent field data.

    This class handles sequential and random-access navigation through
    a sequence of time steps and associated field data. It tracks the
    current time step index and provides methods for advancing, seeking,
    and querying time state.

    Attributes:
        time_steps: 1D array of time values (must be monotonically increasing)
        field_data_list: List of field data dicts, one per time step
        current_step_index: Index of current time step (0-indexed)
    """

    def __init__(
        self,
        time_steps: np.ndarray,
        field_data_list: list[dict[str, Any]],
    ) -> None:
        """Initialize TimeStepManager.

        Args:
            time_steps: 1D numpy array of time values. Must have same length
                as field_data_list.
            field_data_list: List of field data dictionaries, one per time step.
                Each dict contains field names as keys and field data as values.

        Raises:
            ValueError: If time_steps and field_data_list have different lengths.
            TypeError: If time_steps is not a numpy array or field_data_list
                is not a list.
        """
        if not isinstance(time_steps, np.ndarray):
            raise TypeError(f"time_steps must be numpy array, got {type(time_steps)}")

        if not isinstance(field_data_list, list):
            raise TypeError(
                f"field_data_list must be list, got {type(field_data_list)}"
            )

        if len(time_steps) != len(field_data_list):
            raise ValueError(
                f"time_steps and field_data_list must have equal length. "
                f"Got {len(time_steps)} and {len(field_data_list)}"
            )

        self.time_steps = time_steps
        self.field_data_list = field_data_list
        self.current_step_index = 0

        logger.debug(
            "TimeStepManager initialized with %d steps from %.4f to %.4f",
            len(time_steps),
            float(time_steps[0]),
            float(time_steps[-1]),
        )

    @property
    def total_steps(self) -> int:
        """Return total number of time steps."""
        return len(self.time_steps)

    @property
    def current_time(self) -> float:
        """Return current time value."""
        return float(self.time_steps[self.current_step_index])

    @property
    def current_step_index(self) -> int:
        """Return current step index (0-indexed)."""
        return self._current_step_index

    @current_step_index.setter
    def current_step_index(self, value: int) -> None:
        """Set current step index."""
        if not isinstance(value, int):
            raise TypeError(f"Step index must be int, got {type(value)}")
        if value < 0 or value >= self.total_steps:
            max_idx = self.total_steps - 1
            raise ValueError(f"Step index {value} out of bounds [0, {max_idx}]")
        self._current_step_index = value

    @property
    def time_range(self) -> tuple[float, float]:
        """Return (min_time, max_time) tuple."""
        return (float(self.time_steps[0]), float(self.time_steps[-1]))

    @property
    def is_at_end(self) -> bool:
        """Return True if at last time step."""
        return self.current_step_index == self.total_steps - 1

    def get_current_field(self) -> dict[str, Any]:
        """Get field data at current time step.

        Returns:
            Dictionary of field data at current step. Keys are field names,
            values are field arrays/data.
        """
        return self.field_data_list[self.current_step_index]

    def advance_step(self) -> bool:
        """Advance to next time step.

        Returns:
            True if advanced successfully, False if already at last step
            and cannot advance further.
        """
        if self.is_at_end:
            logger.debug("Already at last step, cannot advance")
            return False

        self._current_step_index += 1
        logger.debug(
            "Advanced to step %d (time %.4f)",
            self.current_step_index,
            self.current_time,
        )
        return True

    def seek_to_step(self, index: int) -> None:
        """Seek to specified time step index.

        Args:
            index: Target step index (0-indexed).

        Raises:
            ValueError: If index is out of bounds.
            TypeError: If index is not an integer.
        """
        if not isinstance(index, int):
            raise TypeError(f"Step index must be int, got {type(index)}")

        if index < 0 or index >= self.total_steps:
            raise ValueError(
                f"Step index {index} out of bounds [0, {self.total_steps - 1}]"
            )

        self._current_step_index = index
        logger.debug(
            "Seeked to step %d (time %.4f)",
            self.current_step_index,
            self.current_time,
        )

    def reset(self) -> None:
        """Reset to first time step."""
        self._current_step_index = 0
        logger.debug("Reset to first step")
