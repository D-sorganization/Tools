"""Immersion checklist widget types: ImmersionTask, ImmersionChecklistPanel."""

from dataclasses import dataclass
from typing import Any

from ._base import PanelStyle


@dataclass
class ImmersionTask:
    """A single task in the immersive learning checklist."""

    task_id: str
    title: str
    description: str
    is_complete: bool = False


class ImmersionChecklistPanel:
    """Curated list of activities to guide educational exploration."""

    def __init__(
        self,
        position: tuple[int, int] = (20, 250),
        width: int = 360,
        style: PanelStyle | None = None,
        tasks: list[ImmersionTask] | None = None,
    ):
        """Initialize the checklist panel."""
        assert position is not None, "position must be provided"
        self.position = position
        self.width = width
        self.style = style or PanelStyle()
        self.visible = True
        self._tasks: dict[str, ImmersionTask] = {}
        self._initialize_tasks(tasks)

    def _initialize_tasks(self, tasks: list[ImmersionTask] | None) -> None:
        """Initialize checklist with default or provided tasks."""
        default_tasks = tasks or [
            ImmersionTask(
                task_id="select_body",
                title="Pick a world",
                description=(
                    "Use number keys or click to focus a planet and open its"
                    " fact sheet."
                ),
            ),
            ImmersionTask(
                task_id="navigate_time",
                title="Travel through time",
                description=(
                    "Use the date picker or time navigation hotkeys to see planetary"
                    " alignments."
                ),
            ),
            ImmersionTask(
                task_id="toggle_overlays",
                title="Tune the overlays",
                description=(
                    "Toggle orbits, labels, and the grid to compare scales and"
                    " visibility."
                ),
            ),
            ImmersionTask(
                task_id="historical_events",
                title="Explore mission history",
                description=(
                    "Open the historical events panel and jump to milestone dates."
                ),
            ),
            ImmersionTask(
                task_id="plan_transfer",
                title="Plot a transfer",
                description=(
                    "Plan an Earth→Mars Hohmann transfer to visualize interplanetary"
                    " travel."
                ),
            ),
        ]

        for task in default_tasks:
            self._tasks[task.task_id] = task

    def mark_complete(self, task_id: str) -> None:
        """Mark a checklist task as complete."""
        if task_id in self._tasks:
            self._tasks[task_id].is_complete = True

    def reset(self) -> None:
        """Reset all tasks to incomplete."""
        for task in self._tasks.values():
            task.is_complete = False

    def get_progress(self) -> tuple[int, int]:
        """Return number of completed tasks and total tasks."""
        completed = sum(1 for task in self._tasks.values() if task.is_complete)
        return completed, len(self._tasks)

    def toggle(self) -> None:
        """Toggle visibility of the checklist."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """Get data for rendering."""
        completed, total = self.get_progress()
        tasks = [
            {
                "title": task.title,
                "description": task.description,
                "completed": task.is_complete,
            }
            for task in self._tasks.values()
        ]

        return {
            "position": self.position,
            "width": self.width,
            "tasks": tasks,
            "progress": (completed, total),
            "style": self.style,
            "visible": self.visible,
        }
