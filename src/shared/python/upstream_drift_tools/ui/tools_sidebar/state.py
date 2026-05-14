"""Serializable state for the unified tools sidebar."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VALID_DOCK_AREAS = {"left", "right"}


@dataclass(slots=True)
class SidebarState:
    """JSON-safe dock/tab state shared by host applications."""

    dock_area: str = "right"
    floating: bool = False
    width: int = 360
    height: int = 720
    active_tab: str = "files"

    def __post_init__(self) -> None:
        if self.dock_area not in VALID_DOCK_AREAS:
            self.dock_area = "right"
        self.width = max(240, int(self.width))
        self.height = max(240, int(self.height))
        if not self.active_tab:
            self.active_tab = "files"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SidebarState:
        """Create state from a partial or stale payload."""
        if not payload:
            return cls()
        return cls(
            dock_area=str(payload.get("dock_area", "right")),
            floating=bool(payload.get("floating", False)),
            width=int(payload.get("width", 360)),
            height=int(payload.get("height", 720)),
            active_tab=str(payload.get("active_tab", "files")),
        )

    def save_json(self, path: str | Path) -> None:
        """Persist state to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> SidebarState:
        """Load state from ``path``. Missing files return defaults."""
        source = Path(path)
        if not source.exists():
            return cls()
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
