"""Serializable state for the unified tools sidebar."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_DOCK_AREAS = {"left", "right"}


@dataclass(slots=True)
class SidebarState:
    """JSON-safe dock/tab state shared by host applications."""

    dock_area: str = "right"
    floating: bool = False
    minimized: bool = False
    width: int = 360
    height: int = 720
    active_tab: str = "files"
    tab_order: list[str] = field(default_factory=list)
    hidden_tabs: list[str] = field(default_factory=list)
    popped_out_tabs: list[str] = field(default_factory=list)
    tab_display_names: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dock_area not in VALID_DOCK_AREAS:
            self.dock_area = "right"
        self.width = max(240, int(self.width))
        self.height = max(240, int(self.height))
        if not self.active_tab:
            self.active_tab = "files"
        self.tab_order = _dedupe_strings(self.tab_order)
        self.hidden_tabs = _dedupe_strings(self.hidden_tabs)
        self.popped_out_tabs = _dedupe_strings(self.popped_out_tabs)
        self.tab_display_names = _string_mapping(self.tab_display_names)

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
            minimized=bool(payload.get("minimized", False)),
            width=int(payload.get("width", 360)),
            height=int(payload.get("height", 720)),
            active_tab=str(payload.get("active_tab", "files")),
            tab_order=_string_list(payload.get("tab_order")),
            hidden_tabs=_string_list(payload.get("hidden_tabs")),
            popped_out_tabs=_string_list(payload.get("popped_out_tabs")),
            tab_display_names=_string_mapping(payload.get("tab_display_names")),
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe_strings(value: list[str] | None) -> list[str]:
    if not value:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        name = str(raw_value).strip()
        if key and name:
            result[key] = name
    return result
