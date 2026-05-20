from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TagDefinition(BaseModel):
    """DbC Data Model for a SCADA tag, supporting validation and constraints."""

    name: str = Field(
        ..., description="Unique tag name used in SCADA formulas and PLC maps."
    )
    tag_type: Literal["Real", "Boolean", "Integer", "String"] = Field(
        ..., description="Data type of the tag."
    )
    description: str = Field("", description="Human readable description")
    rw_mode: Literal["Read-only", "Read/Write"] = Field("Read-only")

    # PLC map information
    register_type: str | None = Field(default=None, description="E.g. X, Y, C, V")
    register_num: int | None = Field(default=None, description="Numeric address")
    data_format: str | None = Field(
        default=None, description="Sub-type formatting, e.g., B, LB"
    )
    scale_factor: float | None = Field(
        default=None, description="Engineering unit scale factor"
    )

    @field_validator("scale_factor")
    @classmethod
    def validate_scale_factor(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("Scale factor must be strictly positive if provided.")
        return v


class Equipment(BaseModel):
    """Represents a physical device/equipment containing multiple tags."""

    name: str
    tags: dict[str, TagDefinition] = Field(default_factory=dict)


class Unit(BaseModel):
    """Represents a process unit containing multiple equipment modules."""

    name: str
    equipment: dict[str, Equipment] = Field(default_factory=dict)


class Area(BaseModel):
    """Represents a plant area containing multiple process units."""

    name: str
    units: dict[str, Unit] = Field(default_factory=dict)


class Plant(BaseModel):
    """
    Root of the hierarchy. Enforces Law of Demeter by providing facade access methods.
    """

    name: str
    areas: dict[str, Area] = Field(default_factory=dict)

    # Internal fast lookup mapping (Tag Name -> TagDefinition)
    _tag_map: dict[str, TagDefinition] = {}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """Rebuilds the O(1) flat lookup map."""
        self._tag_map = {}
        for area in self.areas.values():
            for unit in area.units.values():
                for eq in unit.equipment.values():
                    for tag_name, tag in eq.tags.items():
                        self._tag_map[tag_name] = tag

    def get_tag(self, tag_name: str) -> TagDefinition:
        """
        Facade method to retrieve a tag without breaking the Law of Demeter.
        """
        if tag_name not in self._tag_map:
            raise KeyError(f"Tag {tag_name} not found in the plant hierarchy.")
        return self._tag_map[tag_name]
