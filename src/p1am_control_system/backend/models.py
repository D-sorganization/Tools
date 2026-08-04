from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

import hardware
from pydantic import BaseModel, field_validator
from sqlalchemy import Index
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    """Return an aware UTC timestamp for database defaults."""
    return datetime.now(UTC)


def _validate_loop_tag(value: str) -> str:
    """Validate a PID loop tag name against the firmware tag contract.

    Accepts a well-formed ``TAG_<n>`` with ``n`` in ``[0, TAG_COUNT)`` or the
    firmware ``kUnmappedTag`` sentinel (``TAG_255``). Any other value — an empty
    string, a non-``TAG_`` name, a non-numeric or out-of-range index — is
    rejected so an invalid loop config cannot persist and later fault a control
    endpoint (e.g. a KeyError-500 in tuning start/step). Mirrors the acceptance
    policy of ``modbus_codec.tag_to_index`` (issue #3745).
    """
    if value == hardware.UNMAPPED_TAG_NAME:
        return value
    # Delegate to the single strict parser; it raises ValueError/TypeError on a
    # malformed or out-of-range name.
    hardware.tag_index(value)
    return value


class TagLog(SQLModel, table=True):
    """SQLModel representing a logged tag state in the database.

    The composite ``(tag_name, timestamp)`` index serves the historian read hot
    path (``WHERE tag_name=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp``)
    as a pure indexed range scan — no temp-B-tree sort — and also covers
    ``tag_name``-only lookups, so no separate single-column ``tag_name`` index is
    needed. ``timestamp`` keeps its own index for the retention sweep's
    ``timestamp``-only range deletes.
    """

    __table_args__ = (Index("ix_taglog_tag_name_timestamp", "tag_name", "timestamp"),)

    id: int | None = Field(default=None, primary_key=True)
    tag_name: str
    value: float
    source_timestamp: datetime | None = Field(default_factory=utc_now)
    timestamp: datetime = Field(
        default_factory=utc_now,
        index=True,
    )
    quality: str = Field(default="uncertain", index=True)
    diagnostic_reason: str | None = Field(default="legacy_unqualified")
    sequence: int = Field(default=0, index=True)
    source: str = Field(default="legacy.adapter", index=True)


class PlantArea(SQLModel, table=True):
    """SQLModel representing a physical plant area."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)


class PlantUnit(SQLModel, table=True):
    """SQLModel representing a plant unit within an area."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    area_id: int = Field(foreign_key="plantarea.id")


class PlantEquipment(SQLModel, table=True):
    """SQLModel representing an equipment module within a unit."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    unit_id: int = Field(foreign_key="plantunit.id")


class TagDefinitionDb(SQLModel, table=True):
    """SQLModel representing a DB-backed tag definition."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    tag_type: str
    description: str = Field(default="")
    rw_mode: str = Field(default="Read-only")
    register_type: str | None = Field(default=None)
    register_num: int | None = Field(default=None)
    data_format: str | None = Field(default=None)
    scale_factor: float | None = Field(default=None)
    equipment_id: int | None = Field(default=None, foreign_key="plantequipment.id")


class EventLog(SQLModel, table=True):
    """SQLModel representing an event or alarm log in the database."""

    id: int | None = Field(default=None, primary_key=True)
    event_type: str = Field(index=True)  # ALARM, SYSTEM, ACKNOWLEDGE
    description: str
    severity: int = Field(default=0)  # 0: Normal, 1: High/Low, 2: HiHi/LoLo
    timestamp: datetime = Field(
        default_factory=utc_now,
        index=True,
    )


class PIDConfig(BaseModel):
    """Pydantic model validating a PID loop configuration."""

    pv_tag: str
    cv_tag: str
    setpoint: float
    kp: float
    ki: float
    kd: float

    @field_validator("pv_tag", "cv_tag")
    @classmethod
    def _check_loop_tag(cls, value: str) -> str:
        return _validate_loop_tag(value)


class InterlockConfig(BaseModel):
    """Pydantic model validating 4-tier limits for a tag."""

    lolo_limit: float
    low_limit: float
    high_limit: float
    hihi_limit: float


class RoutingConfig(BaseModel):
    """Pydantic model validating the complete DCS config routing matrix."""

    input_routing: list[str]
    output_routing: list[str]
    pids: list[PIDConfig]
    interlocks: dict[str, InterlockConfig]


class PIDTuningStepPayload(BaseModel):
    """Pydantic model validating PID tuning step value command."""

    step_value: float


class AlicatSetpointPayload(BaseModel):
    """Pydantic model validating setpoint changes."""

    setpoint: float


class AlicatGasPayload(BaseModel):
    """Pydantic model validating gas select changes."""

    gas: str


class AlicatMFCState(BaseModel):
    """Pydantic model representing serialized Alicat MFC state."""

    device_id: str
    name: str
    gas: str
    setpoint: float
    mass_flow: float
    volumetric_flow: float
    pressure: float
    temperature: float
    max_flow: float
    connection_type: str
    port_or_ip: str | None
    connection_state: str
