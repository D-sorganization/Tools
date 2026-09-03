from datetime import datetime, timezone
from typing import Any

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

import hardware
from pydantic import BaseModel, field_validator
from sqlalchemy import DateTime, Index
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel

from shared.python.compatibility import StrEnum


def utc_now() -> datetime:
    """Return an aware UTC timestamp for database defaults."""
    return datetime.now(UTC)


# Spelled as literals so the wire values stay greppable and so the table default
# below does not depend on enum attribute access (which the repo's
# ``--follow-imports=skip`` mypy pass cannot see through).
DATA_SOURCE_LIVE = "live"
DATA_SOURCE_SIMULATED = "simulated"
DATA_SOURCE_HELD = "held"
DATA_SOURCE_FAULT = "fault"


class DataSource(StrEnum):
    """Provenance of a scan's tag values (issue #4004).

    Safety-critical distinction: only :attr:`LIVE` (and :attr:`SIMULATED` on a
    bench where the operator *chose* a simulator driver) is a measurement. A
    :attr:`HELD` or :attr:`FAULT` scan carries no fresh reading and must never
    be routed into the control laws, the alarm engine or the historian's tag
    series — a gap in the trend is truthful, fabricated continuity is not.
    """

    LIVE = DATA_SOURCE_LIVE
    SIMULATED = DATA_SOURCE_SIMULATED
    HELD = DATA_SOURCE_HELD
    FAULT = DATA_SOURCE_FAULT

    @property
    def is_measurement(self) -> bool:
        """True when the values may drive control, alarms and the historian."""
        return self in (DataSource.LIVE, DataSource.SIMULATED)


#: Severity attached to the EventLog row emitted on a data-source transition.
DATA_SOURCE_SEVERITY: dict[str, int] = {
    DATA_SOURCE_LIVE: 0,
    DATA_SOURCE_SIMULATED: 1,
    DATA_SOURCE_HELD: 1,
    DATA_SOURCE_FAULT: 2,
}


def ensure_utc(value: datetime) -> datetime:
    """Normalize a datetime to an aware UTC instant.

    A tz-naive value is *assumed* to already be UTC (that is what the historian
    writes); an aware value is converted. This is the single place the codebase
    decides what "naive means UTC" means — every timestamp crossing an API
    boundary goes through it so no ``isoformat()`` can emit an offset-less
    string (issue #4025).

    Args:
        value: The datetime to normalize.

    Returns:
        The same instant as an aware UTC ``datetime``.

    Raises:
        TypeError: If ``value`` is not a ``datetime``.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"value must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class UtcDateTime(TypeDecorator[datetime]):
    """A ``DATETIME`` column that always round-trips an aware **UTC** instant.

    SQLite has no native timestamp type: SQLAlchemy's SQLite ``DATETIME`` bind
    processor formats the datetime's *wall-clock* fields and silently discards
    ``tzinfo``, and its result processor hands back a tz-**naive** value. So a
    plain ``DateTime(timezone=True)`` column is a lie on this dialect — an aware
    ``05:00-07:00`` was stored as ``05:00`` and read back as ``05:00Z``, an
    8-hour error, and every ``.isoformat()`` on the API boundary emitted an
    offset-less string that the browser then re-parsed as *local* time.

    This decorator closes both ends:
        - bind: normalize to UTC before the dialect drops the offset, so the
          stored wall clock is always UTC (byte-compatible with existing rows,
          which the ``utc_now`` default already wrote as UTC).
        - result: re-attach ``UTC`` so callers always receive aware datetimes.

    Precondition: bound values must be ``datetime`` (or ``None``).
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> Any:
        if value is None:
            return None
        return ensure_utc(value)

    def process_result_value(self, value: datetime | None, dialect: Any) -> Any:
        if value is None:
            return None
        return ensure_utc(value)


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


class TagLog(SQLModel, table=True):  # type: ignore[call-arg]
    """SQLModel representing a logged tag state in the database.

    The composite ``(tag_name, timestamp)`` index serves the historian read hot
    path (``WHERE tag_name=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp``)
    as a pure indexed range scan — no temp-B-tree sort — and also covers
    ``tag_name``-only lookups, so no separate single-column ``tag_name`` index is
    needed. ``timestamp`` keeps its own index for the retention sweep's
    ``timestamp``-only range deletes.

    ``quality`` records the provenance of the sample (see :class:`DataSource`)
    so an analyst reading the trend a year later can tell a real measurement
    from a bench simulation. Rows are only ever written for values that were
    actually measured (or deliberately simulated): a comms outage leaves a gap
    rather than a fabricated continuation of the last reading (issue #4004).

    ``timestamp`` uses :class:`UtcDateTime` so reads return aware-UTC datetimes
    and range bounds are compared in UTC whatever offset the caller supplied.
    """

    __table_args__ = (Index("ix_taglog_tag_name_timestamp", "tag_name", "timestamp"),)

    id: int | None = Field(default=None, primary_key=True)
    tag_name: str
    value: float
    quality: str = Field(default=DATA_SOURCE_LIVE, max_length=16)
    timestamp: datetime = Field(
        default_factory=utc_now,
        index=True,
        sa_type=UtcDateTime,
    )


class PlantArea(SQLModel, table=True):  # type: ignore[call-arg]
    """SQLModel representing a physical plant area."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)


class PlantUnit(SQLModel, table=True):  # type: ignore[call-arg]
    """SQLModel representing a plant unit within an area."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    area_id: int = Field(foreign_key="plantarea.id")


class PlantEquipment(SQLModel, table=True):  # type: ignore[call-arg]
    """SQLModel representing an equipment module within a unit."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    unit_id: int = Field(foreign_key="plantunit.id")


class TagDefinitionDb(SQLModel, table=True):  # type: ignore[call-arg]
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


class EventLog(SQLModel, table=True):  # type: ignore[call-arg]
    """SQLModel representing an event or alarm log in the database.

    ``timestamp`` uses :class:`UtcDateTime` (see :class:`TagLog`) so the
    age-based event retention pass compares real UTC instants.
    """

    id: int | None = Field(default=None, primary_key=True)
    event_type: str = Field(index=True)  # ALARM, SYSTEM, ACKNOWLEDGE
    description: str
    severity: int = Field(default=0)  # 0: Normal, 1: High/Low, 2: HiHi/LoLo
    timestamp: datetime = Field(
        default_factory=utc_now,
        index=True,
        sa_type=UtcDateTime,
    )


def _finite(value: float, field_name: str) -> float:
    """DbC helper: reject NaN/Inf in any value that reaches a register.

    Pydantic v2 accepts JSON ``NaN``/``Infinity`` for a plain ``float`` field,
    and ``float_to_registers`` then raises inside the Modbus client's blanket
    I/O handler -- which used to drop the PLC connection over a bad request
    body (issue #3974). Refuse it at the model boundary instead.
    """
    return hardware.require_finite_value(value, field_name)


class PIDConfig(BaseModel):
    """Pydantic model validating a PID loop configuration."""

    pv_tag: str
    cv_tag: str
    setpoint: float
    kp: float
    ki: float
    kd: float

    @field_validator("setpoint", "kp", "ki", "kd")
    @classmethod
    def _check_finite(cls, value: float, info: Any) -> float:
        return _finite(value, info.field_name)

    @field_validator("pv_tag", "cv_tag")
    @classmethod
    def _check_loop_tag(cls, value: str) -> str:
        return _validate_loop_tag(value)


class InterlockConfig(BaseModel):
    """Pydantic model validating 4-tier limits for a tag.

    A limit of ``None`` means "not interlocked / not alarmed on this side".
    It is encoded to the PLC as the firmware's disabled sentinel
    (``hardware.INTERLOCK_DISABLED_LOW`` / ``_HIGH``) and fed to the alarm
    engine as -inf / +inf, so the tag can never trip or alarm on that side.
    This is the default for every tag that is not a routed input (#4001): the
    old ``low_limit=5.0`` for all 32 tags tripped the firmware on any unrouted
    tag reading 0.0 -- and on a routed thermocouple at room temperature.
    """

    lolo_limit: float | None = None
    low_limit: float | None = None
    high_limit: float | None = None
    hihi_limit: float | None = None

    @field_validator("lolo_limit", "low_limit", "high_limit", "hihi_limit")
    @classmethod
    def _check_finite_or_none(cls, value: float | None, info: Any) -> float | None:
        if value is None:
            return None
        return _finite(value, info.field_name)

    def is_disabled(self) -> bool:
        """True when no side of this tag is interlocked or alarmed."""
        return all(
            limit is None
            for limit in (
                self.lolo_limit,
                self.low_limit,
                self.high_limit,
                self.hihi_limit,
            )
        )

    def engine_limits(self) -> dict[str, float]:
        """Limits for the SCADA alarm engine (Rust or fallback).

        ``None`` becomes -inf (low side) / +inf (high side): every comparison
        against it is False, so the engine never enters that state, while the
        engine's ``lolo <= low <= high <= hihi`` contract still holds.
        """
        neg_inf = float("-inf")
        pos_inf = float("inf")
        return {
            "lolo": neg_inf if self.lolo_limit is None else self.lolo_limit,
            "low": neg_inf if self.low_limit is None else self.low_limit,
            "high": pos_inf if self.high_limit is None else self.high_limit,
            "hihi": pos_inf if self.hihi_limit is None else self.hihi_limit,
        }


class RoutingConfig(BaseModel):
    """Pydantic model validating the complete DCS config routing matrix."""

    input_routing: list[str]
    output_routing: list[str]
    pids: list[PIDConfig]
    interlocks: dict[str, InterlockConfig]


class PIDTuningStepPayload(BaseModel):
    """Pydantic model validating PID tuning step value command."""

    step_value: float

    @field_validator("step_value")
    @classmethod
    def _check_finite(cls, value: float) -> float:
        return _finite(value, "step_value")


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
