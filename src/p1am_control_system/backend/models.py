# mypy: ignore-errors
from datetime import datetime

from pydantic import BaseModel, model_validator
from sqlmodel import Field, SQLModel


class TagLog(SQLModel, table=True):
    """SQLModel representing a logged tag state in the database."""

    id: int | None = Field(default=None, primary_key=True)
    tag_name: str = Field(index=True)
    value: float
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        index=True,
    )


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
        default_factory=datetime.utcnow,
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


class InterlockConfig(BaseModel):
    """Pydantic model validating 4-tier limits for a tag."""

    lolo_limit: float
    low_limit: float
    high_limit: float
    hihi_limit: float

    @model_validator(mode="after")
    def validate_limits(self) -> "InterlockConfig":
        if self.low_limit > self.high_limit:
            raise ValueError("low_limit must be less than or equal to high_limit")
        if self.lolo_limit > self.low_limit:
            raise ValueError("lolo_limit must be less than or equal to low_limit")
        if self.high_limit > self.hihi_limit:
            raise ValueError("high_limit must be less than or equal to hihi_limit")
        return self


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
