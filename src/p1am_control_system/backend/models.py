from datetime import datetime

from pydantic import BaseModel
from pydantic import Field as PydanticField
from sqlmodel import Field, SQLModel


class TagLog(SQLModel, table=True):  # type: ignore
    """SQLModel representing a logged tag state in the database."""

    id: int | None = Field(default=None, primary_key=True)
    tag_id: int = Field(index=True)
    value: float
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        index=True,
    )


class EventLog(SQLModel, table=True):  # type: ignore
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

    pv_tag_id: int = PydanticField(..., ge=0, le=255)
    cv_tag_id: int = PydanticField(..., ge=0, le=255)
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


class RoutingConfig(BaseModel):
    """Pydantic model validating the complete DCS config routing matrix."""

    input_routing: list[int] = PydanticField(..., min_length=6, max_length=6)
    output_routing: list[int] = PydanticField(..., min_length=2, max_length=2)
    pids: list[PIDConfig] = PydanticField(..., min_length=4, max_length=4)
    interlocks: list[InterlockConfig] = PydanticField(..., min_length=32, max_length=32)


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
