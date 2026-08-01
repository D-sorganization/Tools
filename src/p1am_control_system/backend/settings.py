"""Central Settings surface for the P1AM backend runtime."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SQLITE_SYNCHRONOUS_MODES = {"OFF", "NORMAL", "FULL", "EXTRA"}
ALICAT_CONNECTION_TYPES = {"mock", "serial", "tcp"}


class P1AMSettings(BaseSettings):
    """Validated runtime configuration for the P1AM backend.

    Legacy ``PLC_*`` variables are accepted for compatibility; new P1AM-scoped
    names are preferred so backend tunables have one discoverable surface.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    plc_driver: str = Field(
        # "simulator" matches the PLCFactory branch; the default must not fall
        # through to the "unknown driver" warning on every bench boot.
        default="simulator",
        validation_alias=AliasChoices("P1AM_PLC_DRIVER", "PLC_DRIVER"),
    )
    plc_ip: str = Field(
        default="192.168.1.100",
        validation_alias=AliasChoices("P1AM_PLC_IP", "PLC_IP"),
    )
    plc_port: int = Field(
        default=502,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("P1AM_PLC_PORT", "PLC_PORT"),
    )
    poll_interval_s: float = Field(
        default=0.1, gt=0.0, validation_alias="P1AM_POLL_INTERVAL_S"
    )
    lightweight_poll_interval_s: float = Field(
        default=2.0,
        gt=0.0,
        validation_alias="P1AM_LIGHTWEIGHT_POLL_INTERVAL_S",
        description=(
            "Poll/broadcast interval used in 'lightweight' performance mode. "
            "Slower than poll_interval_s to cut CPU + HMI re-render load."
        ),
    )
    capture_interval_s: float = Field(
        default=5.0,
        ge=0.0,
        validation_alias="P1AM_CAPTURE_INTERVAL_S",
        description=(
            "Minimum seconds between historian writes. The scan loop still runs "
            "(and the live stream updates) every poll; only persistence is "
            "decimated to this period so the DB doesn't bloat. 0 = every scan. "
            "Operator-adjustable at runtime via /api/capture/config."
        ),
    )
    connect_retry_interval_s: float = Field(
        default=5.0,
        gt=0.0,
        validation_alias="P1AM_CONNECT_RETRY_INTERVAL_S",
    )
    historian_max_bytes: int = Field(
        default=1 * 1024**3,
        ge=0,
        validation_alias="P1AM_HISTORIAN_MAX_BYTES",
    )
    historian_retention_interval_s: float = Field(
        default=300.0,
        gt=0.0,
        validation_alias="P1AM_HISTORIAN_RETENTION_INTERVAL_S",
    )
    sqlite_synchronous: Literal["OFF", "NORMAL", "FULL", "EXTRA"] = Field(
        default="NORMAL",
        validation_alias="P1AM_SQLITE_SYNCHRONOUS",
    )
    require_read_auth: bool = Field(
        default=False,
        validation_alias="P1AM_REQUIRE_READ_AUTH",
        description=(
            "Opt-in gate for the historian/plant read surface (/api/trends, "
            "/api/export, /api/snapshot, /api/events, /api/plant, "
            "/api/project/ladder-explorer, /api/explorer/*). Default False keeps "
            "those routes public so the HMI works in bench mode. When True (and "
            "P1AM_DEV_NO_AUTH is off) a valid operator/admin API key is required."
        ),
    )

    alicat_connection_type: str = Field(
        default="mock",
        validation_alias="P1AM_ALICAT_CONNECTION_TYPE",
        description=(
            "Transport for the Alicat mass flow controllers: 'mock' (simulated "
            "flow, bench only), 'serial', or 'tcp'. 'mock' is refused at "
            "startup when plc_driver is a real PLC — an operator must never be "
            "able to command a purge against simulated gas control (#4031)."
        ),
    )
    alicat_port_or_ip: str | None = Field(
        default=None,
        validation_alias="P1AM_ALICAT_PORT_OR_IP",
        description=(
            "Serial device (e.g. /dev/ttyUSB0) or host/IP shared by the Alicat "
            "MFCs; each controller is addressed by its unit ID. Required when "
            "alicat_connection_type is 'serial' or 'tcp'."
        ),
    )

    @field_validator("alicat_connection_type", mode="before")
    @classmethod
    def _normalize_alicat_connection_type(cls, value: object) -> str:
        connection_type = str(value).strip().lower()
        if connection_type not in ALICAT_CONNECTION_TYPES:
            raise ValueError(
                f"alicat_connection_type must be one of "
                f"{sorted(ALICAT_CONNECTION_TYPES)}; got {value!r}"
            )
        return connection_type

    @field_validator("alicat_port_or_ip", mode="before")
    @classmethod
    def _normalize_alicat_port_or_ip(cls, value: object) -> str | None:
        if value is None:
            return None
        port_or_ip = str(value).strip()
        return port_or_ip or None

    @field_validator("plc_driver", mode="before")
    @classmethod
    def _normalize_driver(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("plc_port", mode="before")
    @classmethod
    def _coerce_plc_port(cls, value: object) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return 502

    @field_validator("sqlite_synchronous", mode="before")
    @classmethod
    def _normalize_sqlite_synchronous(cls, value: object) -> str:
        mode = str(value).strip().upper()
        return mode if mode in SQLITE_SYNCHRONOUS_MODES else "NORMAL"


@lru_cache(maxsize=1)
def get_settings() -> P1AMSettings:
    """Return process-wide P1AM settings resolved from the environment."""
    return P1AMSettings()
