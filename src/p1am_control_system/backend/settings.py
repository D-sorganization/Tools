"""Central Settings surface for the P1AM backend runtime."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SQLITE_SYNCHRONOUS_MODES = {"OFF", "NORMAL", "FULL", "EXTRA"}


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
    # --- Remote plant historian (TimescaleDB) forwarding ------------------
    # Off by default: enabling this is a deployment decision, and a backend
    # that has never been configured for a plant historian must behave exactly
    # as it did before. SQLite remains the local source of truth either way;
    # forwarding is strictly additive and best-effort.
    timescale_enabled: bool = Field(
        default=False,
        validation_alias="P1AM_TIMESCALE_ENABLED",
        description=(
            "Enable best-effort forwarding of historian samples to a remote "
            "TimescaleDB plant historian. Requires timescale_dsn. The local "
            "SQLite historian is unaffected."
        ),
    )
    timescale_dsn: str = Field(
        default="",
        validation_alias="P1AM_TIMESCALE_DSN",
        description=(
            "libpq connection string for the plant historian. Never logged in "
            "full — see timescale_writer.redact_dsn."
        ),
    )
    timescale_queue_max: int = Field(
        default=100_000,
        ge=1,
        validation_alias="P1AM_TIMESCALE_QUEUE_MAX",
        description=(
            "Bounded forward-queue depth. On overflow the oldest samples are "
            "dropped and counted. Bounded deliberately: an unbounded queue on "
            "the control Pi is an out-of-memory crash of the controller."
        ),
    )
    timescale_batch_size: int = Field(
        default=1_000,
        ge=1,
        validation_alias="P1AM_TIMESCALE_BATCH_SIZE",
        description="Maximum samples per remote round-trip.",
    )
    timescale_flush_interval_s: float = Field(
        default=1.0,
        gt=0.0,
        validation_alias="P1AM_TIMESCALE_FLUSH_INTERVAL_S",
        description="Maximum time a partial batch waits before being shipped.",
    )
    timescale_connect_timeout_s: float = Field(
        default=5.0,
        gt=0.0,
        validation_alias="P1AM_TIMESCALE_CONNECT_TIMEOUT_S",
        description="Fail-fast bound on historian connection establishment.",
    )
    timescale_shutdown_flush_s: float = Field(
        default=5.0,
        gt=0.0,
        validation_alias="P1AM_TIMESCALE_SHUTDOWN_FLUSH_S",
        description=(
            "Bound on the shutdown flush. Application shutdown must never hang "
            "waiting on an unreachable historian."
        ),
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

    @model_validator(mode="after")
    def _require_dsn_when_timescale_enabled(self) -> P1AMSettings:
        """Reject an enabled-but-unconfigured plant historian at startup.

        Failing loudly here is deliberate. The alternative — starting with
        forwarding "on" but no destination — produces a plant where everyone
        believes history is being recorded off-box and it is not. A historian
        that is silently absent is worse than one that is openly disabled,
        because nobody goes looking for the gap until they need the data.
        """
        if self.timescale_enabled and not self.timescale_dsn.strip():
            raise ValueError(
                "P1AM_TIMESCALE_ENABLED is true but P1AM_TIMESCALE_DSN is empty. "
                "Set a connection string, or disable forwarding explicitly with "
                "P1AM_TIMESCALE_ENABLED=false."
            )
        return self

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
