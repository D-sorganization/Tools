"""Central Settings surface for the P1AM backend runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
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
    require_read_auth: bool = Field(
        default=True,
        validation_alias="P1AM_REQUIRE_READ_AUTH",
        description=(
            "Gate for the historian/plant/configuration read surface "
            "(/api/routing, /api/trends, /api/export, /api/snapshot, "
            "/api/events, /api/plant, /api/alarms/active, /api/capture/*, "
            "/api/performance, /api/alicats, the service /config + /status "
            "pairs, /api/project/ladder-explorer and /api/explorer/*). "
            "Defaults to True (issue #4037): GET /api/routing alone discloses "
            "the full register map, every scale factor and every interlock trip "
            "limit. Set P1AM_REQUIRE_READ_AUTH=0 (or P1AM_DEV_NO_AUTH=1) for a "
            "credential-free bench setup."
        ),
    )

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


_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def read_auth_required(env: Mapping[str, str] | None = None) -> bool:
    """Resolve whether the read surface requires a credential, per request.

    Precedence: an explicit, recognised ``P1AM_REQUIRE_READ_AUTH`` value wins;
    otherwise the :class:`P1AMSettings` default applies. Reading the variable
    directly (rather than the ``lru_cache``d settings singleton) lets an
    operator toggle the gate without restarting the safety-critical controller,
    while an unset variable still inherits the secure-by-default value.

    Args:
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        True when a valid operator/admin key is required for read routes.
    """
    source = os.environ if env is None else env
    raw = source.get("P1AM_REQUIRE_READ_AUTH", "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return get_settings().require_read_auth
