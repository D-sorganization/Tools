"""Synthetic multi-area process and reusable high-performance faceplate contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

SignalQuality = Literal["good", "uncertain", "bad", "stale", "simulated"]
OperatingMode = Literal["off", "manual", "automatic", "unavailable"]
AlarmState = Literal["normal", "active", "shelved", "suppressed"]
InterlockState = Literal["clear", "permissive_missing", "tripped"]


def _synthetic_identifier(value: str) -> str:
    normalized = value.strip()
    if not normalized.startswith("SYNTHETIC."):
        raise ValueError("identifiers must begin with SYNTHETIC.")
    return normalized


class FaceplateValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float
    unit: str = Field(min_length=1, max_length=24)
    source_timestamp: datetime


class AssetFaceplate(BaseModel):
    """One reusable operator-facing asset summary."""

    model_config = ConfigDict(frozen=True)

    asset_id: str
    label: str = Field(min_length=1, max_length=100)
    asset_type: Literal["pump", "valve", "vessel", "heater", "separator"]
    primary_value: FaceplateValue
    quality: SignalQuality
    mode: OperatingMode
    alarm_state: AlarmState
    interlock_state: InterlockState
    detail_route: str = Field(pattern=r"^/operator/assets/[A-Za-z0-9._-]+$")
    trend_tags: tuple[str, ...] = Field(min_length=1)

    _validate_asset_id = field_validator("asset_id")(_synthetic_identifier)

    @field_validator("trend_tags")
    @classmethod
    def _validate_trend_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_synthetic_identifier(value) for value in values)


class ProcessArea(BaseModel):
    model_config = ConfigDict(frozen=True)

    area_id: str
    label: str = Field(min_length=1, max_length=100)
    detail_route: str = Field(pattern=r"^/operator/areas/[A-Za-z0-9._-]+$")
    assets: tuple[AssetFaceplate, ...] = Field(min_length=1)

    _validate_area_id = field_validator("area_id")(_synthetic_identifier)


class ProcessOverview(BaseModel):
    model_config = ConfigDict(frozen=True)

    overview_id: str
    title: str = Field(min_length=1, max_length=200)
    areas: tuple[ProcessArea, ...] = Field(min_length=2)
    data_classification: Literal["synthetic"]
    not_for_live_control: Literal[True]

    _validate_overview_id = field_validator("overview_id")(_synthetic_identifier)


def _asset(
    asset_id: str,
    label: str,
    asset_type: Literal["pump", "valve", "vessel", "heater", "separator"],
    value: float,
    unit: str,
    *,
    mode: OperatingMode = "automatic",
) -> AssetFaceplate:
    return AssetFaceplate(
        asset_id=asset_id,
        label=label,
        asset_type=asset_type,
        primary_value=FaceplateValue(
            value=value,
            unit=unit,
            source_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        quality="simulated",
        mode=mode,
        alarm_state="normal",
        interlock_state="clear",
        detail_route=f"/operator/assets/{asset_id}",
        trend_tags=(f"{asset_id}.PV", f"{asset_id}.SP"),
    )


def synthetic_process_overview() -> ProcessOverview:
    """Return the fixed representative process; it contains no plant identifiers."""
    return ProcessOverview(
        overview_id="SYNTHETIC.PROCESS",
        title="Representative Process Overview",
        data_classification="synthetic",
        not_for_live_control=True,
        areas=(
            ProcessArea(
                area_id="SYNTHETIC.FEED",
                label="Feed Preparation",
                detail_route="/operator/areas/SYNTHETIC.FEED",
                assets=(
                    _asset("SYNTHETIC.FEED.PUMP", "Feed Pump", "pump", 62.0, "%"),
                    _asset("SYNTHETIC.FEED.VALVE", "Feed Valve", "valve", 58.0, "%"),
                ),
            ),
            ProcessArea(
                area_id="SYNTHETIC.REACTOR",
                label="Reaction",
                detail_route="/operator/areas/SYNTHETIC.REACTOR",
                assets=(
                    _asset("SYNTHETIC.REACTOR.VESSEL", "Reactor", "vessel", 72.0, "°C"),
                    _asset("SYNTHETIC.REACTOR.HEATER", "Heater", "heater", 41.0, "%"),
                ),
            ),
            ProcessArea(
                area_id="SYNTHETIC.SEPARATION",
                label="Separation",
                detail_route="/operator/areas/SYNTHETIC.SEPARATION",
                assets=(
                    _asset(
                        "SYNTHETIC.SEPARATION.VESSEL",
                        "Separator",
                        "separator",
                        48.0,
                        "%",
                    ),
                ),
            ),
        ),
    )
