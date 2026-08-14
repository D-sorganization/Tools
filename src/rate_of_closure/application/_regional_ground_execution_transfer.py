"""Strict transfer-settings projection for regional-ground execution jobs."""

from __future__ import annotations

from typing import Any, cast

from rate_of_closure.application._regional_ground_execution_job_values import (
    MAX_EXECUTION_TIMEOUT_S,
    canonical_text,
    integer,
    positive,
    sha256,
)
from rate_of_closure.application._workspace_validation import exact_mapping
from shared.python.swing_sim.flight import FlightGroundTransferSettings
from shared.python.swing_sim.ground import (
    GroundCalibration,
    GroundProvenance,
    GroundSurfaceProfile,
)
from shared.python.swing_sim.ground.contract_wire import (
    record_from_dict,
    record_to_dict,
)

_TRANSFER_FIELDS = frozenset(
    {
        "request_id",
        "surface",
        "calibration",
        "provenance",
        "max_time_s",
        "output_interval_s",
        "max_events",
        "rotational_inertia_factor",
        "surface_sha256",
        "settings_sha256",
    }
)


def transfer_payload(settings: FlightGroundTransferSettings) -> dict[str, Any]:
    """Return transfer inputs plus canonical surface and settings identities."""
    base = {
        "request_id": settings.request_id,
        "surface": record_to_dict(settings.surface),
        "calibration": record_to_dict(settings.calibration),
        "provenance": record_to_dict(settings.provenance),
        "max_time_s": settings.max_time_s,
        "output_interval_s": settings.output_interval_s,
        "max_events": settings.max_events,
        "rotational_inertia_factor": settings.rotational_inertia_factor,
    }
    return {
        **base,
        "surface_sha256": sha256(base["surface"]),
        "settings_sha256": sha256(base),
    }


def _records(
    data: dict[str, Any],
) -> tuple[GroundSurfaceProfile, GroundCalibration, GroundProvenance]:
    return (
        cast(
            GroundSurfaceProfile,
            record_from_dict(GroundSurfaceProfile, data["surface"]),
        ),
        cast(
            GroundCalibration,
            record_from_dict(GroundCalibration, data["calibration"]),
        ),
        cast(GroundProvenance, record_from_dict(GroundProvenance, data["provenance"])),
    )


def transfer_from_dict(value: object) -> FlightGroundTransferSettings:
    """Parse exact transfer settings and verify both embedded digests."""
    data = exact_mapping(value, _TRANSFER_FIELDS, "transfer")
    surface, calibration, provenance = _records(dict(data))
    max_time_s = positive(
        data["max_time_s"], "transfer max_time_s", MAX_EXECUTION_TIMEOUT_S
    )
    settings = FlightGroundTransferSettings(
        canonical_text(data["request_id"], "transfer request_id"),
        surface,
        calibration,
        provenance,
        max_time_s,
        positive(data["output_interval_s"], "transfer output_interval_s", max_time_s),
        integer(data["max_events"], "transfer max_events", 1, 10_000),
        positive(data["rotational_inertia_factor"], "rotational_inertia_factor", 1.0),
    )
    expected = transfer_payload(settings)
    if data["surface_sha256"] != expected["surface_sha256"]:
        raise ValueError("surface_sha256 must match the embedded surface authority")
    if data["settings_sha256"] != expected["settings_sha256"]:
        raise ValueError("settings_sha256 must match the transfer settings authority")
    return settings


__all__ = ["transfer_from_dict", "transfer_payload"]
