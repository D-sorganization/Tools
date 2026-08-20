"""Validated HTTP client seam for the canonical UpstreamDrift v2 contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ResidualAvailability:
    state: str
    reason: str
    rows: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class AnalysisResponseV2:
    contract_version: str
    payload: dict[str, object]
    row_aligned_residuals: ResidualAvailability


def validate_v2_response(value: object) -> AnalysisResponseV2:
    if not isinstance(value, dict) or value.get("contract_version") != "2.0.0":
        raise ValueError(
            "Upstream analysis response has an unsupported contract version"
        )
    required = {
        "status",
        "analysis",
        "units",
        "lineage",
        "missingness",
        "availability",
        "uncertainty",
        "player_identity",
        "vendor_provenance",
        "claims",
        "warnings",
    }
    if not required.issubset(value):
        raise ValueError("Upstream v2 response is missing required contract fields")
    claims = value.get("claims")
    if (
        not isinstance(claims, dict)
        or claims.get("device_emulation") is not False
        or claims.get("device_certification") is not False
    ):
        raise ValueError(
            "Upstream response makes an unsupported device emulation or "
            "certification claim"
        )
    lineage = value.get("lineage")
    if not isinstance(lineage, dict) or not isinstance(
        lineage.get("backing_records"), list
    ):
        raise ValueError("Upstream v2 lineage/backing-record contract is invalid")
    analysis = value.get("analysis")
    residual_payload = (
        analysis.get("row_aligned_residuals") if isinstance(analysis, dict) else None
    )
    if isinstance(residual_payload, list) and len(residual_payload) == len(
        lineage["backing_records"]
    ):
        residuals = ResidualAvailability(
            "available",
            "v2 row-aligned residuals match backing records",
            tuple(residual_payload),
        )
    else:
        residuals = ResidualAvailability(
            "unavailable",
            "The canonical v2 response does not provide row-aligned residuals "
            "matching backing records.",
        )
    return AnalysisResponseV2("2.0.0", value, residuals)


class UpstreamV2Client:
    """Small replaceable client; statistics remain in UpstreamDrift."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def analyze(self, payload: dict[str, object]) -> AnalysisResponseV2:
        request = Request(
            f"{self.base_url}/tools/launch-monitor-analytics/v2/analyze",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - configured local/private authority
            value: Any = json.loads(response.read().decode("utf-8"))
        return validate_v2_response(value)
