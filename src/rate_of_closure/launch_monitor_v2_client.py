"""Validated HTTP client seam for the canonical UpstreamDrift v2 contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
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


@dataclass(frozen=True)
class StrokesGainedResponseV1:
    """Validated canonical source-backed scoring response."""

    status: str
    count: int
    mean: float | None
    payload: dict[str, object]


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


def _safe_scoring_claims(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("Upstream scoring response is missing typed claims")
    if value.get("is_strokes_gained") is not True:
        raise ValueError("Upstream scoring response is not strokes gained")
    if value.get("source_backed") is not True:
        raise ValueError("Upstream scoring response is not source-backed")
    forbidden = ("device_emulation", "device_certification", "causal_inference")
    if any(value.get(claim) is not False for claim in forbidden):
        raise ValueError("Upstream scoring response makes an unsupported claim")


def validate_strokes_gained_response(value: object) -> StrokesGainedResponseV1:
    """Validate the specialized canonical SG result before UI consumption."""

    if not isinstance(value, dict) or value.get("contract_version") != (
        "launch-monitor-strokes-gained-analysis/1.0.0"
    ):
        raise ValueError("Upstream scoring response has an unsupported contract")
    required = {
        "status",
        "metric_name",
        "unit",
        "value_summary",
        "baseline",
        "formula",
        "units",
        "availability",
        "uncertainty",
        "row_results",
        "excluded_rows",
        "exclusions",
        "group_summaries",
        "longitudinal_summaries",
        "analysis_context",
        "dataset_fingerprint_sha256",
        "claims",
        "warnings",
        "limitations",
    }
    if not required.issubset(value):
        raise ValueError("Upstream scoring response is missing required fields")
    if value.get("metric_name") != "source_backed_strokes_gained":
        raise ValueError("Upstream scoring response has the wrong metric")
    _safe_scoring_claims(value.get("claims"))
    summary = value.get("value_summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("count"), int):
        raise ValueError("Upstream scoring response has an invalid value summary")
    mean = summary.get("mean")
    if mean is not None and not isinstance(mean, (int, float)):
        raise ValueError("Upstream scoring mean must be numeric or unavailable")
    return StrokesGainedResponseV1(
        str(value["status"]),
        summary["count"],
        None if mean is None else float(mean),
        value,
    )


class UpstreamV2Client:
    """Small replaceable client; statistics remain in UpstreamDrift."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        normalized = base_url.rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Upstream authority URL must use HTTP(S) with a host")
        if timeout_seconds <= 0:
            raise ValueError("Upstream authority timeout must be positive")
        self.base_url = normalized
        self.timeout_seconds = timeout_seconds

    def analyze(self, payload: dict[str, object]) -> AnalysisResponseV2:
        value = self._post("/tools/launch-monitor-analytics/v2/analyze", payload)
        return validate_v2_response(value)

    def strokes_gained(self, payload: dict[str, object]) -> StrokesGainedResponseV1:
        """Submit one governed source-backed scoring request."""

        value = self._post("/tools/launch-monitor-analytics/v2/strokes-gained", payload)
        return validate_strokes_gained_response(value)

    def _post(self, path: str, payload: dict[str, object]) -> object:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # The constructor admits only HTTP(S) authorities; file/custom schemes fail.
        with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
            value: Any = json.loads(response.read().decode("utf-8"))
        return value


__all__ = [
    "AnalysisResponseV2",
    "ResidualAvailability",
    "StrokesGainedResponseV1",
    "UpstreamV2Client",
    "validate_strokes_gained_response",
    "validate_v2_response",
]
