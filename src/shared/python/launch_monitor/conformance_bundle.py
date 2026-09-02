"""Versioned, data-free conformance bundle for launch-monitor consumers.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/conformance_bundle.py`` (207 lines) under
ADR-0046 Stage 1 — step **P17** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``). The implementation is
UpstreamDrift's, carried over unchanged rather than reimplemented; its authors
retain authorship. No behaviour is added, removed, or limited by the move: this
module is **AST-identical** to UpstreamDrift's modulo this docstring and the
plan's ``src.shared.python.launch_monitor.X`` to ``shared.python.launch_monitor.X``
import rewrite.

Why this row waited for P18
---------------------------
P17 is the one step in the port order whose blocker is not recorded in the
order itself. ``ConformancePayload`` is a five-way discriminated union and
``_PAYLOAD_TYPES`` is its runtime map, and one of the five arms is
``player_covariation_types.PlayerCovariationResultV1`` — a module the plan
classified ``needs-decision`` and landed at **P18**, one row *after* this one.
The dependency is a runtime import, not a type-checking-only one, so the module
could not be imported at all before P18. It is the same class of edge the plan
caught for ``contract_v2`` -> ``flexible_analysis`` and did record. P18 landed
the symbol under exactly the name UpstreamDrift's conformance payloads expect,
so no adaptation is needed here and the port stays pure.

What the bundle is for
----------------------
Ten scenarios: every one of the five analysis kinds in both an ``available``
and an ``unavailable`` status, each carrying the uniform consumer evidence the
v2 contract promises (units, claims, player/session/order identity, source and
backing lineage, exclusion counts) and none carrying an input row. Both the
scenario and the bundle are content-addressed over their own canonical JSON
with the self-referential hash field removed, and both hashes are re-derived
and compared inside the model validator — so a bundle that has been edited in
transit fails to validate rather than validating into something subtly
different.

This is a **consumer** contract, and this repository is one of the consumers:
``src/rate_of_closure/web/src/model/launchMonitorConformanceGolden.test.ts``
and ``tests/rate_of_closure/test_launch_monitor_conformance_golden.py`` both
read a committed golden bundle and drive it through the TypeScript and Python
v2 client validators. That golden is the cross-runtime obligation this module
now defines the Python authority for, and P17's stay-green gate is that it
keeps passing untouched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.python.launch_monitor.contract_v2 import (
    BackingRecordV2,
    LaunchMonitorAnalysisResultV2,
    MetricUnitsV2,
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
    SourceFileReferenceV2,
)
from shared.python.launch_monitor.longitudinal_types import (
    LongitudinalSessionResultV1,
)
from shared.python.launch_monitor.player_covariation_types import (
    PlayerCovariationResultV1,
)
from shared.python.launch_monitor.strokes_gained_types import (
    OutcomeProxyResultV1,
    StrokesGainedAnalysisResultV1,
)

LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION: Literal[
    "launch-monitor-analytics-conformance/1.0.0"
] = "launch-monitor-analytics-conformance/1.0.0"

AnalysisKind = Literal[
    "analysis_v2",
    "player_covariation",
    "attested_longitudinal",
    "source_backed_strokes_gained",
    "distance_target_proxy",
]
ExpectedStatus = Literal["available", "unavailable"]
ConformancePayload = Annotated[
    LaunchMonitorAnalysisResultV2
    | PlayerCovariationResultV1
    | LongitudinalSessionResultV1
    | StrokesGainedAnalysisResultV1
    | OutcomeProxyResultV1,
    Field(discriminator="contract_version"),
]

_PAYLOAD_TYPES: dict[str, type[BaseModel]] = {
    "analysis_v2": LaunchMonitorAnalysisResultV2,
    "player_covariation": PlayerCovariationResultV1,
    "attested_longitudinal": LongitudinalSessionResultV1,
    "source_backed_strokes_gained": StrokesGainedAnalysisResultV1,
    "distance_target_proxy": OutcomeProxyResultV1,
}
_REQUIRED_CASES = frozenset(
    (kind, status) for kind in _PAYLOAD_TYPES for status in ("available", "unavailable")
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LaunchMonitorConformanceScenarioV1(_StrictModel):
    """One synthetic output case with uniform consumer evidence."""

    scenario_id: str = Field(min_length=1)
    analysis_kind: AnalysisKind
    expected_status: ExpectedStatus
    description: str = Field(min_length=1)
    units: dict[str, MetricUnitsV2] = Field(min_length=1)
    claims: dict[str, bool | str] = Field(min_length=1)
    player_identity: PlayerIdentityV2
    session_identity: SessionIdentityV2
    order_evidence: OrderEvidenceV2
    sources: tuple[SourceFileReferenceV2, ...] = Field(min_length=1)
    backing_records: tuple[BackingRecordV2, ...] = Field(min_length=1)
    exclusions: dict[str, int]
    payload: ConformancePayload
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("exclusions")
    @classmethod
    def require_nonnegative_exclusions(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key or count < 0 for key, count in value.items()):
            raise ValueError("exclusion keys must be non-empty and counts non-negative")
        return value

    @model_validator(mode="after")
    def validate_case(self) -> LaunchMonitorConformanceScenarioV1:
        expected_type = _PAYLOAD_TYPES[self.analysis_kind]
        if not isinstance(self.payload, expected_type):
            raise ValueError("analysis_kind does not match the result payload contract")
        if self.payload.status != self.expected_status:
            raise ValueError("expected_status does not match the result payload status")
        source_ids = {source.source_id for source in self.sources}
        if any(record.source_id not in source_ids for record in self.backing_records):
            raise ValueError("every backing record must join to a declared source_id")
        if self.claims.get("causal_inference") is not False:
            raise ValueError("conformance scenarios must forbid causal inference")
        if self.scenario_sha256 != launch_monitor_conformance_scenario_sha256(self):
            raise ValueError(
                "scenario_sha256 does not match canonical scenario content"
            )
        return self


class LaunchMonitorConformanceBundleV1(_StrictModel):
    """Complete consumer bundle with canonical content-address verification."""

    bundle_version: Literal["launch-monitor-analytics-conformance/1.0.0"] = (
        LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION
    )
    description: str = Field(min_length=1)
    data_classification: Literal["synthetic_contract_fixture_no_private_rows"] = (
        "synthetic_contract_fixture_no_private_rows"
    )
    input_records_embedded: Literal[False] = False
    scenarios: tuple[LaunchMonitorConformanceScenarioV1, ...] = Field(min_length=10)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_bundle(self) -> LaunchMonitorConformanceBundleV1:
        identities = [scenario.scenario_id for scenario in self.scenarios]
        if len(identities) != len(set(identities)):
            raise ValueError("scenario_id values must be unique")
        cases = {
            (scenario.analysis_kind, scenario.expected_status)
            for scenario in self.scenarios
        }
        if cases != _REQUIRED_CASES or len(self.scenarios) != len(_REQUIRED_CASES):
            raise ValueError(
                "bundle must contain exactly one required conformance case"
            )
        if self.bundle_sha256 != launch_monitor_conformance_bundle_sha256(self):
            raise ValueError("bundle_sha256 does not match canonical bundle content")
        return self


def launch_monitor_conformance_scenario_sha256(
    scenario: LaunchMonitorConformanceScenarioV1 | dict[str, Any],
) -> str:
    """Hash a scenario while excluding its self-referential hash field."""

    payload = (
        scenario.model_dump(mode="json")
        if isinstance(scenario, LaunchMonitorConformanceScenarioV1)
        else dict(scenario)
    )
    payload.pop("scenario_sha256", None)
    return _canonical_sha256(payload)


def launch_monitor_conformance_bundle_sha256(
    bundle: LaunchMonitorConformanceBundleV1 | dict[str, Any],
) -> str:
    """Hash a bundle while excluding its self-referential hash field."""

    payload = (
        bundle.model_dump(mode="json")
        if isinstance(bundle, LaunchMonitorConformanceBundleV1)
        else dict(bundle)
    )
    payload.pop("bundle_sha256", None)
    return _canonical_sha256(payload)


def launch_monitor_conformance_bundle_json_schema() -> dict[str, Any]:
    """Return the strict OpenAPI-compatible conformance bundle schema."""

    return LaunchMonitorConformanceBundleV1.model_json_schema()


__all__ = [
    "LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION",
    "LaunchMonitorConformanceBundleV1",
    "LaunchMonitorConformanceScenarioV1",
    "launch_monitor_conformance_bundle_json_schema",
    "launch_monitor_conformance_bundle_sha256",
    "launch_monitor_conformance_scenario_sha256",
]
