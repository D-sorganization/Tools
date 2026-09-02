"""Wire models for governed launch-monitor scoring analyses.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/strokes_gained_types.py`` (440 lines) under
ADR-0046 Stage 1 — step **P12** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``). The implementation is
UpstreamDrift's, carried over rather than reimplemented; its authors retain
authorship.

The baseline half stayed behind, on purpose
-------------------------------------------
P12's row reads "``strokes_gained_types.py`` (minus baseline half)", and the
port plan names that exclusion as the one sub-module in the whole inventory
that is *genuinely already home*:

    the expected-strokes **baseline half** of ``strokes_gained_types.py`` is
    genuinely already home. [...] That half retires into the Tools module at
    port time; only the request/result/uncertainty half travels.

The evidence is numerical, not editorial. G0's
``test_baseline_table_digest_agrees_across_stacks`` runs UpstreamDrift's
``baseline_table_sha256`` and this repository's
``rate_of_closure.launch_monitor_strokes_gained_baseline.baseline_table_hash``
over the same states and pins both to the identical digest
``188a6eafa9eebd8a0f4c9ba288d858ad359e35999ba2706989c75d349f509925``, and the
``rate_of_closure`` module additionally carries a ``MAX_BASELINE_BYTES`` cap
and source-URL validation UpstreamDrift has no equivalent of. Six definitions
therefore did **not** travel — ``ExpectedStrokesStateV2``,
``ExpectedStrokesBaselineV2``, ``baseline_table_sha256`` and its three private
canonicalisation helpers — because porting them would install a second,
weaker copy of an authority this repository already owns.

What travels instead is a *structural* view of that artifact:
:class:`ExpectedStrokesStateLike` and :class:`ExpectedStrokesBaselineLike`.
:mod:`shared.python.launch_monitor.strokes_gained` types its ``baseline``
argument against them, so the already-home loader's
``StrokesGainedBaseline`` satisfies the canonical analysis without this
package importing ``rate_of_closure`` — which would invert the layer
direction and is exactly the convenience seam the port plan's name-collision
risk warns against. ``BASELINE_CONTRACT_VERSION`` stays because
:class:`BaselineProvenanceV1` — a *result* field — records it, and its value
is the same string the already-home module publishes as its
``CONTRACT_VERSION``.

Everything else in this module is UpstreamDrift's, unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.python.launch_monitor.contract_v2 import AnalysisContextV2

BASELINE_CONTRACT_VERSION: Literal["launch-monitor-strokes-gained-baseline/2.0.0"] = (
    "launch-monitor-strokes-gained-baseline/2.0.0"
)
STROKES_GAINED_CONTRACT_VERSION: Literal[
    "launch-monitor-strokes-gained-analysis/1.0.0"
] = "launch-monitor-strokes-gained-analysis/1.0.0"
OUTCOME_PROXY_CONTRACT_VERSION: Literal["launch-monitor-outcome-proxy/1.0.0"] = (
    "launch-monitor-outcome-proxy/1.0.0"
)

TrustedGrouping = Literal[
    "explicit_user_attested",
    "pseudonymous_stable",
    "verified_external",
]
ResultStatus = Literal["available", "partial", "unavailable"]
DistanceUnit = Literal["yd", "m"]

LongitudinalMethod = Literal[
    "session-cell-sg-trend/1",
    "shot-level-sg-trend/1",
]


@runtime_checkable
class ExpectedStrokesStateLike(Protocol):
    """Structural view of one benchmark point in an expected-strokes table.

    Stands in for the ``ExpectedStrokesStateV2`` that did not travel. Both the
    already-home ``rate_of_closure.launch_monitor_strokes_gained_baseline``
    ``BaselineState`` and UpstreamDrift's pydantic model satisfy it.
    """

    lie: str
    context: str
    target: str
    distance_yards: float
    expected_strokes: float
    standard_error: float | None


@runtime_checkable
class ExpectedStrokesBaselineLike(Protocol):
    """Structural view of a hash-verified expected-strokes baseline artifact.

    Loading, byte-capping, source-URL validation, and digest verification are
    the already-home module's job; this package only reads the artifact.
    """

    baseline_id: str
    version: str
    source_url: str
    license: str
    table_sha256: str
    states: Sequence[ExpectedStrokesStateLike]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CourseStateColumnsV1(_ContractModel):
    lie_column: str = Field(min_length=1)
    context_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    distance_column: str = Field(min_length=1)
    distance_unit: DistanceUnit


class GroupingDimensionV1(_ContractModel):
    dimension: Literal["player", "session", "club"]
    column: str = Field(min_length=1)
    trust_level: TrustedGrouping
    evidence: str = Field(min_length=1)

    @field_validator("column", "evidence")
    @classmethod
    def strip_group_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("grouping column and evidence must be non-empty")
        return normalized


class LongitudinalDimensionV1(_ContractModel):
    order_column: str = Field(min_length=1)
    order_unit: str = Field(min_length=1)
    group_column: str | None = None
    group_dimension: Literal["player", "session", "club"] | None = None
    trust_level: TrustedGrouping
    evidence: str = Field(min_length=1)
    min_samples: int = Field(default=3, ge=3)
    method: LongitudinalMethod = "session-cell-sg-trend/1"

    @field_validator("order_column", "order_unit", "evidence")
    @classmethod
    def strip_longitudinal_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("longitudinal fields and evidence must be non-empty")
        return normalized

    @model_validator(mode="after")
    def require_complete_group_mapping(self) -> LongitudinalDimensionV1:
        if (self.group_column is None) != (self.group_dimension is None):
            raise ValueError("longitudinal group column and dimension must be paired")
        return self


class StrokesGainedRequestV1(_ContractModel):
    start: CourseStateColumnsV1
    finish: CourseStateColumnsV1
    shot_id_column: str | None = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    min_samples: int = Field(default=3, ge=1)
    summaries: tuple[GroupingDimensionV1, ...] = ()
    longitudinal: LongitudinalDimensionV1 | None = None

    @model_validator(mode="after")
    def require_unique_summary_dimensions(self) -> StrokesGainedRequestV1:
        dimensions = [summary.dimension for summary in self.summaries]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("summary dimensions must be unique")
        return self


class CourseStateValueV1(_ContractModel):
    lie: str
    context: str
    target: str
    distance_yards: float = Field(ge=0.0)


class InterpolationV1(_ContractModel):
    lower_distance_yards: float
    upper_distance_yards: float
    fraction: float = Field(ge=0.0, le=1.0)


class StrokesGainedRowV1(_ContractModel):
    source_index: int = Field(ge=0)
    shot_id: str | None = None
    input_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start: CourseStateValueV1
    finish: CourseStateValueV1
    expected_start: float
    expected_finish: float
    benchmark_standard_error: float | None = Field(default=None, ge=0.0)
    strokes_gained: float
    start_interpolation: InterpolationV1
    finish_interpolation: InterpolationV1
    groups: dict[str, str] = Field(default_factory=dict)
    longitudinal_order: float | None = None


class ExcludedRowV1(_ContractModel):
    source_index: int = Field(ge=0)
    shot_id: str | None = None
    reason_code: Literal[
        "missing_course_state",
        "invalid_distance",
        "outside_baseline",
    ]
    message: str


class ExclusionSummaryV1(_ContractModel):
    input_row_count: int = Field(ge=0)
    included_row_count: int = Field(ge=0)
    total_excluded: int = Field(ge=0)
    by_reason: dict[str, int]


class ConfidenceIntervalV1(_ContractModel):
    lower: float
    upper: float
    level: float
    method: str


class EstimateSummaryV1(_ContractModel):
    count: int = Field(ge=0)
    mean: float | None = None
    standard_deviation: float | None = Field(default=None, ge=0.0)
    standard_error: float | None = Field(default=None, ge=0.0)
    confidence_interval: ConfidenceIntervalV1 | None = None


class AvailabilityV1(_ContractModel):
    state: Literal["available", "unavailable"]
    reason_code: str | None = None
    message: str | None = None
    observed_count: int = Field(ge=0)
    required_count: int = Field(ge=0)


class StrokesGainedUncertaintyV1(_ContractModel):
    sampling_method: str
    confidence_level: float
    benchmark_method: str
    benchmark_standard_error_mean: float | None = Field(default=None, ge=0.0)
    assumptions: tuple[str, ...]


class GroupSummaryV1(_ContractModel):
    dimension: Literal["player", "session", "club"]
    group_value: str
    estimate: EstimateSummaryV1
    trust_level: TrustedGrouping
    evidence: str


class LongitudinalSummaryV1(_ContractModel):
    group_dimension: Literal["player", "session", "club", "all"]
    group_value: str
    method: LongitudinalMethod
    sample_count: int = Field(ge=3)
    slope: float
    intercept: float
    r_squared: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    slope_unit: str
    trust_level: TrustedGrouping
    evidence: str


class BaselineProvenanceV1(_ContractModel):
    baseline_id: str
    version: str
    source_url: str
    license: str
    table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_version: str


class StrokesGainedClaimsV1(_ContractModel):
    is_strokes_gained: Literal[True] = True
    source_backed: Literal[True] = True
    device_emulation: Literal[False] = False
    device_certification: Literal[False] = False
    causal_inference: Literal[False] = False


class StrokesGainedAnalysisResultV1(_ContractModel):
    contract_version: Literal["launch-monitor-strokes-gained-analysis/1.0.0"] = (
        STROKES_GAINED_CONTRACT_VERSION
    )
    status: ResultStatus
    metric_name: Literal["source_backed_strokes_gained"] = (
        "source_backed_strokes_gained"
    )
    unit: Literal["strokes"] = "strokes"
    value_summary: EstimateSummaryV1
    baseline: BaselineProvenanceV1
    formula: str
    units: dict[str, str]
    availability: AvailabilityV1
    uncertainty: StrokesGainedUncertaintyV1
    row_results: tuple[StrokesGainedRowV1, ...]
    excluded_rows: tuple[ExcludedRowV1, ...]
    exclusions: ExclusionSummaryV1
    group_summaries: tuple[GroupSummaryV1, ...] = ()
    longitudinal_summaries: tuple[LongitudinalSummaryV1, ...] = ()
    analysis_context: AnalysisContextV2
    dataset_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claims: StrokesGainedClaimsV1 = Field(default_factory=StrokesGainedClaimsV1)
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...]


class OutcomeProxyRequestV1(_ContractModel):
    carry_column: str = Field(min_length=1)
    lateral_column: str = Field(min_length=1)
    carry_unit: DistanceUnit
    lateral_unit: DistanceUnit
    target_distance_yards: float = Field(gt=0.0)
    shot_id_column: str | None = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    min_samples: int = Field(default=1, ge=1)


class OutcomeProxyRowV1(_ContractModel):
    source_index: int = Field(ge=0)
    shot_id: str | None = None
    carry_yards: float
    lateral_yards: float
    target_distance_yards: float
    radial_error_yards: float = Field(ge=0.0)


class OutcomeProxyClaimsV1(_ContractModel):
    is_strokes_gained: Literal[False] = False
    source_backed: Literal[False] = False
    causal_inference: Literal[False] = False


class OutcomeProxyResultV1(_ContractModel):
    contract_version: Literal["launch-monitor-outcome-proxy/1.0.0"] = (
        OUTCOME_PROXY_CONTRACT_VERSION
    )
    status: ResultStatus
    metric_name: Literal["expected_proximity_dispersion_proxy"] = (
        "expected_proximity_dispersion_proxy"
    )
    unit: Literal["yd"] = "yd"
    value_summary: EstimateSummaryV1
    row_results: tuple[OutcomeProxyRowV1, ...]
    exclusions: ExclusionSummaryV1
    formula: str
    units: dict[str, str]
    claims: OutcomeProxyClaimsV1 = Field(default_factory=OutcomeProxyClaimsV1)
    limitations: tuple[str, ...]


__all__ = [
    "BASELINE_CONTRACT_VERSION",
    "OUTCOME_PROXY_CONTRACT_VERSION",
    "STROKES_GAINED_CONTRACT_VERSION",
    "AvailabilityV1",
    "BaselineProvenanceV1",
    "ConfidenceIntervalV1",
    "CourseStateColumnsV1",
    "CourseStateValueV1",
    "EstimateSummaryV1",
    "ExcludedRowV1",
    "ExclusionSummaryV1",
    "ExpectedStrokesBaselineLike",
    "ExpectedStrokesStateLike",
    "GroupingDimensionV1",
    "GroupSummaryV1",
    "InterpolationV1",
    "LongitudinalDimensionV1",
    "LongitudinalMethod",
    "LongitudinalSummaryV1",
    "OutcomeProxyRequestV1",
    "OutcomeProxyResultV1",
    "OutcomeProxyRowV1",
    "StrokesGainedAnalysisResultV1",
    "StrokesGainedRequestV1",
    "StrokesGainedRowV1",
    "StrokesGainedUncertaintyV1",
]
