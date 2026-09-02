"""Versioned wire models for player covariation and population synthesis.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/player_covariation_types.py`` (335 lines)
under ADR-0046 Stage 1 — step **P18** of the ADR-0046 G1 port plan
(UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``). The
implementation is UpstreamDrift's, carried over rather than reimplemented; its
authors retain authorship.

P18 is a **union port**, not a plain port
-----------------------------------------
ADR-0046's Amendment 1 corrected the record this package was started from:
``player_covariation`` is *not* an UpstreamDrift-only capability. This
repository already carries a same-shaped counterpart — the
``rate_of_closure.player_covariation`` /
``rate_of_closure._player_covariation_scan`` /
``rate_of_closure._player_covariation_types`` trio, 570 lines against
UpstreamDrift's 1,098, with the same three-module within-player + Fisher-z
design. Neither side is a superset, so the plan's taxonomy needed a merge
bucket and P18 fills it: **UpstreamDrift's implementation is the base, and
every capability that exists only in the ``rate_of_closure`` trio is folded in
explicitly.** Nothing from either side is dropped silently.

G0.1 (UpstreamDrift ``tests/integration/launch_monitor_drift/
test_player_covariation_drift.py``) is why the union is safe: on the shared
160-shot fixture it compared 52 scalars across the two stacks and found 51
identical inside UpstreamDrift's declared 12-decimal reporting quantum. The
one exception, ``q_statistic``, differs by 7.577272143066693e-13 — an
accumulation-order artefact, not a method difference. There is no numerical
disagreement to arbitrate here, which is what makes this a union rather than
the named-method pair G1-D1 required for the pooled longitudinal estimator.

Union decisions carried by this module
--------------------------------------
``MIN_FISHER_SAMPLES``
    Folded in from ``rate_of_closure._player_covariation_types``. UpstreamDrift
    embeds the same 4 twice as an anonymous literal
    (``Field(default=4, ge=4)``); the named constant is the
    ``rate_of_closure`` side's contribution and is now the single source of
    that floor for both request models. The value is unchanged, so no request
    that validated before validates differently now.

``PlayerCovariationResultV1.method_description``
    Folded in from ``rate_of_closure``'s ``PlayerCovariationAnalysis``, whose
    field of the same name is one of the two fields G0.1's D26 pin records as
    existing only on the ``rate_of_closure`` side. UpstreamDrift carries a
    method description on the *scan* result but not on the selected-pair
    result, so a caller reading one pair got no statement of what was
    computed. It is a required field: a result that cannot say what it did is
    not a result this layer emits.

The second D26 field, ``backing_data``, is folded in as well but deliberately
not as a wire field — see :func:`~shared.python.launch_monitor.
player_covariation.covariation_backing_frame` for that decision.

Owner ruling D22 — low-degrees-of-freedom Fisher intervals
----------------------------------------------------------
ADR-0048's "Owner Rulings (2026-09-02)" settles the one place the two stacks
disagreed about what to *publish* rather than what to compute. On the shared
fixture ``rate_of_closure`` returns a between-player Fisher-z interval of
``[-0.6655142653044201, 0.9960866924324187]`` from four player means — an
interval on ``n - 3 = 1`` degree of freedom that covers 83% of the coefficient
range and would read the same way for almost any point estimate.
UpstreamDrift withholds it. **Ruling: the canonical layer withholds the
between-player Fisher interval when degrees of freedom make it uninformative —
UpstreamDrift's posture — with the threshold documented and the absence
explained in the result rather than silently ``None``.**

Three things carry that ruling here, so a reader of a result never has to
infer why a bound is missing:

``BETWEEN_PLAYER_INTERVAL_MIN_GROUPS``
    The documented threshold. Five player means, because the Fisher-z standard
    error is ``1 / sqrt(n - 3)``: at ``n = 4`` that is exactly 1.0, a full unit
    of Fisher-z, and ``tanh(+/-1.96)`` then spans ``[-0.96, +0.96]`` whatever
    the estimate. Requiring ``n - 3 >= 2`` is the first sample size at which
    the interval carries information about the coefficient rather than about
    the transform.

``AssociationEstimateV1.interval_withheld_reason``
    The explanation, on the estimate itself. An *available* estimate now
    carries either an interval or a typed reason it has none — never neither,
    enforced by the validator. That also names the within-player scope's
    long-standing absence, which both stacks withheld but neither explained:
    the centred observations are clustered by player, so an unclustered
    Fisher-z interval would be too narrow.

``CovariationUncertaintyV1.between_player_interval_min_groups``
    The threshold restated in the result's own uncertainty block, next to the
    named methods, so a consumer reading only the wire document can check the
    rule that was applied without reading this source.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from shared.python.launch_monitor.contract_v2 import (
    AnalysisLineageV2,
    AvailabilityState,
    AvailabilityV2,
    ClaimsV2,
    MetricUnitsV2,
    PlayerIdentityV2,
    VendorProvenanceV2,
)

PLAYER_COVARIATION_CONTRACT_VERSION: Literal[
    "launch-monitor-player-covariation/1.0.0"
] = "launch-monitor-player-covariation/1.0.0"

MIN_FISHER_SAMPLES: Final[int] = 4
"""Smallest sample a Fisher-z interval is defined on (``n - 3 >= 1``).

Folded in from ``rate_of_closure._player_covariation_types``; see the module
docstring's union decisions.
"""

BETWEEN_PLAYER_INTERVAL_MIN_GROUPS: Final[int] = 5
"""Fewest player means that earn a between-player Fisher interval (ruling D22).

``n - 3 >= 2``. See the module docstring for why four means do not qualify.
"""

AssociationState = Literal["available", "unavailable"]
IntervalWithheldReason = Literal[
    "insufficient_degrees_of_freedom",
    "clustered_observations",
]
AssociationUnavailableReason = Literal[
    "insufficient_samples",
    "insufficient_groups",
    "constant_x",
    "constant_y",
    "constant_both",
]


class _CovariationModel(BaseModel):
    """Strict immutable base for the public covariation records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerCovariationRequestV1(_CovariationModel):
    """Select one variable pair and the player-level inference rules."""

    x_column: str = Field(min_length=1)
    y_column: str = Field(min_length=1)
    player_column: str = Field(min_length=1)
    min_samples: int = Field(default=MIN_FISHER_SAMPLES, ge=MIN_FISHER_SAMPLES)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def require_distinct_variables(self) -> PlayerCovariationRequestV1:
        if self.x_column == self.y_column:
            raise ValueError("x_column and y_column must differ")
        if self.player_column in {self.x_column, self.y_column}:
            raise ValueError("player_column cannot also be an analyzed variable")
        return self


class PlayerCovariationScanRequestV1(_CovariationModel):
    """Select a bounded exploratory all-pairs scan."""

    player_column: str = Field(min_length=1)
    numeric_columns: tuple[str, ...] = Field(default=(), max_length=20)
    min_samples: int = Field(default=MIN_FISHER_SAMPLES, ge=MIN_FISHER_SAMPLES)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def require_unique_columns(self) -> PlayerCovariationScanRequestV1:
        if any(not column.strip() for column in self.numeric_columns):
            raise ValueError("numeric_columns values must be non-empty")
        if len(set(self.numeric_columns)) != len(self.numeric_columns):
            raise ValueError("numeric_columns values must be unique")
        return self


class AssociationEstimateV1(_CovariationModel):
    """One descriptive association or a typed unavailable state."""

    state: AssociationState
    reason_code: AssociationUnavailableReason | None = None
    sample_count: int = Field(ge=0)
    group_count: int = Field(ge=0)
    pearson_r: float | None = Field(default=None, ge=-1, le=1)
    spearman_r: float | None = Field(default=None, ge=-1, le=1)
    slope: float | None = None
    intercept: float | None = None
    r_squared: float | None = Field(default=None, ge=0, le=1)
    ci_lower: float | None = Field(default=None, ge=-1, le=1)
    ci_upper: float | None = Field(default=None, ge=-1, le=1)
    interval_withheld_reason: IntervalWithheldReason | None = None

    @model_validator(mode="after")
    def require_consistent_state(self) -> AssociationEstimateV1:
        estimates = (
            self.pearson_r,
            self.spearman_r,
            self.slope,
            self.intercept,
            self.r_squared,
        )
        if self.state == "available" and (
            self.reason_code is not None or any(value is None for value in estimates)
        ):
            raise ValueError("available association requires all point estimates")
        if self.state == "unavailable" and (
            self.reason_code is None or any(value is not None for value in estimates)
        ):
            raise ValueError(
                "unavailable association requires reason_code and null estimates"
            )
        if (self.ci_lower is None) != (self.ci_upper is None):
            raise ValueError("association interval bounds must be supplied together")
        if self.state == "available" and (self.ci_lower is None) == (
            self.interval_withheld_reason is None
        ):
            raise ValueError(
                "available association requires either an interval or exactly one "
                "reason it was withheld"
            )
        if self.state == "unavailable" and self.interval_withheld_reason is not None:
            raise ValueError(
                "unavailable association cannot withhold an interval it never had"
            )
        return self


class PlayerAssociationV1(_CovariationModel):
    """A player-level estimate and normalized meta-analysis weights."""

    player_id: str = Field(min_length=1)
    estimate: AssociationEstimateV1
    fixed_weight: float | None = Field(default=None, ge=0, le=1)
    random_weight: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_consistent_weights(self) -> PlayerAssociationV1:
        if (self.fixed_weight is None) != (self.random_weight is None):
            raise ValueError("fixed and random weights must be supplied together")
        if self.estimate.state == "unavailable" and self.fixed_weight is not None:
            raise ValueError("unavailable player estimates cannot carry weights")
        return self


class MetaAnalysisSummaryV1(_CovariationModel):
    """Fixed/random Fisher-z synthesis with heterogeneity diagnostics."""

    state: AssociationState
    reason_code: Literal["insufficient_eligible_players"] | None = None
    contributor_count: int = Field(ge=0)
    total_sample_count: int = Field(ge=0)
    fixed_effect_r: float | None = Field(default=None, ge=-1, le=1)
    fixed_ci_lower: float | None = Field(default=None, ge=-1, le=1)
    fixed_ci_upper: float | None = Field(default=None, ge=-1, le=1)
    random_effect_r: float | None = Field(default=None, ge=-1, le=1)
    random_ci_lower: float | None = Field(default=None, ge=-1, le=1)
    random_ci_upper: float | None = Field(default=None, ge=-1, le=1)
    tau_squared: float | None = Field(default=None, ge=0)
    q_statistic: float | None = Field(default=None, ge=0)
    i_squared_pct: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def require_consistent_state(self) -> MetaAnalysisSummaryV1:
        values = (
            self.fixed_effect_r,
            self.fixed_ci_lower,
            self.fixed_ci_upper,
            self.random_effect_r,
            self.random_ci_lower,
            self.random_ci_upper,
            self.tau_squared,
            self.q_statistic,
            self.i_squared_pct,
        )
        if self.state == "available" and (
            self.reason_code is not None or any(value is None for value in values)
        ):
            raise ValueError("available meta-analysis requires all estimates")
        if self.state == "unavailable" and (
            self.reason_code is None or any(value is not None for value in values)
        ):
            raise ValueError(
                "unavailable meta-analysis requires reason_code and null estimates"
            )
        return self


class CovariationMissingnessV1(_CovariationModel):
    """Row and player exclusions used by a selected-pair analysis."""

    input_row_count: int = Field(ge=0)
    pairwise_complete_row_count: int = Field(ge=0)
    missing_by_variable: dict[str, int]
    non_numeric_by_variable: dict[str, int]
    non_finite_by_variable: dict[str, int]
    excluded_by_reason: dict[str, int]
    eligible_player_count: int = Field(ge=0)
    excluded_player_count_by_reason: dict[str, int]
    policy: Literal["pairwise"] = "pairwise"


class CovariationUncertaintyV1(_CovariationModel):
    """Named uncertainty methods and their scientific limits."""

    confidence_level: float = Field(gt=0.5, lt=1.0)
    per_player_interval: Literal["fisher-z"] = "fisher-z"
    pooled_interval: Literal["fisher-z-unclustered"] = "fisher-z-unclustered"
    within_player_interval: Literal["unavailable-clustered"] = "unavailable-clustered"
    between_player_interval: Literal["fisher-z-above-min-groups"] = (
        "fisher-z-above-min-groups"
    )
    between_player_interval_min_groups: int = Field(
        default=BETWEEN_PLAYER_INTERVAL_MIN_GROUPS, ge=MIN_FISHER_SAMPLES
    )
    fixed_effect_method: Literal["inverse-variance-fisher-z"] = (
        "inverse-variance-fisher-z"
    )
    random_effect_method: Literal["dersimonian-laird-fisher-z"] = (
        "dersimonian-laird-fisher-z"
    )
    assumptions: tuple[str, ...]


class CovariationPairRankV1(_CovariationModel):
    """One deterministically ranked pair from an exploratory scan."""

    rank: int = Field(ge=1)
    state: AssociationState
    reason_code: Literal["insufficient_eligible_players"] | None = None
    x_column: str = Field(min_length=1)
    y_column: str = Field(min_length=1)
    x_unit: MetricUnitsV2
    y_unit: MetricUnitsV2
    random_effect_r: float | None = Field(default=None, ge=-1, le=1)
    fixed_effect_r: float | None = Field(default=None, ge=-1, le=1)
    within_player_r: float | None = Field(default=None, ge=-1, le=1)
    between_player_r: float | None = Field(default=None, ge=-1, le=1)
    contributor_count: int = Field(ge=0)
    total_sample_count: int = Field(ge=0)
    input_row_count: int = Field(ge=0)
    pairwise_complete_row_count: int = Field(ge=0)
    excluded_row_count: int = Field(ge=0)
    i_squared_pct: float | None = Field(default=None, ge=0, le=100)
    direction_consistency: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_consistent_state(self) -> CovariationPairRankV1:
        if self.state == "available" and self.reason_code is not None:
            raise ValueError("available ranked pair cannot have a reason_code")
        if self.state == "unavailable" and self.reason_code is None:
            raise ValueError("unavailable ranked pair requires a reason_code")
        if (
            self.pairwise_complete_row_count + self.excluded_row_count
            != self.input_row_count
        ):
            raise ValueError("ranked pair row counts must reconcile to input_row_count")
        return self


class PlayerCovariationResultV1(_CovariationModel):
    """Evidence-bearing result for one selected variable pair."""

    contract_version: Literal["launch-monitor-player-covariation/1.0.0"] = (
        PLAYER_COVARIATION_CONTRACT_VERSION
    )
    analysis_kind: Literal["selected_pair"] = "selected_pair"
    status: AvailabilityState
    request: PlayerCovariationRequestV1
    pooled: AssociationEstimateV1
    within_player: AssociationEstimateV1
    between_player: AssociationEstimateV1
    per_player: tuple[PlayerAssociationV1, ...]
    meta_analysis: MetaAnalysisSummaryV1
    missingness: CovariationMissingnessV1
    units: dict[str, MetricUnitsV2]
    lineage: AnalysisLineageV2
    availability: tuple[AvailabilityV2, ...]
    uncertainty: CovariationUncertaintyV1
    player_identity: PlayerIdentityV2
    vendor_provenance: tuple[VendorProvenanceV2, ...]
    claims: ClaimsV2 = Field(default_factory=ClaimsV2)
    definitions: dict[str, str]
    warnings: tuple[str, ...]
    method_description: str = Field(min_length=1)


class PlayerCovariationScanResultV1(_CovariationModel):
    """Evidence-bearing deterministic exploratory pair ranking."""

    contract_version: Literal["launch-monitor-player-covariation/1.0.0"] = (
        PLAYER_COVARIATION_CONTRACT_VERSION
    )
    analysis_kind: Literal["pair_scan"] = "pair_scan"
    status: AvailabilityState
    request: PlayerCovariationScanRequestV1
    pair_count: int = Field(ge=0)
    available_pair_count: int = Field(ge=0)
    unavailable_pair_count: int = Field(ge=0)
    ranking: tuple[CovariationPairRankV1, ...]
    lineage: AnalysisLineageV2
    player_identity: PlayerIdentityV2
    vendor_provenance: tuple[VendorProvenanceV2, ...]
    claims: ClaimsV2 = Field(default_factory=ClaimsV2)
    warnings: tuple[str, ...]
    method_description: str

    @model_validator(mode="after")
    def require_consistent_counts(self) -> PlayerCovariationScanResultV1:
        available = sum(item.state == "available" for item in self.ranking)
        unavailable = len(self.ranking) - available
        if (
            self.pair_count != len(self.ranking)
            or self.available_pair_count != available
            or self.unavailable_pair_count != unavailable
        ):
            raise ValueError("pair counts must match the ranked pair states")
        if tuple(item.rank for item in self.ranking) != tuple(
            range(1, self.pair_count + 1)
        ):
            raise ValueError("ranked pairs must use consecutive one-based ranks")
        expected_status: AvailabilityState = (
            "available"
            if unavailable == 0
            else "unavailable"
            if available == 0
            else "partial"
        )
        if self.status != expected_status:
            raise ValueError("scan status must match the ranked pair states")
        return self


CovariationResultUnion = Annotated[
    PlayerCovariationResultV1 | PlayerCovariationScanResultV1,
    Field(discriminator="analysis_kind"),
]


class PlayerCovariationContractV1(RootModel[CovariationResultUnion]):
    """Schema root covering selected-pair and exploratory-scan results."""

    model_config = ConfigDict(frozen=True)


__all__ = [
    "BETWEEN_PLAYER_INTERVAL_MIN_GROUPS",
    "MIN_FISHER_SAMPLES",
    "PLAYER_COVARIATION_CONTRACT_VERSION",
    "AssociationEstimateV1",
    "CovariationMissingnessV1",
    "CovariationPairRankV1",
    "CovariationUncertaintyV1",
    "IntervalWithheldReason",
    "MetaAnalysisSummaryV1",
    "PlayerAssociationV1",
    "PlayerCovariationContractV1",
    "PlayerCovariationRequestV1",
    "PlayerCovariationResultV1",
    "PlayerCovariationScanRequestV1",
    "PlayerCovariationScanResultV1",
]
