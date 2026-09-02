"""Wire types for attested session-unit longitudinal analysis.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/longitudinal_types.py`` (144 lines) under
ADR-0046 Stage 1 — step **P15** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``). The implementation is
UpstreamDrift's, carried over rather than reimplemented; its authors retain
authorship.

Decision G1-D1 — the pooled estimator is a named-method pair
-------------------------------------------------------------
P16's row mandates that this tier carry "G1-D1's named-method pair", and the
consequence the plan spells out lands in this file: "the canonical
``PooledAssociationV1`` gains a required method identifier and the union of
both estimators' output fields (UpstreamDrift's ``standard_error``/``p_value``,
Tools' heterogeneity block). D11's per-player uncertainty gap closes in the
same change."

G0 measured why this had to be a pair rather than a winner. The same four
per-player slopes go in — the gate asserts ``max |UD - Tools| = 0.0`` — and the
pooled verdicts come out opposite: UpstreamDrift's cluster-robust interval
``[-1.576, +0.525]`` crosses zero (p = 0.210) while
``rate_of_closure.launch_monitor_longitudinal``'s DerSimonian-Laird interval
``[-1.015, -0.042]`` does not, and the latter additionally reports a 98.3%
improvement probability. The point estimates agree to 0.52%; it is the
uncertainty model that differs, and UpstreamDrift's interval is 2.16x wider.
Cluster-robust inference at four clusters is anti-conservative in the opposite
direction from what a t-distribution assumes, and DerSimonian-Laird with four
studies underestimates ``tau_squared``. Neither is "the" right answer at k = 4,
so both are preserved as named, provenance-carrying options exactly as ADR-0045
preserved the two putting roll models. **Neither is removed**, and a result
never reports one estimator's number under the other's name.

The two identifiers are :data:`PooledMethod`. ``LongitudinalPlayerAssociationV1``
gains the six per-player uncertainty fields that closed D11 —
``standard_error``, ``ci_lower``, ``ci_upper``, ``p_value``, ``r_squared`` and
``first_to_last_change`` — which UpstreamDrift "cannot express today at all".
Every added field is optional, so a fit that cannot support one says so by
absence rather than by inventing a number.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.python.launch_monitor.contract_v2 import (
    AnalysisLineageV2,
    AvailabilityV2,
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
)

LONGITUDINAL_SESSION_CONTRACT_VERSION: Literal[
    "launch-monitor-longitudinal-session/1.0.0"
] = "launch-monitor-longitudinal-session/1.0.0"

PooledMethod = Literal["ud-cluster-robust-fe/1", "dl-random-effects/1"]

POOLED_METHOD_DESCRIPTIONS: dict[str, str] = {
    "ud-cluster-robust-fe/1": (
        "Player fixed-effects OLS with standard errors clustered by player "
        "and a finite-cluster corrected sandwich covariance."
    ),
    "dl-random-effects/1": (
        "Inverse-variance weighting of per-player session-cell slopes with "
        "the DerSimonian-Laird between-player variance estimate."
    ),
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LongitudinalSessionRequestV1(_StrictModel):
    """Scientific choices for session-unit directional association."""

    metric: str = Field(min_length=1)
    direction: Literal["higher_is_better", "lower_is_better", "descriptive_only"] = (
        "descriptive_only"
    )
    session_aggregate: Literal["mean", "median"] = "mean"
    strata: tuple[str, ...] = ()
    confounders: tuple[str, ...] = ()
    minimum_sessions_per_player: int = Field(default=3, ge=3)
    minimum_player_clusters: int = Field(default=4, ge=4)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    pooled_method: PooledMethod = "ud-cluster-robust-fe/1"

    @model_validator(mode="after")
    def validate_design_terms(self) -> LongitudinalSessionRequestV1:
        terms = (*self.strata, *self.confounders)
        if len(terms) != len(set(terms)) or set(self.strata) & set(self.confounders):
            raise ValueError("strata and confounders must be unique and disjoint")
        if self.metric in terms:
            raise ValueError(
                "metric, strata, and confounders must be unique and disjoint"
            )
        if any(not term.strip() for term in terms):
            raise ValueError("strata and confounder names must be non-empty")
        return self


class LongitudinalDesignV1(_StrictModel):
    primary_unit: Literal["player_session_stratum"] = "player_session_stratum"
    session_aggregate: Literal["mean", "median"]
    strata: tuple[str, ...]
    confounders: tuple[str, ...]
    pooled_terms: tuple[str, ...]


class LongitudinalClaimsV1(_StrictModel):
    association_scope: Literal["descriptive_directional"] = "descriptive_directional"
    primary_unit: Literal["player_session_stratum"] = "player_session_stratum"
    shot_level_inference: bool = False
    causal_improvement: bool = False
    confounder_control_is_causal: bool = False


class LongitudinalMissingnessV1(_StrictModel):
    input_row_count: int = Field(ge=0)
    included_shot_count: int = Field(ge=0)
    session_cell_count: int = Field(ge=0)
    excluded_by_reason: dict[str, int]


class SessionAggregateV1(_StrictModel):
    player_id: str
    session_id: str
    order_value: float
    order_unit: str
    stratum: dict[str, str]
    shot_count: int = Field(ge=1)
    metric_value: float
    confounder_values: dict[str, float]


class LongitudinalPlayerAssociationV1(_StrictModel):
    player_id: str
    session_count: int = Field(ge=0)
    estimate_per_order_unit: float | None = None
    direction: Literal["increasing", "decreasing", "flat", "unavailable"]
    state: Literal["available", "unavailable"]
    reason_code: str | None = None
    standard_error: float | None = Field(default=None, ge=0.0)
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    r_squared: float | None = Field(default=None, ge=0.0, le=1.0)
    first_to_last_change: float | None = None


class PooledAssociationV1(_StrictModel):
    """One pooled estimate, always carrying the name of the estimator.

    G1-D1: results from different estimators are never numerically compared
    without the names attached, so ``method`` is required and has no default.
    ``standard_error``, ``confidence_interval_*`` and ``confidence_level`` are
    produced by both estimators; ``p_value`` by ``ud-cluster-robust-fe/1``; the
    heterogeneity block and ``improvement_probability`` by
    ``dl-random-effects/1``.
    """

    method: PooledMethod
    estimate_per_order_unit: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_level: float
    cluster_count: int = Field(ge=1)
    session_cell_count: int = Field(ge=1)
    uncertainty_state: Literal["available"] = "available"
    tau_squared: float | None = Field(default=None, ge=0.0)
    q_statistic: float | None = Field(default=None, ge=0.0)
    i_squared_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    improvement_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_method_consistent_outputs(self) -> PooledAssociationV1:
        heterogeneity = (
            self.tau_squared,
            self.q_statistic,
            self.i_squared_pct,
        )
        if self.method == "ud-cluster-robust-fe/1":
            if any(value is not None for value in heterogeneity):
                raise ValueError(
                    "ud-cluster-robust-fe/1 does not estimate between-player "
                    "heterogeneity"
                )
            if self.improvement_probability is not None:
                raise ValueError(
                    "ud-cluster-robust-fe/1 does not report an improvement probability"
                )
        elif any(value is None for value in heterogeneity):
            raise ValueError(
                "dl-random-effects/1 must report tau_squared, q_statistic, and "
                "i_squared_pct"
            )
        return self


class LongitudinalSessionResultV1(_StrictModel):
    """Evidence-bearing result that never labels association as improvement."""

    contract_version: Literal["launch-monitor-longitudinal-session/1.0.0"] = (
        LONGITUDINAL_SESSION_CONTRACT_VERSION
    )
    status: Literal["available", "partial", "unavailable"]
    request: LongitudinalSessionRequestV1
    design: LongitudinalDesignV1
    session_aggregates: tuple[SessionAggregateV1, ...]
    player_associations: tuple[LongitudinalPlayerAssociationV1, ...]
    pooled_association: PooledAssociationV1 | None
    availability: tuple[AvailabilityV2, ...]
    missingness: LongitudinalMissingnessV1
    lineage: AnalysisLineageV2
    player_identity: PlayerIdentityV2
    session_identity: SessionIdentityV2
    order_evidence: OrderEvidenceV2
    claims: LongitudinalClaimsV1 = Field(default_factory=LongitudinalClaimsV1)
    warnings: tuple[str, ...] = ()


__all__ = [
    "LONGITUDINAL_SESSION_CONTRACT_VERSION",
    "POOLED_METHOD_DESCRIPTIONS",
    "LongitudinalClaimsV1",
    "LongitudinalDesignV1",
    "LongitudinalMissingnessV1",
    "LongitudinalSessionRequestV1",
    "LongitudinalSessionResultV1",
    "LongitudinalPlayerAssociationV1",
    "PooledAssociationV1",
    "PooledMethod",
    "SessionAggregateV1",
]
