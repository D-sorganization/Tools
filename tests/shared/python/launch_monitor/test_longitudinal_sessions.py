"""Contract tests for attested, session-unit longitudinal analysis.

Travels with steps **P15** (``longitudinal_types`` + ``longitudinal_statistics``)
and **P16** (``longitudinal``) of the ADR-0046 G1 port plan, from UpstreamDrift
``tests/unit/launch_monitor/test_longitudinal_sessions.py``.

One of UpstreamDrift's cases does not travel:
``test_published_longitudinal_schema_matches_python_authority`` compares the
generated schema to UpstreamDrift's committed ``docs/api/contracts/`` artifact,
which is UpstreamDrift's published API surface rather than part of this model
layer. Its obligation is asserted against the generated schema instead.

``tests/api/test_launch_monitor_longitudinal.py`` also does not travel: it is a
FastAPI route test for UpstreamDrift's ``src/api/routes/launch_monitor_analytics.py``,
an HTTP surface this repository does not host. Every model-layer obligation it
carries — the published contract version, the structured-unavailable posture on
untrusted identity, and the refusal of unknown request fields — is asserted
here directly against the model layer, which is the stronger place for it.

Decision G1-D1 is exercised as a pair, never as a winner
--------------------------------------------------------
``test_named_pooled_estimators_reproduce_both_g0_verdicts`` runs both
estimators over the exact ADR-0046 G0 fixture and pins each stack's own G0
numbers against the canonical implementation: ``ud-cluster-robust-fe/1``
reproduces UpstreamDrift's D10 pins and ``dl-random-effects/1`` reproduces
``rate_of_closure``'s, including the heterogeneity block D12 pinned and the
98.3% improvement probability. Both, on the same data, disagreeing about
significance — which is precisely why the plan preserved both.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from shared.python.launch_monitor import (
    POOLED_METHOD_DESCRIPTIONS,
    AnalysisContextV2,
    LongitudinalSessionRequestV1,
    OrderEvidenceV2,
    PlayerIdentityV2,
    PooledAssociationV1,
    SessionIdentityV2,
    analyze_longitudinal_sessions,
    longitudinal_session_contract_json_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURE_DIR / "longitudinal_attested_v1.json"
SOURCE_FIXTURE = FIXTURE.with_suffix(".csv")
CROSS_STACK_FIXTURE = FIXTURE_DIR / "adr0046_cross_stack_session_v1.json"

YARDS_PER_METRE = 1.0936132983377078
METRIC = "proximity_yards"

# G0 D10 pins for ``ud-cluster-robust-fe/1`` (UpstreamDrift's estimator).
G0_FE_ESTIMATE = -0.5255315268208663
G0_FE_STANDARD_ERROR = 0.3301766523964166
G0_FE_CI_LOW = -1.5763009943307855
G0_FE_CI_HIGH = 0.5252379406890527
G0_FE_P_VALUE = 0.20969656193018768

# G0 D10/D12 pins for ``dl-random-effects/1`` (the rate_of_closure estimator).
G0_DL_ESTIMATE = -0.5282789828979909
G0_DL_CI_LOW = -1.0145384362562389
G0_DL_CI_HIGH = -0.04201952953974292
G0_DL_TAU_SQUARED = 0.1594137105940229
G0_DL_Q_STATISTIC = 9.799861688653488
G0_DL_I_SQUARED_PCT = 69.38732305300319
G0_DL_IMPROVEMENT_PROBABILITY = 0.9833865960693259

# G0 D11 pins: per-player uncertainty UpstreamDrift could not express at all.
G0_PLAYER_SLOPES = {
    "P1": -0.6333284530839894,
    "P2": -0.6566383830927385,
    "P3": 0.2922907370953632,
    "P4": -1.1044500082020998,
}
G0_P1_STANDARD_ERROR = 0.427793725141183
G0_P1_P_VALUE = 0.23532861203653252
G0_P1_R_SQUARED = 0.4221591266852809
G0_P1_FIRST_TO_LAST = -1.0598996609798768

G0_SESSION_CELLS = 20


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _analyze(payload: dict[str, Any]):
    return analyze_longitudinal_sessions(
        pd.DataFrame.from_records(payload["records"]),
        LongitudinalSessionRequestV1.model_validate(payload["request"]),
        context=AnalysisContextV2.model_validate(payload["context"]),
    )


def _projection(result: Any) -> dict[str, Any]:
    return {
        "contract_version": result.contract_version,
        "status": result.status,
        "primary_unit": result.claims.primary_unit,
        "causal_improvement": result.claims.causal_improvement,
        "association_scope": result.claims.association_scope,
        "session_cells": len(result.session_aggregates),
        "player_count": len(result.player_associations),
        "player_directions": [item.direction for item in result.player_associations],
        "pooled_method": result.pooled_association.method,
        "pooled_clusters": result.pooled_association.cluster_count,
        "pooled_estimate": round(result.pooled_association.estimate_per_order_unit, 8),
        "order_unit": result.order_evidence.unit,
        "backing_records": len(result.lineage.backing_records),
    }


@pytest.fixture(scope="module")
def cross_stack_frame() -> pd.DataFrame:
    """Return the G0 160-shot session with the shared lower-is-better metric."""
    payload = json.loads(CROSS_STACK_FIXTURE.read_text(encoding="utf-8"))
    frame = pd.DataFrame.from_records(payload["records"])
    frame[METRIC] = frame["finish_distance_metres"] * YARDS_PER_METRE
    return frame


def _cross_stack_context() -> AnalysisContextV2:
    return AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="pseudonymous_stable",
            identifier_column="player_id",
            evidence="Synthetic ADR-0046 G0 fixture pseudonym.",
        ),
        session_identity=SessionIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="session_id",
            evidence="Synthetic ADR-0046 G0 fixture session identifier.",
        ),
        order_evidence=OrderEvidenceV2(
            trust_level="explicit_user_attested",
            order_column="session_order",
            order_kind="ordinal",
            unit="session",
            evidence="Synthetic ADR-0046 G0 fixture session ordinal.",
        ),
    )


def _cross_stack_request(pooled_method: str) -> LongitudinalSessionRequestV1:
    return LongitudinalSessionRequestV1(
        metric=METRIC,
        direction="lower_is_better",
        session_aggregate="mean",
        minimum_sessions_per_player=3,
        minimum_player_clusters=4,
        confidence_level=0.95,
        pooled_method=pooled_method,  # type: ignore[arg-type]
    )


# ── Ported UpstreamDrift cases ───────────────────────────────────────────


def test_attested_fixture_matches_golden_session_level_contract() -> None:
    payload = _fixture()

    result = _analyze(payload)

    assert _projection(result) == payload["expected"]
    assert result.pooled_association.uncertainty_state == "available"
    assert result.pooled_association.confidence_interval_low is not None
    assert result.pooled_association.confidence_interval_high is not None
    assert len(result.lineage.dataset_fingerprint_sha256) == 64
    assert all(item.shot_count == 3 for item in result.session_aggregates)


def test_golden_source_reference_is_content_addressed_and_exact() -> None:
    payload = _fixture()
    source = payload["context"]["sources"][0]

    source_rows = list(
        csv.DictReader(SOURCE_FIXTURE.read_text(encoding="utf-8").splitlines())
    )

    assert sha256(SOURCE_FIXTURE.read_bytes()).hexdigest() == source["file_sha256"]
    assert source["source_uri"] == (
        "tests/fixtures/launch_monitor/longitudinal_attested_v1.csv"
    )
    assert source_rows == [
        {key: str(value) for key, value in row.items()} for row in payload["records"]
    ]


def test_duplicate_shots_do_not_reweight_session_level_association() -> None:
    payload = _fixture()
    baseline = _analyze(payload)
    repeated = deepcopy(payload)
    duplicated = [
        row
        for row in repeated["records"]
        if row["player_id"] == "player-1" and row["session_id"] == "p1-s1"
    ]
    repeated["records"].extend(duplicated * 4)

    result = _analyze(repeated)

    assert len(result.session_aggregates) == len(baseline.session_aggregates)
    assert result.pooled_association.estimate_per_order_unit == pytest.approx(
        baseline.pooled_association.estimate_per_order_unit
    )
    assert len(result.lineage.backing_records) == len(repeated["records"])


@pytest.mark.parametrize(
    ("context_path", "reason_code"),
    [
        (("player_identity", "trust_level"), "untrusted_player_identity"),
        (("session_identity", "trust_level"), "untrusted_session_identity"),
        (("order_evidence", "trust_level"), "untrusted_order_evidence"),
    ],
)
def test_analysis_fails_closed_without_trusted_identity_and_order_evidence(
    context_path: tuple[str, str], reason_code: str
) -> None:
    payload = _fixture()
    payload["context"][context_path[0]][context_path[1]] = "untrusted_inferred"

    result = _analyze(payload)

    assert result.status == "unavailable"
    assert result.session_aggregates == ()
    assert result.pooled_association is None
    assert result.availability[0].reason_code == reason_code
    assert len(result.lineage.backing_records) == len(payload["records"])


def test_pooled_uncertainty_is_unavailable_with_too_few_player_clusters() -> None:
    payload = _fixture()
    payload["records"] = [
        row
        for row in payload["records"]
        if row["player_id"] in {"player-1", "player-2"}
    ]
    payload["context"]["sources"][0]["session_ids"] = sorted(
        {row["session_id"] for row in payload["records"]}
    )

    result = _analyze(payload)

    assert result.status == "partial"
    assert len(result.player_associations) == 2
    assert all(item.direction == "increasing" for item in result.player_associations)
    assert result.pooled_association is None
    pooled = next(
        item for item in result.availability if item.result_path == "pooled_association"
    )
    assert pooled.reason_code == "insufficient_player_clusters"
    assert pooled.observed_count == 2
    assert pooled.required_count == 4


def test_nonconstant_order_within_session_is_structured_unavailable() -> None:
    payload = _fixture()
    payload["records"][0]["session_number"] = 99

    result = _analyze(payload)

    assert result.status == "unavailable"
    assert result.session_aggregates == ()
    assert result.availability[0].reason_code == "nonconstant_session_order"


def test_analysis_is_unavailable_when_no_complete_finite_shots_remain() -> None:
    payload = _fixture()
    for row in payload["records"]:
        row["carry_distance"] = float("inf")

    result = _analyze(payload)

    assert result.status == "unavailable"
    assert result.session_aggregates == ()
    assert result.availability[0].reason_code == "no_complete_finite_shots"
    assert result.missingness.excluded_by_reason == {
        "incomplete_or_nonfinite_selected_fields": len(payload["records"])
    }
    assert len(result.lineage.backing_records) == len(payload["records"])


def test_blank_attested_identity_values_are_excluded_not_grouped() -> None:
    payload = _fixture()
    payload["records"][0]["player_id"] = "   "

    result = _analyze(payload)

    assert result.status == "available"
    assert result.missingness.included_shot_count == len(payload["records"]) - 1
    assert result.missingness.excluded_by_reason == {
        "incomplete_or_nonfinite_selected_fields": 1
    }


def test_declared_strata_and_confounders_are_explicit_design_terms() -> None:
    payload = _fixture()
    payload["request"]["strata"] = ["club"]
    payload["request"]["confounders"] = ["temperature_c"]

    result = _analyze(payload)

    assert result.design.strata == ("club",)
    assert result.design.confounders == ("temperature_c",)
    assert result.design.session_aggregate == "mean"
    assert result.claims.confounder_control_is_causal is False
    assert all(item.stratum == {"club": "7i"} for item in result.session_aggregates)


def test_contract_schema_is_versioned_and_forbids_extra_fields() -> None:
    schema = longitudinal_session_contract_json_schema()

    assert schema["properties"]["contract_version"]["const"] == (
        "launch-monitor-longitudinal-session/1.0.0"
    )
    assert schema["additionalProperties"] is False
    assert (
        schema["$defs"]["LongitudinalSessionRequestV1"]["additionalProperties"] is False
    )


def test_generated_schema_names_both_estimators_and_the_widened_fields() -> None:
    """Replaces UpstreamDrift's comparison against its published artifact."""
    definitions = longitudinal_session_contract_json_schema()["$defs"]
    pooled = definitions["PooledAssociationV1"]
    player = definitions["LongitudinalPlayerAssociationV1"]

    assert "method" in pooled["required"]
    assert pooled["properties"]["method"]["enum"] == [
        "ud-cluster-robust-fe/1",
        "dl-random-effects/1",
    ]
    assert {
        "tau_squared",
        "q_statistic",
        "i_squared_pct",
        "improvement_probability",
    } <= set(pooled["properties"])
    assert {
        "standard_error",
        "ci_lower",
        "ci_upper",
        "p_value",
        "r_squared",
        "first_to_last_change",
    } <= set(player["properties"])
    assert pooled["additionalProperties"] is False
    assert player["additionalProperties"] is False


def test_request_rejects_overlapping_or_duplicate_design_terms() -> None:
    with pytest.raises(ValueError, match="unique and disjoint"):
        LongitudinalSessionRequestV1(
            metric="carry_distance",
            strata=("club", "club"),
            confounders=("club",),
        )


def test_request_rejects_unknown_fields() -> None:
    """Carries the API test's schema-boundary obligation into the model layer."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LongitudinalSessionRequestV1.model_validate(
            {"metric": "carry_distance", "shot_level_inference": True}
        )


# ── P16: decision G1-D1, the named-method pair ───────────────────────────


def test_named_pooled_estimators_reproduce_both_g0_verdicts(
    cross_stack_frame: pd.DataFrame,
) -> None:
    """Both estimators, one dataset, opposite significance verdicts.

    The canonical implementations reproduce each stack's own G0 pins exactly,
    which is the evidence that the pair was *preserved* rather than
    reimplemented: ``ud-cluster-robust-fe/1`` lands on UpstreamDrift's D10
    numbers and ``dl-random-effects/1`` on ``rate_of_closure``'s D10/D12
    numbers, from the same twenty session cells.
    """
    context = _cross_stack_context()
    fixed_effects = analyze_longitudinal_sessions(
        cross_stack_frame,
        _cross_stack_request("ud-cluster-robust-fe/1"),
        context=context,
    )
    random_effects = analyze_longitudinal_sessions(
        cross_stack_frame,
        _cross_stack_request("dl-random-effects/1"),
        context=context,
    )

    assert fixed_effects.missingness.session_cell_count == G0_SESSION_CELLS
    assert random_effects.missingness.session_cell_count == G0_SESSION_CELLS

    fe = fixed_effects.pooled_association
    dl = random_effects.pooled_association
    assert fe is not None and dl is not None

    assert fe.method == "ud-cluster-robust-fe/1"
    assert fe.estimate_per_order_unit == pytest.approx(G0_FE_ESTIMATE, rel=1e-12)
    assert fe.standard_error == pytest.approx(G0_FE_STANDARD_ERROR, rel=1e-12)
    assert fe.confidence_interval_low == pytest.approx(G0_FE_CI_LOW, rel=1e-12)
    assert fe.confidence_interval_high == pytest.approx(G0_FE_CI_HIGH, rel=1e-12)
    assert fe.p_value == pytest.approx(G0_FE_P_VALUE, rel=1e-12)
    assert fe.tau_squared is None
    assert fe.improvement_probability is None

    assert dl.method == "dl-random-effects/1"
    assert dl.estimate_per_order_unit == pytest.approx(G0_DL_ESTIMATE, rel=1e-12)
    assert dl.confidence_interval_low == pytest.approx(G0_DL_CI_LOW, rel=1e-12)
    assert dl.confidence_interval_high == pytest.approx(G0_DL_CI_HIGH, rel=1e-12)
    assert dl.tau_squared == pytest.approx(G0_DL_TAU_SQUARED, rel=1e-12)
    assert dl.q_statistic == pytest.approx(G0_DL_Q_STATISTIC, rel=1e-12)
    assert dl.i_squared_pct == pytest.approx(G0_DL_I_SQUARED_PCT, rel=1e-12)
    assert dl.improvement_probability == pytest.approx(
        G0_DL_IMPROVEMENT_PROBABILITY, rel=1e-12
    )

    # The verdicts disagree — the whole reason G1-D1 keeps both.
    assert fe.confidence_interval_low < 0.0 < fe.confidence_interval_high
    assert dl.confidence_interval_high < 0.0


def test_per_player_uncertainty_closes_d11(cross_stack_frame: pd.DataFrame) -> None:
    """D11: UpstreamDrift 'cannot express per-player slope uncertainty at all'."""
    result = analyze_longitudinal_sessions(
        cross_stack_frame,
        _cross_stack_request("ud-cluster-robust-fe/1"),
        context=_cross_stack_context(),
    )
    players = {item.player_id: item for item in result.player_associations}

    assert set(players) == set(G0_PLAYER_SLOPES)
    for player, slope in G0_PLAYER_SLOPES.items():
        assert players[player].estimate_per_order_unit == pytest.approx(
            slope, rel=1e-12
        )
        assert players[player].standard_error is not None
        assert players[player].ci_lower is not None
        assert players[player].ci_upper is not None
        assert players[player].p_value is not None
        assert players[player].r_squared is not None
        assert players[player].first_to_last_change is not None

    assert players["P1"].standard_error == pytest.approx(
        G0_P1_STANDARD_ERROR, rel=1e-12
    )
    assert players["P1"].p_value == pytest.approx(G0_P1_P_VALUE, rel=1e-12)
    assert players["P1"].r_squared == pytest.approx(G0_P1_R_SQUARED, rel=1e-12)
    assert players["P1"].first_to_last_change == pytest.approx(
        G0_P1_FIRST_TO_LAST, rel=1e-12
    )


def test_default_pooled_method_preserves_upstreamdrift_behaviour() -> None:
    """G1-D1 renames the estimator; it does not swap the default out."""
    request = LongitudinalSessionRequestV1(metric="carry_distance")

    assert request.pooled_method == "ud-cluster-robust-fe/1"

    result = _analyze(_fixture())
    assert result.pooled_association is not None
    assert result.pooled_association.method == "ud-cluster-robust-fe/1"


def test_result_states_which_estimator_produced_it() -> None:
    """Results from different estimators are never comparable anonymously."""
    payload = _fixture()
    result = _analyze(payload)
    pooled_availability = next(
        item for item in result.availability if item.result_path == "pooled_association"
    )

    assert (
        pooled_availability.message
        == POOLED_METHOD_DESCRIPTIONS["ud-cluster-robust-fe/1"]
    )
    assert any("ud-cluster-robust-fe/1" in warning for warning in result.warnings)
    assert any("not numerically comparable" in warning for warning in result.warnings)


def test_random_effects_declines_rather_than_inventing_a_weight() -> None:
    """Zero within-player residual variance is not an inverse-variance weight."""
    payload = _fixture()
    payload["request"]["pooled_method"] = "dl-random-effects/1"

    result = _analyze(payload)
    pooled = next(
        item for item in result.availability if item.result_path == "pooled_association"
    )

    assert result.status == "partial"
    assert result.pooled_association is None
    assert pooled.reason_code == "insufficient_estimable_player_slopes"
    assert result.design.pooled_terms == ()


def test_random_effects_reports_no_improvement_probability_without_a_direction(
    cross_stack_frame: pd.DataFrame,
) -> None:
    """``descriptive_only`` declines to say which way is better, so P is absent."""
    request = _cross_stack_request("dl-random-effects/1").model_copy(
        update={"direction": "descriptive_only"}
    )

    result = analyze_longitudinal_sessions(
        cross_stack_frame, request, context=_cross_stack_context()
    )

    assert result.pooled_association is not None
    assert result.pooled_association.improvement_probability is None
    assert result.pooled_association.tau_squared is not None


def test_pooled_association_requires_its_method_identifier() -> None:
    with pytest.raises(ValidationError, match="method"):
        PooledAssociationV1(  # type: ignore[call-arg]
            estimate_per_order_unit=1.0,
            standard_error=0.1,
            confidence_interval_low=0.8,
            confidence_interval_high=1.2,
            confidence_level=0.95,
            cluster_count=4,
            session_cell_count=12,
        )


def test_pooled_association_refuses_cross_method_output_fields() -> None:
    """A cluster-robust fit may not carry a heterogeneity block, and vice versa."""
    with pytest.raises(ValidationError, match="heterogeneity"):
        PooledAssociationV1(
            method="ud-cluster-robust-fe/1",
            estimate_per_order_unit=1.0,
            standard_error=0.1,
            confidence_interval_low=0.8,
            confidence_interval_high=1.2,
            confidence_level=0.95,
            cluster_count=4,
            session_cell_count=12,
            tau_squared=0.5,
        )
    with pytest.raises(ValidationError, match="improvement"):
        PooledAssociationV1(
            method="ud-cluster-robust-fe/1",
            estimate_per_order_unit=1.0,
            standard_error=0.1,
            confidence_interval_low=0.8,
            confidence_interval_high=1.2,
            confidence_level=0.95,
            cluster_count=4,
            session_cell_count=12,
            improvement_probability=0.9,
        )
    with pytest.raises(ValidationError, match="tau_squared"):
        PooledAssociationV1(
            method="dl-random-effects/1",
            estimate_per_order_unit=1.0,
            standard_error=0.1,
            confidence_interval_low=0.8,
            confidence_interval_high=1.2,
            confidence_level=0.95,
            cluster_count=4,
            session_cell_count=12,
        )


def test_request_refuses_an_unnamed_pooled_estimator() -> None:
    with pytest.raises(ValidationError):
        LongitudinalSessionRequestV1(
            metric="carry_distance",
            pooled_method="inverse_variance",  # type: ignore[arg-type]
        )


def test_pooled_terms_name_the_contributing_players_for_random_effects(
    cross_stack_frame: pd.DataFrame,
) -> None:
    """The FE design lists matrix columns; DL lists the studies it synthesised."""
    context = _cross_stack_context()
    fixed_effects = analyze_longitudinal_sessions(
        cross_stack_frame,
        _cross_stack_request("ud-cluster-robust-fe/1"),
        context=context,
    )
    random_effects = analyze_longitudinal_sessions(
        cross_stack_frame,
        _cross_stack_request("dl-random-effects/1"),
        context=context,
    )

    assert fixed_effects.design.pooled_terms[:2] == ("intercept", "order_value")
    assert random_effects.design.pooled_terms == ("P1", "P2", "P3", "P4")


def test_analysis_never_mutates_the_caller_frame() -> None:
    payload = _fixture()
    frame = pd.DataFrame.from_records(payload["records"])
    before = frame.copy(deep=True)

    analyze_longitudinal_sessions(
        frame,
        LongitudinalSessionRequestV1.model_validate(payload["request"]),
        context=AnalysisContextV2.model_validate(payload["context"]),
    )

    pd.testing.assert_frame_equal(frame, before)


def test_missing_analysis_column_is_structured_not_raised() -> None:
    payload = _fixture()
    payload["request"]["metric"] = "absent_metric"

    result = _analyze(payload)

    assert result.status == "unavailable"
    assert result.availability[0].reason_code == "analysis_column_missing"
