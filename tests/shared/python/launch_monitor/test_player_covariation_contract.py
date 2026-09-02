"""Canonical player-covariation and population-synthesis contracts.

Travels with step **P18** of the ADR-0046 G1 port plan, from UpstreamDrift
``tests/unit/launch_monitor/test_player_covariation_contract.py``, together
with the portable half of UpstreamDrift's
``tests/api/test_routes_launch_monitor_covariation.py``.

Two of UpstreamDrift's obligations change shape on the way.

``test_contract_schema_and_consumer_golden_are_fresh`` compares the generated
schema to a committed ``docs/api/contracts/fixtures/`` artifact. That file is
UpstreamDrift's published HTTP surface, not part of this model layer; a second
committed copy here would be a second thing to drift. Its obligation travels
as assertions against the generated schema directly, which cannot go stale
behind an un-regenerated file.

``tests/api/test_routes_launch_monitor_covariation.py`` exercises a FastAPI
router this repository does not have. Its *logic* travels — the fail-closed
identity gate, the refusal of a session identifier as a player identity, the
structural missing-column error, the ranked/unavailable scan counts and the
published contract version — as direct calls against the same functions the
route wraps. The HTTP status codes do not travel; there is no route to give
them.

P18 is a **union port**, so this file also pins what the union folded in from
this repository's ``rate_of_closure`` trio, and what it refused to fold in.
See :mod:`shared.python.launch_monitor.player_covariation_types` for the
decisions themselves.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from shared.python.launch_monitor import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_V2,
    MIN_FISHER_SAMPLES,
    PLAYER_COVARIATION_CONTRACT_VERSION,
    SELECTED_PAIR_METHOD_DESCRIPTION,
    AnalysisContextV2,
    PlayerCovariationRequestV1,
    PlayerCovariationScanRequestV1,
    PlayerIdentityV2,
    SourceFileReferenceV2,
    analyze_player_covariation_v1,
    covariation_backing_frame,
    player_association_frame,
    player_covariation_contract_json_schema,
    scan_player_covariation_v1,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

PACKAGE_DIR = (
    Path(__file__).resolve().parents[4] / "src" / "shared" / "python" / "launch_monitor"
)
P18_MODULES = (
    "player_covariation.py",
    "player_covariation_core.py",
    "player_covariation_types.py",
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CROSS_STACK_FIXTURE = FIXTURE_DIR / "adr0046_cross_stack_session_v1.json"

# G0.1 pins (UpstreamDrift test_player_covariation_drift.py), reproduced here
# against the canonical layer rather than assumed. G0.1 measured 51 of 52
# shared scalars identical across the two stacks inside this quantum, which is
# why P18 is a union port and not a named-method pair.
REPORTING_QUANTUM = 1e-12
G0_X_COLUMN = "start_distance_yards"
G0_Y_COLUMN = "carry_distance_metres"
G0_PLAYER_COLUMN = "player_id"
G0_SCAN_COLUMNS = (
    "start_distance_yards",
    "carry_distance_metres",
    "lateral_carry_metres",
    "session_order",
)
G0_SAMPLE_COUNT = 160
G0_PLAYER_COUNT = 4
G0_PER_PLAYER_SAMPLES = 40
G0_PER_PLAYER_PEARSON = {
    "P1": -0.000937129848,
    "P2": -0.073982232796,
    "P3": -0.158714297045,
    "P4": -0.138858573218,
}
G0_POOLED_PEARSON = 0.060093306233
G0_WITHIN_PEARSON = -0.089711605547
G0_BETWEEN_PEARSON = 0.820163413566
G0_META_EFFECT_R = -0.093447506928
G0_META_CI = (-0.24945259306, 0.067285280773)
G0_Q_STATISTIC = 0.574044790862
G0_SCAN_PAIR_COUNT = 6
G0_DIRECTION_CONSISTENCY = (1.0, 0.75, 0.75, 0.5, 0.5, 0.25)

# Union pin: the between-player Fisher interval this repository's
# ``rate_of_closure`` trio reports, which G0.1 pinned as the D22 divergence at
# raw precision ``(-0.6655142653044201, 0.9960866924324187)``. The union
# carries the behaviour through UpstreamDrift's 12-decimal reporting quantum,
# so the canonical value is that interval rounded.
UNION_BETWEEN_CI = (-0.665514265304, 0.996086692432)


def _context() -> AnalysisContextV2:
    return AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="player_id",
            evidence="The fixture owner attests these stable player labels.",
        ),
        sources=(
            SourceFileReferenceV2(
                source_id="synthetic-source",
                file_sha256="1" * 64,
                rights_status="public_redistributable",
            ),
        ),
    )


def _confounded_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(10)],
            "source_id": ["synthetic-source"] * 10,
            "source_row": list(range(10)),
            "player_id": ["A"] * 5 + ["B"] * 5,
            "face_angle": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
            "club_path": [4, 3, 2, 1, 0, 14, 13, 12, 11, 10],
            "ball_speed": [100, 102, 104, 106, 108, 120, 122, 124, 126, 128],
            "monitor_vendor": ["TrackMan"] * 10,
            "monitor_model": ["fixture-comparable"] * 10,
            "software_version": ["fixture-1"] * 10,
        }
    )


def _cross_stack_frame() -> pd.DataFrame:
    payload = json.loads(CROSS_STACK_FIXTURE.read_text(encoding="utf-8"))
    return pd.DataFrame.from_records(payload["records"])


def _cross_stack_context() -> AnalysisContextV2:
    return AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column=G0_PLAYER_COLUMN,
            evidence="ADR-0046 cross-stack fixture declares player_id per shot",
        )
    )


def _g0_request() -> PlayerCovariationRequestV1:
    return PlayerCovariationRequestV1(
        x_column=G0_X_COLUMN,
        y_column=G0_Y_COLUMN,
        player_column=G0_PLAYER_COLUMN,
    )


# ---------------------------------------------------------------------------
# Ported from UpstreamDrift's contract suite.
# ---------------------------------------------------------------------------


def test_separates_pooled_within_between_and_population_effects() -> None:
    result = analyze_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationRequestV1(
            x_column="face_angle",
            y_column="club_path",
            player_column="player_id",
        ),
        context=_context(),
    )

    assert result.contract_version == PLAYER_COVARIATION_CONTRACT_VERSION
    assert result.status == "available"
    assert result.pooled.pearson_r == pytest.approx(0.8518518519)
    assert result.within_player.pearson_r == pytest.approx(-1.0)
    assert result.within_player.slope == -1.0
    assert result.within_player.intercept == 0.0
    assert result.within_player.ci_lower is None
    assert result.between_player.pearson_r == pytest.approx(1.0)
    assert result.meta_analysis.state == "available"
    assert result.meta_analysis.contributor_count == 2
    assert result.meta_analysis.fixed_effect_r == pytest.approx(-1.0, abs=1e-5)
    assert [item.player_id for item in result.per_player] == ["A", "B"]
    assert sum(item.fixed_weight or 0 for item in result.per_player) == pytest.approx(1)
    assert any("aggregation reversal" in warning for warning in result.warnings)

    player_payload = result.per_player[0].model_dump(mode="json")
    player_payload["random_weight"] = None
    with pytest.raises(ValueError, match="weights must be supplied together"):
        type(result.per_player[0]).model_validate(player_payload)


def test_result_retains_units_vendor_provenance_and_source_joinable_rows() -> None:
    result = analyze_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationRequestV1(
            x_column="face_angle",
            y_column="club_path",
            player_column="player_id",
        ),
        context=_context(),
    )

    assert result.units["face_angle"].canonical_unit == "rad"
    assert result.units["face_angle"].display_unit == "deg"
    assert result.units["face_angle"].authority == "canonical_registry"
    assert result.vendor_provenance[0].vendor == "TrackMan"
    assert len(result.lineage.backing_records) == 10
    assert {row.source_id for row in result.lineage.backing_records} == {
        "synthetic-source"
    }
    assert all(len(row.record_sha256) == 64 for row in result.lineage.backing_records)
    assert result.claims.causal_inference is False
    assert result.claims.device_emulation is False


def test_population_synthesis_reports_material_heterogeneity() -> None:
    frame = pd.DataFrame(
        {
            "source_id": ["synthetic-source"] * 18,
            "player_id": ["positive"] * 6 + ["mixed"] * 6 + ["negative"] * 6,
            "x": list(range(6)) * 3,
            "y": list(range(6)) + [0, 1, 0, 1, 0, 1] + list(reversed(range(6))),
        }
    )
    result = analyze_player_covariation_v1(
        frame,
        PlayerCovariationRequestV1(
            x_column="x", y_column="y", player_column="player_id"
        ),
        context=_context().model_copy(
            update={"source_units": {"x": "deg", "y": "deg"}}
        ),
    )

    assert result.meta_analysis.state == "available"
    assert result.meta_analysis.q_statistic is not None
    assert result.meta_analysis.q_statistic > 1
    assert result.meta_analysis.tau_squared is not None
    assert result.meta_analysis.tau_squared > 0
    assert result.meta_analysis.i_squared_pct is not None
    assert result.meta_analysis.i_squared_pct > 50


def test_player_analysis_requires_matching_trusted_identity() -> None:
    request = PlayerCovariationRequestV1(
        x_column="face_angle",
        y_column="club_path",
        player_column="player_id",
    )

    with pytest.raises(ValueError, match="trusted player identity"):
        analyze_player_covariation_v1(_confounded_frame(), request)

    mismatched = AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="verified_external",
            identifier_column="athlete_id",
            evidence="Joined against the governed participant register.",
        )
    )
    with pytest.raises(ValueError, match="must match"):
        analyze_player_covariation_v1(_confounded_frame(), request, context=mismatched)


def test_missing_constant_and_small_groups_are_structurally_unavailable() -> None:
    frame = pd.DataFrame(
        {
            "source_id": ["synthetic-source"] * 12,
            "player_id": ["good"] * 4 + ["small"] * 3 + ["constant"] * 4 + [""],
            "x": [1, 2, 3, 4, 1, 2, np.nan, 5, 5, 5, 5, 9],
            "y": [2, 4, 6, 8, 2, 4, 6, 1, 2, 3, 4, np.inf],
        }
    )
    context = _context().model_copy(update={"source_units": {"x": "mph", "y": "deg"}})

    result = analyze_player_covariation_v1(
        frame,
        PlayerCovariationRequestV1(
            x_column="x", y_column="y", player_column="player_id"
        ),
        context=context,
    )

    states = {item.player_id: item.estimate.reason_code for item in result.per_player}
    assert states == {
        "constant": "constant_x",
        "good": None,
        "small": "insufficient_samples",
    }
    assert result.meta_analysis.state == "unavailable"
    assert result.meta_analysis.reason_code == "insufficient_eligible_players"
    assert result.missingness.missing_by_variable["x"] == 1
    assert result.missingness.non_finite_by_variable["y"] == 1
    assert result.missingness.excluded_by_reason["blank_player_identity"] == 1
    assert result.status == "partial"
    assert any(
        item.result_path == "meta_analysis" and item.state == "unavailable"
        for item in result.availability
    )
    unavailable = next(item for item in result.per_player if item.player_id == "small")
    unavailable_payload = unavailable.model_dump(mode="json")
    unavailable_payload.update({"fixed_weight": 0.5, "random_weight": 0.5})
    with pytest.raises(ValueError, match="unavailable player"):
        type(unavailable).model_validate(unavailable_payload)


def test_pair_scan_is_deterministic_and_carries_multiplicity_boundary() -> None:
    frame = _confounded_frame().assign(constant_metric=1.0)
    result = scan_player_covariation_v1(
        frame,
        PlayerCovariationScanRequestV1(
            player_column="player_id",
            numeric_columns=(
                "club_path",
                "ball_speed",
                "face_angle",
                "constant_metric",
            ),
        ),
        context=_context(),
    )

    assert result.pair_count == 6
    assert result.ranking[0].rank == 1
    assert result.ranking[0].x_column == "ball_speed"
    assert result.ranking[0].y_column == "club_path"
    assert result.ranking[0].random_effect_r == pytest.approx(-1.0, abs=1e-5)
    assert result.ranking[0].input_row_count == len(frame)
    assert result.ranking[0].pairwise_complete_row_count == len(frame)
    assert result.ranking[0].excluded_row_count == 0
    assert result.ranking[-1].state == "unavailable"
    assert len(result.lineage.backing_records) == len(frame)
    assert any("multiplicity" in warning.lower() for warning in result.warnings)
    assert any("exploratory" in warning.lower() for warning in result.warnings)


def test_pair_scan_contract_rejects_inconsistent_states_and_counts() -> None:
    result = scan_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationScanRequestV1(
            player_column="player_id",
            numeric_columns=("club_path", "ball_speed", "face_angle"),
        ),
        context=_context(),
    )
    payload = result.model_dump(mode="json")
    payload["available_pair_count"] = 0
    with pytest.raises(ValueError, match="pair counts"):
        type(result).model_validate(payload)

    rank_payload = result.ranking[0].model_dump(mode="json")
    rank_payload["reason_code"] = "insufficient_eligible_players"
    with pytest.raises(ValueError, match="available ranked pair"):
        type(result.ranking[0]).model_validate(rank_payload)


def test_default_scan_excludes_numeric_source_structure() -> None:
    result = scan_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationScanRequestV1(player_column="player_id"),
        context=_context(),
    )

    selected = {item.x_column for item in result.ranking} | {
        item.y_column for item in result.ranking
    }
    assert selected == {"ball_speed", "club_path", "face_angle"}
    assert "source_row" not in selected


def test_contract_schema_is_published_for_both_analysis_kinds() -> None:
    """UpstreamDrift's golden-file half is replaced by direct assertions."""
    schema = player_covariation_contract_json_schema()

    assert schema["title"] == "PlayerCovariationContractV1"
    assert "PlayerCovariationResultV1" in schema["$defs"]
    assert "PlayerCovariationScanResultV1" in schema["$defs"]
    assert schema["discriminator"]["propertyName"] == "analysis_kind"
    selected = schema["$defs"]["PlayerCovariationResultV1"]["properties"]
    assert selected["analysis_kind"]["const"] == "selected_pair"
    assert selected["contract_version"]["const"] == (
        "launch-monitor-player-covariation/1.0.0"
    )
    scan = schema["$defs"]["PlayerCovariationScanResultV1"]["properties"]
    assert scan["analysis_kind"]["const"] == "pair_scan"


def test_generic_analysis_contract_versions_remain_unchanged() -> None:
    assert CONTRACT_VERSION == "1.0.0"
    assert CONTRACT_VERSION_V2 == "2.0.0"


# ---------------------------------------------------------------------------
# Ported from UpstreamDrift's API-route suite, as direct calls.
# ---------------------------------------------------------------------------


def test_route_logic_selected_pair_result_is_source_traceable() -> None:
    result = analyze_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationRequestV1(
            x_column="face_angle",
            y_column="club_path",
            player_column="player_id",
        ),
        context=_context(),
    )
    payload = result.model_dump(mode="json")

    assert payload["analysis_kind"] == "selected_pair"
    assert payload["meta_analysis"]["contributor_count"] == 2
    assert len(payload["lineage"]["backing_records"]) == 10
    assert payload["lineage"]["backing_records"][0]["source_id"] == "synthetic-source"
    assert payload["claims"]["causal_inference"] is False
    assert payload["contract_version"] == PLAYER_COVARIATION_CONTRACT_VERSION


def test_route_logic_rejects_a_session_identifier_as_player_identity() -> None:
    with pytest.raises(ValidationError, match="cannot be used as player identity"):
        PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="session_id",
            evidence="Attested, but still not person identity.",
        )


def test_route_logic_reports_missing_selected_columns_structurally() -> None:
    with pytest.raises(
        ValueError, match=r"Columns not present in dataset: not_in_source"
    ):
        analyze_player_covariation_v1(
            _confounded_frame(),
            PlayerCovariationRequestV1(
                x_column="face_angle",
                y_column="not_in_source",
                player_column="player_id",
            ),
            context=_context(),
        )


def test_route_logic_scan_reports_ranked_and_unavailable_pairs() -> None:
    frame = _confounded_frame().assign(constant_metric=1.0)
    result = scan_player_covariation_v1(
        frame,
        PlayerCovariationScanRequestV1(
            player_column="player_id",
            numeric_columns=(
                "face_angle",
                "club_path",
                "ball_speed",
                "constant_metric",
            ),
        ),
        context=_context(),
    )

    assert result.pair_count == 6
    assert result.unavailable_pair_count == 3
    assert result.available_pair_count == 3
    assert result.status == "partial"
    assert result.ranking[0].rank == 1


def test_route_logic_capability_advertises_the_contract_version() -> None:
    assert PLAYER_COVARIATION_CONTRACT_VERSION == (
        "launch-monitor-player-covariation/1.0.0"
    )


# ---------------------------------------------------------------------------
# G0.1 evidence, reproduced against the canonical layer.
# ---------------------------------------------------------------------------


def test_g0_cross_stack_scalars_are_reproduced_by_the_canonical_layer() -> None:
    """The 51-identical-scalar surface G0.1 measured, pinned canonically."""
    result = analyze_player_covariation_v1(
        _cross_stack_frame(), _g0_request(), context=_cross_stack_context()
    )

    assert result.pooled.sample_count == G0_SAMPLE_COUNT
    assert result.pooled.group_count == G0_PLAYER_COUNT
    assert result.pooled.pearson_r == G0_POOLED_PEARSON
    assert result.within_player.pearson_r == G0_WITHIN_PEARSON
    assert result.within_player.intercept == 0.0
    assert result.between_player.pearson_r == G0_BETWEEN_PEARSON
    assert result.between_player.sample_count == G0_PLAYER_COUNT

    assert len(result.per_player) == G0_PLAYER_COUNT
    for item in result.per_player:
        assert item.estimate.state == "available"
        assert item.estimate.sample_count == G0_PER_PLAYER_SAMPLES
        assert item.estimate.pearson_r == G0_PER_PLAYER_PEARSON[item.player_id]
        assert item.fixed_weight == pytest.approx(0.25)
        assert item.random_weight == pytest.approx(0.25)

    meta = result.meta_analysis
    assert meta.contributor_count == G0_PLAYER_COUNT
    assert meta.total_sample_count == G0_SAMPLE_COUNT
    assert meta.fixed_effect_r == G0_META_EFFECT_R
    assert meta.random_effect_r == G0_META_EFFECT_R
    assert (meta.fixed_ci_lower, meta.fixed_ci_upper) == G0_META_CI
    assert (meta.random_ci_lower, meta.random_ci_upper) == G0_META_CI
    assert meta.tau_squared == 0.0
    assert meta.i_squared_pct == 0.0
    # D21: the one scalar that did not round-trip across the two stacks. The
    # canonical layer inherits UpstreamDrift's rounding and accumulation, so
    # it reproduces UpstreamDrift's value exactly.
    assert meta.q_statistic == G0_Q_STATISTIC
    assert any("aggregation reversal" in warning for warning in result.warnings)


def test_g0_cross_stack_scan_ranking_is_reproduced() -> None:
    result = scan_player_covariation_v1(
        _cross_stack_frame(),
        PlayerCovariationScanRequestV1(
            player_column=G0_PLAYER_COLUMN, numeric_columns=G0_SCAN_COLUMNS
        ),
        context=_cross_stack_context(),
    )

    assert result.pair_count == G0_SCAN_PAIR_COUNT
    assert result.status == "available"
    assert tuple(
        item.direction_consistency for item in result.ranking
    ) == pytest.approx(G0_DIRECTION_CONSISTENCY)
    assert result.ranking[3].x_column == "carry_distance_metres"
    assert result.ranking[3].y_column == "session_order"
    assert result.ranking[3].i_squared_pct == 74.480825075496


def test_g0_three_shot_player_is_excluded_not_pooled() -> None:
    frame = _cross_stack_frame()
    extra = pd.DataFrame(
        [{**frame.iloc[index].to_dict(), G0_PLAYER_COLUMN: "P5"} for index in range(3)]
    )
    result = analyze_player_covariation_v1(
        pd.concat([frame, extra], ignore_index=True),
        _g0_request(),
        context=_cross_stack_context(),
    )

    ghost = next(item for item in result.per_player if item.player_id == "P5")
    assert ghost.estimate.state == "unavailable"
    assert ghost.estimate.reason_code == "insufficient_samples"
    assert ghost.estimate.sample_count == 3
    assert result.meta_analysis.contributor_count == G0_PLAYER_COUNT


def test_g0_player_without_complete_rows_is_retained_and_booked() -> None:
    """D24: a player with zero usable rows stays visible and downgrades status."""
    frame = _cross_stack_frame()
    ghost_row = pd.DataFrame(
        [{**frame.iloc[0].to_dict(), G0_PLAYER_COLUMN: "P5", G0_X_COLUMN: np.nan}]
    )
    result = analyze_player_covariation_v1(
        pd.concat([frame, ghost_row], ignore_index=True),
        _g0_request(),
        context=_cross_stack_context(),
    )

    assert [item.player_id for item in result.per_player] == [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
    ]
    assert result.status == "partial"
    assert result.missingness.excluded_by_reason == {
        "blank_player_identity": 0,
        "pairwise_incomplete": 1,
    }
    assert result.missingness.excluded_player_count_by_reason == {
        "insufficient_samples": 1
    }


def test_g0_blank_identity_is_booked_against_its_own_cause() -> None:
    """D27: the exclusion is named ``blank_player_identity``, not non-finite."""
    frame = _cross_stack_frame()
    blank = pd.DataFrame([{**frame.iloc[0].to_dict(), G0_PLAYER_COLUMN: "  "}])
    result = analyze_player_covariation_v1(
        pd.concat([frame, blank], ignore_index=True),
        _g0_request(),
        context=_cross_stack_context(),
    )

    assert result.pooled.sample_count == G0_SAMPLE_COUNT
    assert result.missingness.input_row_count == G0_SAMPLE_COUNT + 1
    assert result.missingness.excluded_by_reason == {
        "blank_player_identity": 1,
        "pairwise_incomplete": 0,
    }
    assert result.missingness.non_finite_by_variable == {
        G0_X_COLUMN: 0,
        G0_Y_COLUMN: 0,
    }


# ---------------------------------------------------------------------------
# Union pins: what P18 folded in from ``rate_of_closure``, and what it did not.
# ---------------------------------------------------------------------------


def test_union_named_minimum_fisher_sample_floor_is_the_request_floor() -> None:
    """Folded in from ``rate_of_closure._player_covariation_types``."""
    assert MIN_FISHER_SAMPLES == 4
    assert (
        PlayerCovariationRequestV1(
            x_column="x", y_column="y", player_column="p"
        ).min_samples
        == MIN_FISHER_SAMPLES
    )
    with pytest.raises(ValidationError):
        PlayerCovariationRequestV1(
            x_column="x",
            y_column="y",
            player_column="p",
            min_samples=MIN_FISHER_SAMPLES - 1,
        )


def test_union_selected_pair_result_states_its_method() -> None:
    """Folded in from ``rate_of_closure``'s ``method_description`` (D26)."""
    result = analyze_player_covariation_v1(
        _confounded_frame(),
        PlayerCovariationRequestV1(
            x_column="face_angle", y_column="club_path", player_column="player_id"
        ),
        context=_context(),
    )

    assert result.method_description == SELECTED_PAIR_METHOD_DESCRIPTION
    assert "Spearman is descriptive" in result.method_description
    assert "does not imply causation" in result.method_description
    # It is required: a result cannot omit what it computed.
    payload = result.model_dump(mode="json")
    del payload["method_description"]
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_union_backing_frame_matches_the_rate_of_closure_export_shape() -> None:
    """Folded in from ``rate_of_closure``'s ``backing_data`` (D26), as a
    function rather than a wire field."""
    request = PlayerCovariationRequestV1(
        x_column="face_angle", y_column="club_path", player_column="player_id"
    )
    frame = covariation_backing_frame(_confounded_frame(), request)

    assert list(frame.columns) == [
        "source_index",
        "player_id",
        "x",
        "y",
        "centered_x",
        "centered_y",
    ]
    assert frame.shape == (10, 6)
    assert frame.groupby("player_id")["centered_x"].mean().abs().max() < 1e-12
    assert frame["source_index"].tolist() == list(range(10))

    # The result document itself stays row-free.
    result = analyze_player_covariation_v1(
        _confounded_frame(), request, context=_context()
    )
    assert "backing_data" not in type(result).model_fields
    assert len(result.lineage.backing_records) == 10


def test_union_player_association_frame_matches_the_export_columns() -> None:
    """Folded in from ``rate_of_closure``'s DataFrame ``per_player`` (D26)."""
    frame = pd.DataFrame(
        {
            "source_id": ["synthetic-source"] * 11,
            "player_id": ["good"] * 4 + ["small"] * 3 + ["constant"] * 4,
            "x": [1, 2, 3, 4, 1, 2, 3, 5, 5, 5, 5],
            "y": [2, 4, 6, 8, 2, 4, 6, 1, 2, 3, 4],
        }
    )
    result = analyze_player_covariation_v1(
        frame,
        PlayerCovariationRequestV1(
            x_column="x", y_column="y", player_column="player_id"
        ),
        context=_context(),
    )
    table = player_association_frame(result.per_player)

    assert list(table.columns) == [
        "player_id",
        "sample_count",
        "pearson_r",
        "spearman_r",
        "slope",
        "intercept",
        "r_squared",
        "ci_lower",
        "ci_upper",
        "status",
        "fixed_weight",
        "random_weight",
    ]
    assert dict(zip(table["player_id"], table["status"], strict=True)) == {
        "constant": "constant_x",
        "good": "ok",
        "small": "insufficient_samples",
    }
    assert table.query("status != 'ok'")["pearson_r"].isna().all()


def test_union_request_validation_covers_the_rate_of_closure_refusals() -> None:
    """``rate_of_closure``'s three validation cases, at model construction."""
    with pytest.raises(ValidationError, match="must differ"):
        PlayerCovariationRequestV1(x_column="x", y_column="x", player_column="p")
    with pytest.raises(ValidationError):
        PlayerCovariationRequestV1(
            x_column="x", y_column="y", player_column="p", confidence_level=1.0
        )
    with pytest.raises(ValueError, match="Columns not present in dataset: missing"):
        analyze_player_covariation_v1(
            _confounded_frame(),
            PlayerCovariationRequestV1(
                x_column="face_angle", y_column="club_path", player_column="missing"
            ),
            context=_context().model_copy(
                update={
                    "player_identity": PlayerIdentityV2(
                        trust_level="explicit_user_attested",
                        identifier_column="missing",
                        evidence="Attested but absent from the frame.",
                    )
                }
            ),
        )


def test_union_refuses_to_raise_when_no_row_survives() -> None:
    """G1-D3: exclude-and-audit, where ``rate_of_closure`` raises outright.

    ``rate_of_closure.analyze_player_covariation`` raises "analysis requires at
    least one pairwise-complete player shot" on this frame. The canonical
    posture is a result that says so.
    """
    frame = pd.DataFrame(
        {
            "source_id": ["synthetic-source"] * 4,
            "player_id": ["A"] * 4,
            "x": [np.nan] * 4,
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = analyze_player_covariation_v1(
        frame,
        PlayerCovariationRequestV1(
            x_column="x", y_column="y", player_column="player_id"
        ),
        context=_context(),
    )

    assert result.status == "unavailable"
    assert result.pooled.state == "unavailable"
    assert result.pooled.reason_code == "insufficient_samples"
    assert result.missingness.pairwise_complete_row_count == 0
    assert result.missingness.excluded_by_reason["pairwise_incomplete"] == 4


def test_union_between_player_interval_is_reported() -> None:
    """The ``rate_of_closure`` posture the union carried, pending ruling D22.

    G0.1 pinned this interval as the D22 divergence: with four player means it
    is a Fisher-z interval on ``n - 3 = 1`` degree of freedom.
    """
    result = analyze_player_covariation_v1(
        _cross_stack_frame(), _g0_request(), context=_cross_stack_context()
    )

    assert result.between_player.state == "available"
    assert result.between_player.sample_count == G0_PLAYER_COUNT
    assert result.between_player.ci_lower == UNION_BETWEEN_CI[0]
    assert result.between_player.ci_upper == UNION_BETWEEN_CI[1]


def test_union_does_not_carry_the_column_name_suffix_unit_heuristic() -> None:
    """D23: units come from the registry, never from how a column is spelled.

    ``rate_of_closure.player_covariation``'s ``UNIT_SUFFIXES`` table labels
    ``start_distance_yards`` ``"s"`` — seconds — because the name ends in an
    ``s``. Nothing here does that.
    """
    result = analyze_player_covariation_v1(
        _cross_stack_frame(), _g0_request(), context=_cross_stack_context()
    )

    assert set(result.units) == {G0_X_COLUMN, G0_Y_COLUMN}
    assert result.units[G0_X_COLUMN].canonical_unit == "unknown"
    assert result.units[G0_X_COLUMN].authority == "unknown"
    assert result.units[G0_Y_COLUMN].canonical_unit == "unknown"
    assert result.units[G0_Y_COLUMN].authority == "unknown"

    # The heuristic is absent as code, not merely unused: no module defines a
    # suffix table or a name-based unit guesser. Checked structurally so a
    # docstring that names the deleted construct cannot satisfy it.
    for module in P18_MODULES:
        tree = ast.parse(PACKAGE_DIR.joinpath(module).read_text(encoding="utf-8"))
        defined = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        } | {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert "UNIT_SUFFIXES" not in defined, module
        assert not any("infer_unit" in name for name in defined), module


def test_p18_modules_import_nothing_from_rate_of_closure() -> None:
    """The layer rule: ``src/shared`` never depends on ``rate_of_closure``."""
    for module in P18_MODULES:
        tree = ast.parse(PACKAGE_DIR.joinpath(module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("rate_of_closure"), module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("rate_of_closure"), module
