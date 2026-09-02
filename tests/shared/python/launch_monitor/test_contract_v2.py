"""Canonical v2 launch-monitor analysis contract tests (ADR-0046 G1 step P11).

Ported from UpstreamDrift's ``tests/unit/launch_monitor/test_contract_v2.py``,
travelling with the module they exercise. Every case travels except
``test_published_schema_matches_the_python_authority``, which compares the
generated schema against UpstreamDrift's committed
``docs/api/contracts/launch-monitor-analysis-v2.schema.json`` — that artifact is
UpstreamDrift's published API surface rather than part of this model layer, and
committing a second copy here would create a second thing to drift. Its
structural obligations are asserted directly against
:func:`~shared.python.launch_monitor.contract_v2.contract_v2_json_schema` in
``test_generated_schema_is_the_python_authority`` instead, which pins the same
guarantees against the authority itself rather than against a file that has to
be regenerated to stay true.

The remaining added cases pin the module's refusals per this repo's
design-by-contract standard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from shared.python.launch_monitor.contract_v2 import (
    CONTRACT_VERSION_V2,
    AnalysisContextV2,
    DatasetAuthorityV2,
    ModelProvenanceV2,
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
    SourceFileReferenceV2,
    TransformRecordV2,
    adapt_v2_to_v1,
    analysis_lineage_v2,
    analyze_variables_v2,
    contract_v2_json_schema,
    metric_units_v2,
    vendor_provenance_v2,
)
from shared.python.launch_monitor.flexible_analysis import (
    CONTRACT_VERSION,
    FlexibleAnalysisRequest,
    analyze_variables,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _shots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(12)],
            "session_id": ["session-a"] * 6 + ["session-b"] * 6,
            "source_row": list(range(2, 8)) * 2,
            "monitor_vendor": ["TrackMan"] * 6 + ["Foresight"] * 6,
            "monitor_model": ["TrackMan 4"] * 6 + ["GCQuad"] * 6,
            "software_version": ["4.4"] * 6 + ["FSX 2020"] * 6,
            "player": ["player-01"] * 12,
            "tags": [["range", "validated"]] * 12,
            "club_speed": np.linspace(38.0, 49.0, 12),
            "ball_speed": np.linspace(57.0, 73.5, 12),
            "attack_angle": [
                np.nan,
                -0.04,
                0.01,
                -0.02,
                0.03,
                -0.01,
                0.04,
                -0.03,
                0.02,
                -0.015,
                0.035,
                0.0,
            ],
            "status::ball_speed": ["reported"] * 6 + ["measured"] * 6,
        }
    )


def _context() -> AnalysisContextV2:
    return AnalysisContextV2(
        authority=DatasetAuthorityV2(
            dataset_id="private-shot-corpus",
            repository="D-sorganization/Launch-Monitor-Flight-Model-Campaign",
            commit="9" * 40,
            dataset_path="data/authority/database/shot_corpus_parquet",
        ),
        player_identity=PlayerIdentityV2(
            trust_level="pseudonymous_stable",
            identifier_column="player",
            evidence="Explicit stable pseudonym supplied by the study owner.",
        ),
        transformations=(
            TransformRecordV2(
                transform_id="canonical-unit-normalization",
                version="1.0.0",
                parameters_sha256="a" * 64,
            ),
        ),
        sources=(
            SourceFileReferenceV2(
                source_id="trackman-study",
                file_sha256="b" * 64,
                session_ids=("session-a",),
                rights_status="restricted_internal",
            ),
        ),
        source_units={"source::custom_metric": "source-unit"},
    )


# ---------------------------------------------------------------------------
# Ported from UpstreamDrift's test_contract_v2.py
# ---------------------------------------------------------------------------


def test_v2_envelope_covers_units_lineage_missingness_and_provenance() -> None:
    result = analyze_variables_v2(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="comprehensive",
            min_samples=5,
        ),
        context=_context(),
    )

    payload = result.model_dump(mode="json", exclude_none=True)
    assert payload["contract_version"] == "2.0.0"
    assert payload["status"] == "available"
    assert payload["analysis"]["contract_version"] == "1.0.0"
    assert payload["units"]["ball_speed"] == {
        "canonical_unit": "m/s",
        "display_unit": "mph",
        "authority": "canonical_registry",
    }
    assert payload["missingness"]["missing_by_variable"]["attack_angle"] == 1
    assert payload["missingness"]["excluded_by_reason"]["regression_incomplete"] == 1
    assert payload["lineage"]["authority"]["commit"] == "9" * 40
    assert len(payload["lineage"]["backing_records"]) == 12
    assert all(
        len(reference["record_sha256"]) == 64
        for reference in payload["lineage"]["backing_records"]
    )
    assert payload["lineage"]["transformations"][0]["transform_id"] == (
        "canonical-unit-normalization"
    )
    assert payload["lineage"]["sources"][0]["file_sha256"] == "b" * 64
    assert payload["lineage"]["backing_records"][0]["source_id"] == ("trackman-study")
    assert payload["lineage"]["backing_records"][6]["unlinked_reason"] == (
        "session_not_linked_to_source_reference"
    )
    assert payload["player_identity"]["trust_level"] == "pseudonymous_stable"
    assert {item["vendor"] for item in payload["vendor_provenance"]} == {
        "TrackMan",
        "Foresight",
    }
    assert payload["uncertainty"]["confidence_level"] == pytest.approx(0.95)
    assert payload["uncertainty"]["multiplicity_adjustment"] == ("benjamini-hochberg")
    assert payload["claims"]["vendor_comparison"] == "descriptive"
    assert payload["claims"]["device_emulation"] is False


def test_v2_makes_per_estimate_unavailability_explicit() -> None:
    frame = _shots()
    frame.loc[:8, "attack_angle"] = np.nan
    result = analyze_variables_v2(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="correlation",
            min_samples=5,
        ),
    )

    assert result.status == "partial"
    unavailable = {
        item.result_path: item
        for item in result.availability
        if item.state == "unavailable"
    }
    item = unavailable["correlations.attack_angle"]
    assert item.reason_code == "insufficient_samples"
    assert item.observed_count == 3
    assert item.required_count == 5


def test_v2_returns_unavailable_result_for_insufficient_regression() -> None:
    result = analyze_variables_v2(
        _shots().iloc[:5],
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="regression",
            min_samples=10,
        ),
    )

    assert result.status == "unavailable"
    assert result.analysis is None
    assert len(result.availability) == 1
    assert result.availability[0].result_path == "regression"
    assert result.availability[0].reason_code == "insufficient_complete_rows"
    assert result.availability[0].observed_count == 4
    assert result.availability[0].required_count == 10


def test_v2_player_grouping_requires_explicit_trusted_identity() -> None:
    with pytest.raises(ValueError, match="explicit trusted player identity"):
        analyze_variables_v2(
            _shots(),
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed",),
                group_by="player",
                analysis_mode="correlation",
                min_samples=5,
            ),
        )


@pytest.mark.parametrize(
    "identifier_column",
    [
        "session",
        "session_id",
        "Session ID",
        "club",
        "club_id",
        "source",
        "source_id",
        "file",
        "filename",
        "file_name",
        "row_order",
        "source_row",
        "source-row",
    ],
)
def test_v2_rejects_forbidden_player_pseudo_identity_even_when_attested(
    identifier_column: str,
) -> None:
    with pytest.raises(ValidationError, match="cannot be used as player identity"):
        PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column=identifier_column,
            evidence="The user attested this source field.",
        )


def test_v2_separates_session_identity_and_order_evidence() -> None:
    context = AnalysisContextV2(
        session_identity=SessionIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="session_id",
            evidence="The data owner attested the session boundaries.",
        ),
        order_evidence=OrderEvidenceV2(
            trust_level="source_reported",
            order_column="captured_at",
            order_kind="timestamp",
            unit="iso8601-utc",
            evidence="Exported capture timestamp from the source device.",
        ),
    )

    assert context.session_identity.identifier_column == "session_id"
    assert context.order_evidence.order_kind == "timestamp"
    assert context.player_identity.trust_level == "not_provided"


def test_v2_session_identity_requires_complete_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        SessionIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="session_id",
        )


def test_v2_order_contract_requires_complete_evidence() -> None:
    with pytest.raises(ValidationError, match="unit"):
        OrderEvidenceV2(
            trust_level="source_reported",
            order_column="captured_at",
            order_kind="timestamp",
            evidence="Device timestamp.",
        )


def test_v2_accepts_only_full_commit_shas() -> None:
    with pytest.raises(ValidationError, match="commit"):
        DatasetAuthorityV2(dataset_id="corpus", commit="97f3ecf")
    with pytest.raises(ValidationError, match="code_commit"):
        ModelProvenanceV2(model_id="penner", version="1", code_commit="deadbee")
    with pytest.raises(ValidationError, match="trust_level"):
        PlayerIdentityV2(trust_level="explicit_session_label")  # type: ignore[arg-type]
    attested = PlayerIdentityV2(
        trust_level="explicit_user_attested",
        identifier_column="player",
        evidence="The user explicitly assigned this player label.",
    )
    assert attested.trust_level == "explicit_user_attested"


def test_v2_rejects_undeclared_backing_source_reference() -> None:
    frame = _shots()
    frame["source_id"] = "not-declared"
    with pytest.raises(ValueError, match="not declared in context.sources"):
        analyze_variables_v2(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed",),
                analysis_mode="correlation",
                min_samples=5,
            ),
            context=_context(),
        )


def test_v2_missing_selected_column_is_a_contract_error() -> None:
    with pytest.raises(ValueError, match="Columns not present.*missing_metric"):
        analyze_variables_v2(
            _shots(),
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("missing_metric",),
                analysis_mode="correlation",
                min_samples=5,
            ),
        )


def test_v2_unknown_units_are_explicitly_source_declared_or_unknown() -> None:
    frame = _shots().iloc[:6].copy()
    frame["source::custom_metric"] = np.linspace(1.0, 2.0, 6)
    declared = analyze_variables_v2(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("source::custom_metric",),
            analysis_mode="correlation",
            min_samples=5,
        ),
        context=_context(),
    )
    assert declared.units["source::custom_metric"].authority == "source_declared"
    assert declared.units["source::custom_metric"].canonical_unit == "source-unit"

    unknown = analyze_variables_v2(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("source::custom_metric",),
            analysis_mode="correlation",
            min_samples=5,
        ),
    )
    assert unknown.units["source::custom_metric"].authority == "unknown"
    assert unknown.units["source::custom_metric"].canonical_unit == "unknown"


def test_v1_adapter_remains_unchanged() -> None:
    result = analyze_variables(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            min_samples=5,
        ),
    )
    assert CONTRACT_VERSION == "1.0.0"
    assert result.to_dict()["contract_version"] == "1.0.0"

    v2 = analyze_variables_v2(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            min_samples=5,
        ),
    )
    adapted = adapt_v2_to_v1(v2)
    assert adapted["contract_version"] == "1.0.0"
    assert adapted["dataset"]["fingerprint_sha256"] == (
        v2.lineage.dataset_fingerprint_sha256
    )


# ---------------------------------------------------------------------------
# The published-schema pin, asserted against the Python authority
# ---------------------------------------------------------------------------


def test_generated_schema_is_the_python_authority() -> None:
    """Replaces UpstreamDrift's committed-artifact comparison.

    UpstreamDrift pins ``contract_v2_json_schema()`` against
    ``docs/api/contracts/launch-monitor-analysis-v2.schema.json``. That file is
    UpstreamDrift's published API surface, not part of this model layer; the
    obligations it enforced are asserted here directly, which cannot go stale
    behind an un-regenerated artifact.
    """
    schema = contract_v2_json_schema()

    assert CONTRACT_VERSION_V2 == "2.0.0"
    assert schema["title"] == "LaunchMonitorAnalysisResultV2"
    assert schema["properties"]["contract_version"]["const"] == "2.0.0"

    # extra="forbid" must reach the wire, or a v2 consumer cannot tell a typo
    # from a field it does not know yet.
    assert schema["additionalProperties"] is False

    # The full model set, so a new record type cannot be added silently and an
    # existing one cannot be dropped from the published surface.
    assert set(schema["$defs"]) == {
        "AnalysisLineageV2",
        "AvailabilityV2",
        "BackingRecordV2",
        "ClaimsV2",
        "DatasetAuthorityV2",
        "MetricUnitsV2",
        "MissingnessV2",
        "ModelProvenanceV2",
        "OrderEvidenceV2",
        "PlayerIdentityV2",
        "SessionIdentityV2",
        "SourceFileReferenceV2",
        "TransformRecordV2",
        "UncertaintyV2",
        "VendorProvenanceV2",
    }
    assert set(schema["required"]) == {
        "analysis",
        "availability",
        "lineage",
        "missingness",
        "player_identity",
        "status",
        "uncertainty",
        "units",
        "vendor_provenance",
    }

    # UpstreamDrift's two identity assertions.
    properties = schema["properties"]
    assert properties["session_identity"]["$ref"].endswith("/SessionIdentityV2")
    assert properties["order_evidence"]["$ref"].endswith("/OrderEvidenceV2")

    # The forbidden-player-identifier guard has to survive into the schema, or a
    # static client generated from it will happily offer ``session_id`` as an
    # identity even though the Python model refuses it.
    player_identifier = schema["$defs"]["PlayerIdentityV2"]["properties"][
        "identifier_column"
    ]
    assert "session_id" in player_identifier["not"]["enum"]
    assert "source_row" in player_identifier["not"]["enum"]
    assert set(player_identifier["not"]["enum"]) == {
        "club",
        "club_id",
        "file",
        "file_name",
        "filename",
        "row_order",
        "session",
        "session_id",
        "source",
        "source_id",
        "source_row",
    }


# ---------------------------------------------------------------------------
# Design-by-contract refusals (CLAUDE.md: every public function validates input)
# ---------------------------------------------------------------------------


def test_context_refuses_duplicate_source_ids() -> None:
    with pytest.raises(ValidationError, match="source_id values must be unique"):
        AnalysisContextV2(
            sources=(
                SourceFileReferenceV2(source_id="one", file_sha256="a" * 64),
                SourceFileReferenceV2(source_id="one", file_sha256="b" * 64),
            )
        )


def test_context_refuses_a_session_claimed_by_two_sources() -> None:
    with pytest.raises(ValidationError, match="cannot link to multiple source"):
        AnalysisContextV2(
            sources=(
                SourceFileReferenceV2(
                    source_id="one", file_sha256="a" * 64, session_ids=("s",)
                ),
                SourceFileReferenceV2(
                    source_id="two", file_sha256="b" * 64, session_ids=("s",)
                ),
            )
        )


def test_context_refuses_blank_source_unit_declarations() -> None:
    with pytest.raises(ValidationError, match="source_units keys and values"):
        AnalysisContextV2(source_units={"custom": "   "})


def test_order_evidence_fields_require_a_declared_trust_level() -> None:
    """The default trust level cannot carry evidence fields alongside it."""
    with pytest.raises(ValidationError, match="require a non-default trust_level"):
        OrderEvidenceV2(order_column="captured_at")


def test_session_identity_fields_require_a_declared_trust_level() -> None:
    with pytest.raises(ValidationError, match="require a non-default trust_level"):
        SessionIdentityV2(identifier_column="session_id")


def test_every_v2_record_is_frozen_and_forbids_extra_fields() -> None:
    """``_ContractModel`` is the guarantee every serialized record inherits."""
    with pytest.raises(ValidationError):
        DatasetAuthorityV2(dataset_id="corpus", not_a_field="x")  # type: ignore[call-arg]

    authority = DatasetAuthorityV2(dataset_id="corpus")
    with pytest.raises(ValidationError):
        authority.dataset_id = "mutated"  # type: ignore[misc]


def test_backing_record_requires_exactly_one_link_or_reason() -> None:
    """Every row either joins a declared source or says why it does not."""
    frame = _shots()
    lineage = analysis_lineage_v2(frame, _context(), ("ball_speed", "club_speed"))
    for record in lineage.backing_records:
        assert (record.source_id is None) != (record.unlinked_reason is None)

    linked = [r for r in lineage.backing_records if r.source_id == "trackman-study"]
    unlinked = [
        r
        for r in lineage.backing_records
        if r.unlinked_reason == "session_not_linked_to_source_reference"
    ]
    assert len(linked) == 6
    assert len(unlinked) == 6


def test_lineage_refuses_empty_or_absent_selected_columns() -> None:
    frame = _shots()
    with pytest.raises(ValueError, match="must contain non-empty names"):
        analysis_lineage_v2(frame, _context(), ())
    with pytest.raises(ValueError, match="must contain non-empty names"):
        analysis_lineage_v2(frame, _context(), ("ball_speed", ""))
    with pytest.raises(ValueError, match="Columns not present.*no_such_column"):
        analysis_lineage_v2(frame, _context(), ("ball_speed", "no_such_column"))


def test_lineage_fingerprint_binds_the_selected_columns_not_only_the_rows() -> None:
    """Two selections over identical rows must not share a fingerprint."""
    frame = _shots()
    context = _context()
    one = analysis_lineage_v2(frame, context, ("ball_speed", "club_speed"))
    two = analysis_lineage_v2(frame, context, ("ball_speed", "attack_angle"))
    again = analysis_lineage_v2(frame, context, ("ball_speed", "club_speed"))

    assert one.dataset_fingerprint_sha256 == again.dataset_fingerprint_sha256
    assert one.dataset_fingerprint_sha256 != two.dataset_fingerprint_sha256
    assert [r.record_sha256 for r in one.backing_records] == [
        r.record_sha256 for r in two.backing_records
    ]


def test_group_by_must_match_the_declared_player_identifier_column() -> None:
    frame = _shots()
    frame["player_alias"] = "player-01"
    context = AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="player_alias",
            evidence="The study owner attested this alias column.",
        )
    )
    with pytest.raises(ValueError, match="must match the declared player"):
        analyze_variables_v2(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed",),
                group_by="player",
                analysis_mode="correlation",
                min_samples=5,
            ),
            context=context,
        )


def test_v1_adapter_refuses_an_unavailable_result() -> None:
    result = analyze_variables_v2(
        _shots().iloc[:5],
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="regression",
            min_samples=10,
        ),
    )
    assert result.analysis is None
    with pytest.raises(ValueError, match="No v1 analysis is available"):
        adapt_v2_to_v1(result)


def test_metric_units_helper_matches_the_envelope_it_fills() -> None:
    context = _context()
    known = metric_units_v2("ball_speed", context)
    assert known.authority == "canonical_registry"
    assert known.canonical_unit == "m/s"
    assert known.display_unit == "mph"

    declared = metric_units_v2("source::custom_metric", context)
    assert declared.authority == "source_declared"

    unknown = metric_units_v2("source::custom_metric", AnalysisContextV2())
    assert unknown.authority == "unknown"
    assert unknown.canonical_unit == unknown.display_unit == "unknown"


def test_vendor_provenance_is_empty_without_a_vendor_column() -> None:
    frame = _shots().drop(columns=["monitor_vendor"])
    assert vendor_provenance_v2(frame, ("ball_speed",)) == ()


def test_vendor_provenance_partitions_metric_statuses_per_vendor() -> None:
    items = {
        item.vendor: item
        for item in vendor_provenance_v2(_shots(), ("ball_speed", "club_speed"))
    }
    assert set(items) == {"Foresight", "TrackMan"}
    assert items["TrackMan"].row_count == 6
    assert items["TrackMan"].models == ("TrackMan 4",)
    assert items["TrackMan"].software_versions == ("4.4",)
    assert items["TrackMan"].metric_statuses["ball_speed"] == ("reported",)
    assert items["Foresight"].metric_statuses["ball_speed"] == ("measured",)
    # A metric with no ``status::`` column contributes no status entry at all.
    assert "club_speed" not in items["TrackMan"].metric_statuses


def test_rank_deficient_regression_degrades_instead_of_raising() -> None:
    """A comprehensive request keeps its correlations when the OLS cannot fit."""
    frame = _shots()
    frame["club_speed_copy"] = frame["club_speed"]
    result = analyze_variables_v2(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "club_speed_copy"),
            analysis_mode="comprehensive",
            min_samples=5,
        ),
    )

    assert result.status == "partial"
    assert result.analysis is not None
    assert result.analysis["regression"] is None
    regression = next(
        item for item in result.availability if item.result_path == "regression"
    )
    assert regression.reason_code == "rank_deficient_design"
    assert regression.required_count == 3
    assert any(item.state == "available" for item in result.availability)


def test_correlation_only_requests_never_swallow_a_regression_error() -> None:
    """The degradation path is scoped: a correlation request re-raises."""
    frame = _shots()
    frame["constant"] = 1.0
    with pytest.raises(ValueError, match="Constant variables"):
        analyze_variables_v2(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("constant",),
                analysis_mode="correlation",
                min_samples=5,
            ),
        )


def test_uncertainty_names_the_methods_actually_requested() -> None:
    correlation_only = analyze_variables_v2(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            correlation_method="spearman",
            min_samples=5,
        ),
    )
    assert correlation_only.uncertainty.correlation_interval == "unavailable"
    assert correlation_only.uncertainty.regression_interval == "not_requested"
    assert correlation_only.uncertainty.multiplicity_adjustment == (
        "benjamini-hochberg"
    )

    regression_only = analyze_variables_v2(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="regression",
            min_samples=5,
        ),
    )
    assert regression_only.uncertainty.correlation_interval == "not_requested"
    assert regression_only.uncertainty.regression_interval == "student-t"
    assert regression_only.uncertainty.multiplicity_adjustment == "not_requested"
    assert "Correlation does not establish causality." in (
        regression_only.uncertainty.assumptions
    )


def test_claims_default_to_the_weakest_possible_assertion() -> None:
    """Nothing claims emulation, certification, or causality by default."""
    result = analyze_variables_v2(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            min_samples=5,
        ),
    )
    assert result.claims.vendor_comparison == "descriptive"
    assert result.claims.device_emulation is False
    assert result.claims.device_certification is False
    assert result.claims.causal_inference is False
    assert result.model_provenance == ()


def test_the_input_frame_is_never_mutated() -> None:
    frame = _shots()
    before = frame.copy(deep=True)
    analyze_variables_v2(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="comprehensive",
            min_samples=5,
        ),
        context=_context(),
    )
    pd.testing.assert_frame_equal(frame, before)
