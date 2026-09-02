"""Canonical launch-monitor analytics model layer (ADR-0046 Stage 1).

ADR-0046 ("Launch-Monitor Analytics — Two Workbenches, One Model Layer",
accepted 2026-08-30, recorded in UpstreamDrift ``docs/adr/``) converges the
fleet's two independent full-depth launch-monitor stacks onto a single model
layer. Stage 1 grows that canonical layer **here**, in Tools' shared code —
the fleet's DRY leaf — and explicitly *not* inside ``rate_of_closure``.

Modules arrive by port, never by reimplementation: UpstreamDrift's
``src/shared/python/launch_monitor/`` implementations are the reference, they
travel with their tests, their authors' attribution is retained in each
module's docstring, and no functionality is deleted or limited on the way. The
per-module order, classification, and gates are the ADR-0046 G1 port plan,
UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``.

Filenames mirror UpstreamDrift's exactly, so the Stage 2 re-pointing of the
UpstreamDrift workbench is a mechanical import rewrite
(``src.shared.python.launch_monitor.X`` → ``shared.python.launch_monitor.X``)
rather than a symbol remap.

Landed so far
-------------
====  =====================  =================================================
Step  Module                 Notes
====  =====================  =================================================
P1    :mod:`.dispersion`     2-D target-relative dispersion. Shares a name with
                             ``rate_of_closure`` and nothing else — G0 pinned
                             the gap as divergences D6-D9.
P2    :mod:`.multivariate`   PCA and variance-inflation diagnostics. No
                             ``rate_of_closure`` counterpart exists.
P3    :mod:`.trends`         Per-calendar-day robust trend, EWMA, and ranked
                             change candidates. ``TrendResult`` is renamed
                             ``TemporalTrendResult`` here — see that module.
P4    :mod:`.comparison`     Matched (Bland-Altman) and descriptive
                             cross-monitor comparison. Confirmed
                             UpstreamDrift-only by the port plan.
P5    :mod:`.schema`         The layer's vocabulary: 33 unit-carrying metric
                             definitions, identity columns, and the import
                             mapping/session contracts.
P6    :mod:`.treatment`      Flag-then-optionally-exclude quality pipeline with
                             a full audit log. Confirmed UpstreamDrift-only by
                             the port plan.
P7    :mod:`.relationships`  FDR-corrected correlation matrix, partial
                             correlations, and a screened dependency network.
                             Ruling D17 (explicit boolean-projection labelling)
                             applied. D15 (FDR denominator) does not reach
                             this module — no separate ``min_samples`` tier
                             above its own three-pair floor for the ruling's
                             defect to exist in; see the module docstring.
P8    :mod:`.modeling`       Reproducible NumPy regressions plus an optional
                             shallow MLP, with an identity-leakage guard. No
                             ``rate_of_closure`` counterpart exists.
P9    :mod:`.profiles`       Header-fingerprint vendor detection and the alias
                             and unit-default tables it detects with.
P9    :mod:`.importer`       CSV/TSV/XLSX/JSON session import into canonical
                             units, with a provenance manifest recording how
                             each unit was established.
P10   :mod:`.flexible_analysis`
                             Arbitrary outcome/predictor correlation + OLS with
                             dataset lineage. Its ``rate_of_closure`` twin was
                             measured by UpstreamDrift#9372 (G0.1) before this
                             moved. Rulings D15 (FDR excludes under-sampled
                             predictors before correcting) and D17 (carries P7's
                             ``boolean_projected`` label through as
                             ``CorrelationEstimate.is_boolean_projected``) are
                             applied.
P11   :mod:`.contract_v2`    The v2 serialization boundary over P10: evidence,
                             row-level lineage, availability, and the JSON
                             Schema every static client is generated from.
P12   :mod:`.strokes_gained_types`
                             Request/result/uncertainty wire models for governed
                             scoring, **minus the expected-strokes baseline
                             half** - that half is already home in
                             ``rate_of_closure.launch_monitor_strokes_gained_baseline``
                             and is reached structurally instead.
P12   :mod:`._scoring_statistics`
                             The uncertainty, grouping and trend helpers behind
                             G0 divergences D2, D3 and D4. ADR-0046's module
                             list omits this file; the port plan's third
                             correction restores it.
P13   :mod:`.outcome_proxy`  Target-relative radial dispersion that is
                             explicitly *not* strokes gained. Its new
                             target-error gate landed with it.
P14   :mod:`.strokes_gained` Source-backed SG from hash-verified expected-
                             strokes lookups. Carries ruling **G1-D2**: the
                             canonical estimand is the session cell, and the
                             shot-level fit survives as
                             ``shot-level-sg-trend/1``.
P15   :mod:`.longitudinal_types`
                             Session-unit longitudinal wire types, widened by
                             **G1-D1** to the union of both pooled estimators'
                             outputs and the six per-player uncertainty fields
                             that close D11.
P15   :mod:`.longitudinal_statistics`
                             Per-player slopes plus **both** named pooled
                             estimators - ``ud-cluster-robust-fe/1`` and
                             ``dl-random-effects/1``.
P16   :mod:`.longitudinal`   Attested longitudinal analysis. Carries ruling
                             **G1-D1**: the request names the pooled estimator
                             and the result carries that name.
====  =====================  =================================================

**Name-collision containment.** Symbols in this package collide by name with
``rate_of_closure`` symbols that compute something else — ``analyze_dispersion``
and ``DispersionResult`` already do, and ``TrendResult`` would have, which is
why P3 renamed it. The separate package is what keeps the rest apart, and that
containment lasts exactly as long as nobody adds a convenience re-export
between the two packages. Do not add one.
"""

from .comparison import (
    MonitorComparisonResult,
    MonitorSummary,
    PairwiseMonitorComparison,
    compare_monitors,
)
from .contract_v2 import (
    CONTRACT_VERSION_V2,
    AnalysisContextV2,
    AnalysisLineageV2,
    AvailabilityV2,
    BackingRecordV2,
    ClaimsV2,
    DatasetAuthorityV2,
    LaunchMonitorAnalysisResultV2,
    MetricUnitsV2,
    MissingnessV2,
    ModelProvenanceV2,
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
    SourceFileReferenceV2,
    TransformRecordV2,
    UncertaintyV2,
    VendorProvenanceV2,
    adapt_v2_to_v1,
    analysis_lineage_v2,
    analyze_variables_v2,
    build_analysis_lineage_v2,
    contract_v2_json_schema,
    metric_units_v2,
    vendor_provenance_v2,
)
from .dispersion import DispersionResult, analyze_dispersion
from .flexible_analysis import (
    CONTRACT_VERSION,
    AnalysisMode,
    CoefficientEstimate,
    CorrelationEstimate,
    CorrelationMethod,
    DatasetSummary,
    FlexibleAnalysisRequest,
    FlexibleAnalysisResult,
    GroupAnalysis,
    MissingPolicy,
    RegressionEstimate,
    ResidualDiagnostics,
    analyze_variables,
)
from .importer import import_session
from .longitudinal import (
    analyze_longitudinal_sessions,
    longitudinal_session_contract_json_schema,
)
from .longitudinal_statistics import (
    clustered_pooled_association,
    dersimonian_laird_pooled_association,
    player_associations,
)
from .longitudinal_types import (
    LONGITUDINAL_SESSION_CONTRACT_VERSION,
    POOLED_METHOD_DESCRIPTIONS,
    LongitudinalClaimsV1,
    LongitudinalDesignV1,
    LongitudinalMissingnessV1,
    LongitudinalPlayerAssociationV1,
    LongitudinalSessionRequestV1,
    LongitudinalSessionResultV1,
    PooledAssociationV1,
    PooledMethod,
    SessionAggregateV1,
)
from .modeling import PredictiveModelResult, fit_predictive_model
from .multivariate import PCAResult, VIFResult, compute_pca, compute_vif
from .outcome_proxy import analyze_outcome_proxy
from .profiles import (
    COMMON_ALIASES,
    PROFILES,
    ImportProfile,
    ProfileDetection,
    detect_profile,
    normalize_header,
)
from .relationships import (
    CorrelationResult,
    DependencyEdge,
    compute_correlations,
)
from .schema import (
    IDENTITY_COLUMNS,
    METRICS,
    ColumnMapping,
    ImportedSession,
    ImportManifest,
    ImportOptions,
    MetricDefinition,
    numeric_metric_columns,
)
from .strokes_gained import (
    analyze_source_backed_strokes_gained,
    strokes_gained_contract_json_schema,
)
from .strokes_gained_types import (
    BASELINE_CONTRACT_VERSION,
    OUTCOME_PROXY_CONTRACT_VERSION,
    STROKES_GAINED_CONTRACT_VERSION,
    AvailabilityV1,
    BaselineProvenanceV1,
    ConfidenceIntervalV1,
    CourseStateColumnsV1,
    CourseStateValueV1,
    EstimateSummaryV1,
    ExcludedRowV1,
    ExclusionSummaryV1,
    ExpectedStrokesBaselineLike,
    ExpectedStrokesStateLike,
    GroupingDimensionV1,
    GroupSummaryV1,
    InterpolationV1,
    LongitudinalDimensionV1,
    LongitudinalMethod,
    LongitudinalSummaryV1,
    OutcomeProxyRequestV1,
    OutcomeProxyResultV1,
    OutcomeProxyRowV1,
    StrokesGainedAnalysisResultV1,
    StrokesGainedRequestV1,
    StrokesGainedRowV1,
    StrokesGainedUncertaintyV1,
)
from .treatment import (
    FilterRule,
    TreatmentConfig,
    TreatmentResult,
    apply_treatment,
)
from .trends import ChangeCandidate, TemporalTrendResult, analyze_trend

__all__ = [
    "AnalysisContextV2",
    "AnalysisLineageV2",
    "AnalysisMode",
    "AvailabilityV1",
    "AvailabilityV2",
    "BASELINE_CONTRACT_VERSION",
    "BackingRecordV2",
    "BaselineProvenanceV1",
    "COMMON_ALIASES",
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_V2",
    "ChangeCandidate",
    "ClaimsV2",
    "CoefficientEstimate",
    "ColumnMapping",
    "ConfidenceIntervalV1",
    "CorrelationEstimate",
    "CorrelationMethod",
    "CorrelationResult",
    "CourseStateColumnsV1",
    "CourseStateValueV1",
    "DatasetAuthorityV2",
    "DatasetSummary",
    "DependencyEdge",
    "DispersionResult",
    "EstimateSummaryV1",
    "ExcludedRowV1",
    "ExclusionSummaryV1",
    "ExpectedStrokesBaselineLike",
    "ExpectedStrokesStateLike",
    "FilterRule",
    "FlexibleAnalysisRequest",
    "FlexibleAnalysisResult",
    "GroupAnalysis",
    "GroupSummaryV1",
    "GroupingDimensionV1",
    "IDENTITY_COLUMNS",
    "ImportManifest",
    "ImportOptions",
    "ImportProfile",
    "ImportedSession",
    "InterpolationV1",
    "LONGITUDINAL_SESSION_CONTRACT_VERSION",
    "LaunchMonitorAnalysisResultV2",
    "LongitudinalClaimsV1",
    "LongitudinalDesignV1",
    "LongitudinalDimensionV1",
    "LongitudinalMethod",
    "LongitudinalMissingnessV1",
    "LongitudinalPlayerAssociationV1",
    "LongitudinalSessionRequestV1",
    "LongitudinalSessionResultV1",
    "LongitudinalSummaryV1",
    "METRICS",
    "MetricDefinition",
    "MetricUnitsV2",
    "MissingPolicy",
    "MissingnessV2",
    "ModelProvenanceV2",
    "MonitorComparisonResult",
    "MonitorSummary",
    "OUTCOME_PROXY_CONTRACT_VERSION",
    "OrderEvidenceV2",
    "OutcomeProxyRequestV1",
    "OutcomeProxyResultV1",
    "OutcomeProxyRowV1",
    "PCAResult",
    "POOLED_METHOD_DESCRIPTIONS",
    "PROFILES",
    "PairwiseMonitorComparison",
    "PlayerIdentityV2",
    "PooledAssociationV1",
    "PooledMethod",
    "PredictiveModelResult",
    "ProfileDetection",
    "RegressionEstimate",
    "ResidualDiagnostics",
    "STROKES_GAINED_CONTRACT_VERSION",
    "SessionAggregateV1",
    "SessionIdentityV2",
    "SourceFileReferenceV2",
    "StrokesGainedAnalysisResultV1",
    "StrokesGainedRequestV1",
    "StrokesGainedRowV1",
    "StrokesGainedUncertaintyV1",
    "TemporalTrendResult",
    "TransformRecordV2",
    "TreatmentConfig",
    "TreatmentResult",
    "UncertaintyV2",
    "VIFResult",
    "VendorProvenanceV2",
    "adapt_v2_to_v1",
    "analysis_lineage_v2",
    "analyze_dispersion",
    "analyze_longitudinal_sessions",
    "analyze_outcome_proxy",
    "analyze_source_backed_strokes_gained",
    "analyze_trend",
    "analyze_variables",
    "analyze_variables_v2",
    "apply_treatment",
    "build_analysis_lineage_v2",
    "clustered_pooled_association",
    "compare_monitors",
    "compute_correlations",
    "compute_pca",
    "compute_vif",
    "contract_v2_json_schema",
    "dersimonian_laird_pooled_association",
    "detect_profile",
    "fit_predictive_model",
    "import_session",
    "longitudinal_session_contract_json_schema",
    "metric_units_v2",
    "normalize_header",
    "numeric_metric_columns",
    "player_associations",
    "strokes_gained_contract_json_schema",
    "vendor_provenance_v2",
]
