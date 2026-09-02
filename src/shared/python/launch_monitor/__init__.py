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

The ladder is complete: **P1 through P20 have landed.** Each module's own
docstring carries its step number, its line count, what travelled and what did
not, and any ruling it implements. Read that first; this file only re-exports.

Rows that are not plain ports
-----------------------------
Three modules changed shape on the way, and none of it is recoverable from a
diff:

* **P3** :mod:`.trends` renamed ``TrendResult`` to ``TemporalTrendResult``,
  deliberately with no back-compat alias, because ``rate_of_closure`` exports
  the same name for a different estimand. Stage 2's import rewrite must
  special-case this symbol.
* **P12** :mod:`.strokes_gained_types` landed **minus the expected-strokes
  baseline half**, which the plan names as the one sub-module already home in
  ``rate_of_closure.launch_monitor_strokes_gained_baseline``. Two structural
  protocols reach it instead of an import.
* **P18** :mod:`.player_covariation` and **P19** :mod:`.corpus` are **merges,
  not ports** (ADR-0046 Amendment 1): ``rate_of_closure`` carries a
  same-shaped counterpart for each, and neither side was a subset.
  UpstreamDrift is the base and every ``rate_of_closure``-only capability is
  folded in explicitly. P19's merge is the one that changes behaviour:
  manifest validation is now **mandatory**, so the canonical loader refuses
  corpora UpstreamDrift's accepted.

Owner rulings applied here
--------------------------
**D15** and **D17** in :mod:`.relationships` / :mod:`.flexible_analysis`;
**D22** and **D23** in :mod:`.player_covariation*`; **G1-D1** in
:mod:`.longitudinal*` (the named pooled-estimator pair); **G1-D2** in
:mod:`.strokes_gained` (the session cell is the canonical estimand);
**G1-D3** as ported (exclude-and-audit). Each ruling's *legacy* half — the
matching change to ``rate_of_closure`` — is a coordinated cross-repo change
that UpstreamDrift's drift gates pin, and is tracked rather than smuggled into
a Tools-only PR.

**Name-collision containment.** Symbols in this package collide by name with
``rate_of_closure`` symbols that compute something else — ``analyze_dispersion``
and ``DispersionResult`` already do, and ``TrendResult`` would have, which is
why P3 renamed it. The separate package is what keeps the rest apart, and that
containment lasts exactly as long as nobody adds a convenience re-export
between the two packages. Do not add one. Every module in this package carries
an AST pin asserting it does not import ``rate_of_closure``.
"""

from .comparison import (
    MonitorComparisonResult,
    MonitorSummary,
    PairwiseMonitorComparison,
    compare_monitors,
)
from .conformance_bundle import (
    LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION,
    LaunchMonitorConformanceBundleV1,
    LaunchMonitorConformanceScenarioV1,
    launch_monitor_conformance_bundle_json_schema,
    launch_monitor_conformance_bundle_sha256,
    launch_monitor_conformance_scenario_sha256,
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
from .corpus import (
    CORPUS_COLUMN_MAP,
    CORPUS_RELATIVE_PATH,
    MAX_RETAINED_ROWS,
    PRIVATE_DATA_ENV,
    CanonicalPrivateCorpus,
    CorpusManifest,
    corpus_dataset_path,
    load_private_corpus,
    load_private_corpus_with_provenance,
    read_corpus_manifest,
    resolve_private_corpus_path,
    validate_corpus_manifest,
)
from .dataset_reference import (
    DATASET_JOB_CONTRACT_VERSION,
    MAX_PAGE_SIZE,
    DatasetJobRequestV1,
    DatasetOperationV1,
    DatasetReferenceV1,
    DatasetUnavailableError,
    DatasetUnavailableStateV1,
    VerifiedDataset,
    dataset_content_sha256,
    dataset_job_contract_json_schema,
    execute_dataset_operation,
    verify_dataset_reference,
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
from .player_covariation import (
    SELECTED_PAIR_METHOD_DESCRIPTION,
    analyze_player_covariation_v1,
    covariation_backing_frame,
    player_association_frame,
    player_covariation_contract_json_schema,
    scan_player_covariation_v1,
)
from .player_covariation_types import (
    BETWEEN_PLAYER_INTERVAL_MIN_GROUPS,
    MIN_FISHER_SAMPLES,
    PLAYER_COVARIATION_CONTRACT_VERSION,
    AssociationEstimateV1,
    CovariationMissingnessV1,
    CovariationPairRankV1,
    CovariationUncertaintyV1,
    MetaAnalysisSummaryV1,
    PlayerAssociationV1,
    PlayerCovariationContractV1,
    PlayerCovariationRequestV1,
    PlayerCovariationResultV1,
    PlayerCovariationScanRequestV1,
    PlayerCovariationScanResultV1,
)
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
    "AssociationEstimateV1",
    "AvailabilityV1",
    "AvailabilityV2",
    "BASELINE_CONTRACT_VERSION",
    "BETWEEN_PLAYER_INTERVAL_MIN_GROUPS",
    "BackingRecordV2",
    "BaselineProvenanceV1",
    "COMMON_ALIASES",
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_V2",
    "CORPUS_COLUMN_MAP",
    "CORPUS_RELATIVE_PATH",
    "CanonicalPrivateCorpus",
    "ChangeCandidate",
    "ClaimsV2",
    "CoefficientEstimate",
    "ColumnMapping",
    "ConfidenceIntervalV1",
    "CorpusManifest",
    "CorrelationEstimate",
    "CorrelationMethod",
    "CorrelationResult",
    "CourseStateColumnsV1",
    "CourseStateValueV1",
    "CovariationMissingnessV1",
    "CovariationPairRankV1",
    "CovariationUncertaintyV1",
    "DATASET_JOB_CONTRACT_VERSION",
    "DatasetAuthorityV2",
    "DatasetJobRequestV1",
    "DatasetOperationV1",
    "DatasetReferenceV1",
    "DatasetSummary",
    "DatasetUnavailableError",
    "DatasetUnavailableStateV1",
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
    "LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION",
    "LONGITUDINAL_SESSION_CONTRACT_VERSION",
    "LaunchMonitorAnalysisResultV2",
    "LaunchMonitorConformanceBundleV1",
    "LaunchMonitorConformanceScenarioV1",
    "LongitudinalClaimsV1",
    "LongitudinalDesignV1",
    "LongitudinalDimensionV1",
    "LongitudinalMethod",
    "LongitudinalMissingnessV1",
    "LongitudinalPlayerAssociationV1",
    "LongitudinalSessionRequestV1",
    "LongitudinalSessionResultV1",
    "LongitudinalSummaryV1",
    "MAX_PAGE_SIZE",
    "MAX_RETAINED_ROWS",
    "METRICS",
    "MIN_FISHER_SAMPLES",
    "MetaAnalysisSummaryV1",
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
    "PLAYER_COVARIATION_CONTRACT_VERSION",
    "POOLED_METHOD_DESCRIPTIONS",
    "PRIVATE_DATA_ENV",
    "PROFILES",
    "PairwiseMonitorComparison",
    "PlayerAssociationV1",
    "PlayerCovariationContractV1",
    "PlayerCovariationRequestV1",
    "PlayerCovariationResultV1",
    "PlayerCovariationScanRequestV1",
    "PlayerCovariationScanResultV1",
    "PlayerIdentityV2",
    "PooledAssociationV1",
    "PooledMethod",
    "PredictiveModelResult",
    "ProfileDetection",
    "RegressionEstimate",
    "ResidualDiagnostics",
    "SELECTED_PAIR_METHOD_DESCRIPTION",
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
    "VerifiedDataset",
    "adapt_v2_to_v1",
    "analysis_lineage_v2",
    "analyze_dispersion",
    "analyze_longitudinal_sessions",
    "analyze_outcome_proxy",
    "analyze_player_covariation_v1",
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
    "corpus_dataset_path",
    "covariation_backing_frame",
    "dataset_content_sha256",
    "dataset_job_contract_json_schema",
    "dersimonian_laird_pooled_association",
    "detect_profile",
    "execute_dataset_operation",
    "fit_predictive_model",
    "import_session",
    "launch_monitor_conformance_bundle_json_schema",
    "launch_monitor_conformance_bundle_sha256",
    "launch_monitor_conformance_scenario_sha256",
    "load_private_corpus",
    "load_private_corpus_with_provenance",
    "longitudinal_session_contract_json_schema",
    "metric_units_v2",
    "normalize_header",
    "numeric_metric_columns",
    "player_association_frame",
    "player_associations",
    "player_covariation_contract_json_schema",
    "read_corpus_manifest",
    "resolve_private_corpus_path",
    "scan_player_covariation_v1",
    "strokes_gained_contract_json_schema",
    "validate_corpus_manifest",
    "vendor_provenance_v2",
    "verify_dataset_reference",
]
