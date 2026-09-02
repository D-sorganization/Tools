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
                             Rulings D15/D17 land in a follow-up, not the port.
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
                             moved; rulings D15/D17 land in a follow-up.
P11   :mod:`.contract_v2`    The v2 serialization boundary over P10: evidence,
                             row-level lineage, availability, and the JSON
                             Schema every static client is generated from.
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
from .modeling import PredictiveModelResult, fit_predictive_model
from .multivariate import PCAResult, VIFResult, compute_pca, compute_vif
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
from .treatment import (
    FilterRule,
    TreatmentConfig,
    TreatmentResult,
    apply_treatment,
)
from .trends import ChangeCandidate, TemporalTrendResult, analyze_trend

__all__ = [
    "COMMON_ALIASES",
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_V2",
    "IDENTITY_COLUMNS",
    "METRICS",
    "PROFILES",
    "AnalysisContextV2",
    "AnalysisLineageV2",
    "AnalysisMode",
    "AvailabilityV2",
    "BackingRecordV2",
    "ChangeCandidate",
    "ClaimsV2",
    "CoefficientEstimate",
    "ColumnMapping",
    "CorrelationEstimate",
    "CorrelationMethod",
    "CorrelationResult",
    "DatasetAuthorityV2",
    "DatasetSummary",
    "DependencyEdge",
    "DispersionResult",
    "FilterRule",
    "FlexibleAnalysisRequest",
    "FlexibleAnalysisResult",
    "GroupAnalysis",
    "ImportManifest",
    "ImportOptions",
    "ImportProfile",
    "ImportedSession",
    "LaunchMonitorAnalysisResultV2",
    "MetricDefinition",
    "MetricUnitsV2",
    "MissingnessV2",
    "MissingPolicy",
    "MonitorComparisonResult",
    "MonitorSummary",
    "ModelProvenanceV2",
    "OrderEvidenceV2",
    "PCAResult",
    "PairwiseMonitorComparison",
    "PlayerIdentityV2",
    "PredictiveModelResult",
    "ProfileDetection",
    "RegressionEstimate",
    "ResidualDiagnostics",
    "SessionIdentityV2",
    "SourceFileReferenceV2",
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
    "analyze_trend",
    "analyze_variables",
    "analyze_variables_v2",
    "build_analysis_lineage_v2",
    "apply_treatment",
    "compare_monitors",
    "compute_correlations",
    "compute_pca",
    "compute_vif",
    "contract_v2_json_schema",
    "detect_profile",
    "fit_predictive_model",
    "import_session",
    "metric_units_v2",
    "normalize_header",
    "numeric_metric_columns",
    "vendor_provenance_v2",
]
