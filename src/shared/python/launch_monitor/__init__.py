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
====  ====================  ==================================================
Step  Module                Notes
====  ====================  ==================================================
P1    :mod:`.dispersion`    2-D target-relative dispersion. Shares a name with
                            ``rate_of_closure`` and nothing else — G0 pinned
                            the gap as divergences D6-D9.
P2    :mod:`.multivariate`  PCA and variance-inflation diagnostics. No
                            ``rate_of_closure`` counterpart exists.
P3    :mod:`.trends`        Per-calendar-day robust trend, EWMA, and ranked
                            change candidates. ``TrendResult`` is renamed
                            ``TemporalTrendResult`` here — see that module.
====  ====================  ==================================================

**Name-collision containment.** Symbols in this package collide by name with
``rate_of_closure`` symbols that compute something else — ``analyze_dispersion``
and ``DispersionResult`` already do, and ``TrendResult`` would have, which is
why P3 renamed it. The separate package is what keeps the rest apart, and that
containment lasts exactly as long as nobody adds a convenience re-export
between the two packages. Do not add one.
"""

from .dispersion import DispersionResult, analyze_dispersion
from .multivariate import PCAResult, VIFResult, compute_pca, compute_vif
from .trends import ChangeCandidate, TemporalTrendResult, analyze_trend

__all__ = [
    "ChangeCandidate",
    "DispersionResult",
    "PCAResult",
    "TemporalTrendResult",
    "VIFResult",
    "analyze_dispersion",
    "analyze_trend",
    "compute_pca",
    "compute_vif",
]
