"""Versioned impact study wire and truthful application surface (IA-T6, #5075).

The study wire is the versioned exchange record for impact studies: it pins
the model tier, exact provenance IDs, coordinate frame, launch/contact/
vibration/acoustic metric sections, completeness checks, uncertainty and
invalid-case reasons.  Serialization is deterministic (sorted keys, finite
floats only) so identical inputs produce byte-identical wires.

The application surface in :mod:`.surface` renders a study truthfully: it
labels the evidence tier, keeps ball-launch separate from sound, and
refuses unavailable acoustics instead of fabricating zeros.

Child of epic #5068; the v1 ``golf_club.impact_coupling_report`` format is
preserved verbatim and may travel inside a study through
``v1_coupling_report``.
"""

from __future__ import annotations

from shared.python.swing_sim.impact_studies.surface import (
    AcousticsUnavailableError,
    acoustic_metrics_or_raise,
    acoustics_available,
    study_statement,
)
from shared.python.swing_sim.impact_studies.wire import (
    STUDY_WIRE_FORMAT,
    CompletenessChecks,
    EvidenceResolver,
    EvidenceTier,
    ImpactStudyV1,
    InvalidCase,
    MetricRecord,
    Provenance,
    parse_study_wire,
    serialize_study_wire,
    verify_study_qualification,
)

__all__ = [
    "STUDY_WIRE_FORMAT",
    "AcousticsUnavailableError",
    "CompletenessChecks",
    "EvidenceResolver",
    "EvidenceTier",
    "ImpactStudyV1",
    "InvalidCase",
    "MetricRecord",
    "Provenance",
    "acoustic_metrics_or_raise",
    "acoustics_available",
    "parse_study_wire",
    "serialize_study_wire",
    "study_statement",
    "verify_study_qualification",
]
