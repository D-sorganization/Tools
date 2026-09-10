"""Immutable declared distributed coefficients, independent of operating state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from ._shaft_inertia import SectionInertia
from ._shaft_section import SectionElement
from ._validation import require_identifier

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_KINDS = frozenset({"synthetic", "analytical", "measurement-derived"})


def _digest(value: object, name: str) -> str:
    result: str = require_identifier(value, name)
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return result


@dataclass(frozen=True)
class ShaftModelSource:
    """Declared coefficient derivation and exact supporting artifact identities.

    Attributes:
        source_id: Unique identifier referenced by section records.
        kind: Synthetic, analytical or measurement-derived declaration.
        artifact_sha256: Digest of the exact derivation/data artifact bytes.
        calibration_sha256: Required for measurement-derived declarations.
        method: How coefficients were derived, including quadrature if relevant.
        uncertainty_note: Explicit limitations; unquantified is not zero.
        data_license: Declared license for the supporting artifact.

    Neither construction nor matching bytes authenticates the declarations,
    validates a calibration or qualifies a physical frequency/strain domain.
    """

    source_id: str
    kind: str
    artifact_sha256: str
    calibration_sha256: str | None
    method: str
    uncertainty_note: str
    data_license: str

    def __post_init__(self) -> None:
        for name in ("source_id", "kind", "method", "uncertainty_note", "data_license"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        if self.kind not in _SOURCE_KINDS:
            raise ValueError("unsupported source kind")
        object.__setattr__(
            self, "artifact_sha256", _digest(self.artifact_sha256, "artifact_sha256")
        )
        if self.calibration_sha256 is not None:
            object.__setattr__(
                self,
                "calibration_sha256",
                _digest(self.calibration_sha256, "calibration_sha256"),
            )
        if self.kind == "measurement-derived" and self.calibration_sha256 is None:
            raise ValueError("measurement-derived source requires calibration_sha256")


@dataclass(frozen=True)
class ShaftModelSection:
    """One explicit elastic law and integrated inertia quadrature with sources.

    The existing section kernels own the constitutive, geometric and physical
    mass contracts. This record does not infer shear, axial or rotary terms.
    """

    elastic: SectionElement
    inertia: SectionInertia
    coefficient_source_id: str
    inertia_source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.elastic, SectionElement):
            raise TypeError("elastic must be SectionElement")
        if not isinstance(self.inertia, SectionInertia):
            raise TypeError("inertia must be SectionInertia")
        for name in ("coefficient_source_id", "inertia_source_id"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )

    @property
    def material_frame_ids(self) -> frozenset[str]:
        """Return the material convention of all physical quadrature samples."""
        return frozenset(sample.body.frame_id for sample in self.inertia.samples)


@dataclass(frozen=True)
class DistributedShaftModel:
    """Declared ordered section model, without a grip, head or operating state.

    Preconditions: nonempty sections, one material-axis convention, unique
    source identifiers and exactly the set of sources referenced by sections.
    Postconditions: immutable owned tuples; no experimental validation claim.
    Sources are sorted by identifier for deterministic metadata ordering.
    """

    model_id: str
    material_frame_id: str
    sections: tuple[ShaftModelSection, ...]
    sources: tuple[ShaftModelSource, ...]
    validation_status: ClassVar[str] = "unqualified"

    def __post_init__(self) -> None:
        for name in ("model_id", "material_frame_id"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        sections, sources = tuple(self.sections), tuple(self.sources)
        if not sections:
            raise ValueError("model requires sections")
        if any(not isinstance(item, ShaftModelSection) for item in sections):
            raise TypeError("sections must contain ShaftModelSection")
        if any(not isinstance(item, ShaftModelSource) for item in sources):
            raise TypeError("sources must contain ShaftModelSource")
        identifiers = {source.source_id for source in sources}
        if len(identifiers) != len(sources):
            raise ValueError("source identifiers must be unique")
        used = {
            identifier
            for section in sections
            for identifier in (section.coefficient_source_id, section.inertia_source_id)
        }
        if used != identifiers:
            raise ValueError("source identifiers must exactly match section references")
        if any(
            section.material_frame_ids != {self.material_frame_id}
            for section in sections
        ):
            raise ValueError("all samples must use the declared material frame")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(
            self, "sources", tuple(sorted(sources, key=lambda item: item.source_id))
        )


__all__ = ()
