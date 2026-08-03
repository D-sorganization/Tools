"""Self-contained ZIP adapter for synthetic acceptance evidence."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict
from scenario_evidence import (
    ScenarioDefinition,
    ScenarioEvidence,
    canonical_model_bytes,
    sha256_bytes,
)

PACKAGE_SCHEMA = "p1am.acceptance-package/v1"
PACKAGE_ENTRIES = frozenset({"manifest.json", "scenario.json", "evidence.json"})
MAX_PACKAGE_BYTES = 5_000_000


class EvidencePackageManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: Literal[PACKAGE_SCHEMA] = PACKAGE_SCHEMA
    evidence_id: str
    data_classification: Literal["synthetic"] = "synthetic"
    not_for_live_control: Literal[True] = True
    entries: dict[str, str]


@dataclass(frozen=True)
class EvidenceArtifact:
    payload: bytes = field(repr=False)
    sha256: str
    manifest: EvidencePackageManifest


@dataclass(frozen=True)
class VerifiedEvidencePackage:
    manifest: EvidencePackageManifest
    scenario: ScenarioDefinition
    evidence: ScenarioEvidence
    package_sha256: str


class EvidencePackageService:
    def create(
        self, scenario: ScenarioDefinition, evidence: ScenarioEvidence
    ) -> EvidenceArtifact:
        scenario_payload = canonical_model_bytes(scenario)
        evidence_payload = canonical_model_bytes(evidence)
        if evidence.scenario_sha256 != sha256_bytes(scenario_payload):
            raise ValueError("evidence does not identify the supplied scenario")
        manifest = EvidencePackageManifest(
            evidence_id=evidence.evidence_id,
            entries={
                "scenario.json": sha256_bytes(scenario_payload),
                "evidence.json": sha256_bytes(evidence_payload),
            },
        )
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest.model_dump_json(indent=2))
            archive.writestr("scenario.json", scenario_payload)
            archive.writestr("evidence.json", evidence_payload)
        payload = output.getvalue()
        return EvidenceArtifact(payload, sha256_bytes(payload), manifest)

    def verify(
        self, payload: bytes, expected_sha256: str | None = None
    ) -> VerifiedEvidencePackage:
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > MAX_PACKAGE_BYTES
        ):
            raise ValueError("evidence package size is outside the allowed boundary")
        package_sha = sha256_bytes(payload)
        if expected_sha256 is not None and package_sha != expected_sha256.lower():
            raise ValueError("evidence package checksum does not match")
        try:
            with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                if frozenset(archive.namelist()) != PACKAGE_ENTRIES:
                    raise ValueError("evidence package entries are not allowed")
                manifest_payload = archive.read("manifest.json")
                scenario_payload = archive.read("scenario.json")
                evidence_payload = archive.read("evidence.json")
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError("evidence package is not a valid archive") from exc
        manifest = EvidencePackageManifest.model_validate_json(manifest_payload)
        for name, content in (
            ("scenario.json", scenario_payload),
            ("evidence.json", evidence_payload),
        ):
            if manifest.entries.get(name) != sha256_bytes(content):
                raise ValueError(f"{name} checksum does not match")
        scenario = ScenarioDefinition.model_validate_json(scenario_payload)
        evidence = ScenarioEvidence.model_validate_json(evidence_payload)
        if evidence.scenario_sha256 != sha256_bytes(canonical_model_bytes(scenario)):
            raise ValueError("evidence scenario identity does not match")
        if evidence.evidence_id != manifest.evidence_id:
            raise ValueError("evidence identity does not match the manifest")
        return VerifiedEvidencePackage(manifest, scenario, evidence, package_sha)
