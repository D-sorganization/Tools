"""Path-free, bounded preview artifacts and non-destructive cleanup evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ._validation import require_text
from .experiment_resources import ExperimentResourceBudget

PREVIEW_MANIFEST_SCHEMA_VERSION = "preview-manifest/1.0.0"
DEFAULT_PREVIEW_DIRECTORY = "Motion_Matching_Previews"
_OPAQUE_TOKEN = re.compile(r"^opaque:[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VIDEO_TYPES = frozenset({"video/mp4", "video/webm"})


def _positive_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_digest(value: str, field: str) -> str:
    digest = require_text(value, field)
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return str(digest)


def _require_opaque(value: str, field: str) -> str:
    token = require_text(value, field)
    if not _OPAQUE_TOKEN.fullmatch(token):
        raise ValueError(f"{field} must be an opaque identifier")
    return str(token)


@dataclass(frozen=True, slots=True)
class PreviewArtifact:
    """A path-free digest reference to one completed preview video."""

    artifact_id: str
    sha256: str
    byte_size: int
    media_type: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", _require_opaque(self.artifact_id, "artifact_id")
        )
        object.__setattr__(self, "sha256", _require_digest(self.sha256, "sha256"))
        _positive_integer(self.byte_size, "byte_size")
        if self.media_type not in _VIDEO_TYPES:
            raise ValueError("preview artifacts must be an approved video media type")


@dataclass(frozen=True, slots=True)
class PreviewManifest:
    """Completed, bounded video inventory without paths or source metadata."""

    experiment_id: str
    cache_key_sha256: str
    provider_sha256: str
    artifacts: tuple[PreviewArtifact, ...]
    schema_version: str = PREVIEW_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", _require_opaque(self.experiment_id, "experiment_id")
        )
        object.__setattr__(
            self,
            "cache_key_sha256",
            _require_digest(self.cache_key_sha256, "cache_key_sha256"),
        )
        object.__setattr__(
            self,
            "provider_sha256",
            _require_digest(self.provider_sha256, "provider_sha256"),
        )
        if self.schema_version != PREVIEW_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported preview manifest schema version")
        artifacts = tuple(self.artifacts)
        if not artifacts or any(
            not isinstance(item, PreviewArtifact) for item in artifacts
        ):
            raise ValueError("manifest requires completed preview artifacts")
        object.__setattr__(self, "artifacts", artifacts)
        ids = tuple(item.artifact_id for item in artifacts)
        if len(ids) != len(set(ids)):
            raise ValueError("preview artifact IDs must be unique")

    @classmethod
    def create(
        cls,
        experiment_id: str,
        artifacts: tuple[PreviewArtifact, ...],
        budget: ExperimentResourceBudget,
        *,
        cache_key_sha256: str,
        provider_sha256: str,
    ) -> PreviewManifest:
        """Build only a complete manifest within the caller's output budget."""
        if not isinstance(budget, ExperimentResourceBudget):
            raise TypeError("budget must be an ExperimentResourceBudget")
        artifact_tuple = tuple(artifacts)
        if not artifact_tuple or any(
            not isinstance(item, PreviewArtifact) for item in artifact_tuple
        ):
            raise ValueError("manifest requires completed preview artifacts")
        if len(artifact_tuple) > budget.max_preview_artifacts:
            raise ValueError("preview artifact count exceeds configured budget")
        if sum(item.byte_size for item in artifact_tuple) > budget.max_preview_bytes:
            raise ValueError("preview output exceeds configured byte budget")
        return cls(experiment_id, cache_key_sha256, provider_sha256, artifact_tuple)

    @property
    def sha256(self) -> str:
        """Return the stable digest of the complete path-free manifest payload."""
        encoded = json.dumps(
            _manifest_payload(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _manifest_payload(manifest: PreviewManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "experiment_id": manifest.experiment_id,
        "cache_key_sha256": manifest.cache_key_sha256,
        "provider_sha256": manifest.provider_sha256,
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "sha256": artifact.sha256,
                "byte_size": artifact.byte_size,
                "media_type": artifact.media_type,
            }
            for artifact in manifest.artifacts
        ],
    }


def build_preview_root(
    environment: Mapping[str, str], *, desktop_directory: Path | None = None
) -> Path:
    """Resolve configured preview storage without creating a directory."""
    configured = environment.get("MOTION_MATCHING_PREVIEW_ROOT")
    if configured is not None:
        if not configured.strip():
            raise ValueError("MOTION_MATCHING_PREVIEW_ROOT cannot be empty")
        return Path(configured).expanduser()
    desktop = desktop_directory or Path.home() / "Desktop"
    return desktop / DEFAULT_PREVIEW_DIRECTORY


@dataclass(frozen=True, slots=True)
class RemoteResourceReceipt:
    """Path-free preflight evidence for an optional remote execution host."""

    host_id: str
    provider_id: str
    provider_sha256: str
    license_identity_sha256: str
    preview_root_identity: str
    access_confirmed: bool
    available_workers: int
    available_memory_bytes: int
    available_disk_bytes: int

    def __post_init__(self) -> None:
        for field in ("host_id", "provider_id"):
            object.__setattr__(self, field, require_text(getattr(self, field), field))
        for field in ("provider_sha256", "license_identity_sha256"):
            object.__setattr__(
                self, field, _require_digest(getattr(self, field), field)
            )
        object.__setattr__(
            self,
            "preview_root_identity",
            _require_opaque(self.preview_root_identity, "preview_root_identity"),
        )
        if not isinstance(self.access_confirmed, bool):
            raise TypeError("access_confirmed must be a boolean")
        for field in (
            "available_workers",
            "available_memory_bytes",
            "available_disk_bytes",
        ):
            _nonnegative_integer(getattr(self, field), field)

    def validate_for_dispatch(
        self,
        *,
        required_host_id: str,
        required_provider_id: str,
        required_provider_sha256: str,
        required_license_identity_sha256: str,
        required_preview_root_identity: str,
        workers: int,
        memory_bytes: int,
        disk_bytes: int,
    ) -> None:
        """Fail closed unless provider, license, access, and capacity match."""
        required_host_id = require_text(required_host_id, "required_host_id")
        required_provider_id = require_text(
            required_provider_id, "required_provider_id"
        )
        required_provider_sha256 = _require_digest(
            required_provider_sha256, "required_provider_sha256"
        )
        required_license_identity_sha256 = _require_digest(
            required_license_identity_sha256, "required_license_identity_sha256"
        )
        required_preview_root_identity = _require_opaque(
            required_preview_root_identity, "required_preview_root_identity"
        )
        if self.host_id != required_host_id:
            raise ValueError("remote host identity does not match request")
        if (
            self.provider_id != required_provider_id
            or self.provider_sha256 != required_provider_sha256
        ):
            raise ValueError("remote provider identity does not match request")
        if self.license_identity_sha256 != required_license_identity_sha256:
            raise ValueError("remote license identity does not match request")
        if self.preview_root_identity != required_preview_root_identity:
            raise ValueError("remote preview root identity does not match request")
        if not self.access_confirmed:
            raise ValueError("remote preview root access is not confirmed")
        capacity = (
            (workers, self.available_workers, "worker"),
            (memory_bytes, self.available_memory_bytes, "memory"),
            (disk_bytes, self.available_disk_bytes, "disk"),
        )
        for requested, available, resource in capacity:
            _positive_integer(requested, f"requested {resource}")
            if requested > available:
                raise ValueError(f"remote host lacks required {resource} capacity")

    def validate_returned_manifest(
        self,
        manifest: PreviewManifest,
        *,
        expected_cache_key_sha256: str,
        expected_provider_sha256: str,
    ) -> None:
        """Bind returned preview metadata to the exact replay key and host provider."""
        if not isinstance(manifest, PreviewManifest):
            raise TypeError("manifest must be a PreviewManifest")
        expected_cache_key_sha256 = _require_digest(
            expected_cache_key_sha256, "expected_cache_key_sha256"
        )
        expected_provider_sha256 = _require_digest(
            expected_provider_sha256, "expected_provider_sha256"
        )
        if manifest.cache_key_sha256 != expected_cache_key_sha256:
            raise ValueError("remote manifest cache identity differs from request")
        if manifest.provider_sha256 != expected_provider_sha256:
            raise ValueError("remote manifest provider differs from request")
        if self.provider_sha256 != manifest.provider_sha256:
            raise ValueError("remote manifest provider differs from host receipt")

    def validate_returned_artifacts(
        self,
        manifest: PreviewManifest,
        *,
        expected_cache_key_sha256: str,
        expected_provider_sha256: str,
        preview_root: Path,
        artifact_paths: Mapping[str, Path],
    ) -> None:
        """Validate remote run identity and verify every returned artifact file."""
        self.validate_returned_manifest(
            manifest,
            expected_cache_key_sha256=expected_cache_key_sha256,
            expected_provider_sha256=expected_provider_sha256,
        )
        verify_preview_artifacts(manifest, preview_root, artifact_paths)


@dataclass(frozen=True, slots=True)
class PreviewReleaseAuthorization:
    """Opaque approval bound to the exact manifest intended for release."""

    authorization_id: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authorization_id",
            _require_opaque(self.authorization_id, "authorization_id"),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _require_digest(self.manifest_sha256, "manifest_sha256"),
        )


def require_preview_release(
    manifest: PreviewManifest,
    *,
    source_is_private: bool,
    authorization: PreviewReleaseAuthorization | None,
) -> None:
    """Require exact-manifest approval before releasing previews from private data."""
    if not isinstance(manifest, PreviewManifest):
        raise TypeError("manifest must be a PreviewManifest")
    if not isinstance(source_is_private, bool):
        raise TypeError("source_is_private must be a boolean")
    if not source_is_private:
        return
    if authorization is None:
        raise ValueError("private preview requires explicit release authorization")
    if authorization.manifest_sha256 != manifest.sha256:
        raise ValueError("private preview authorization does not match manifest")


def verify_preview_artifacts(
    manifest: PreviewManifest,
    preview_root: Path,
    artifact_paths: Mapping[str, Path],
) -> None:
    """Verify resolved files stay under the root and match size and digest claims."""
    if not isinstance(manifest, PreviewManifest):
        raise TypeError("manifest must be a PreviewManifest")
    if not isinstance(preview_root, Path):
        raise TypeError("preview_root must be a pathlib.Path")
    root = preview_root.resolve(strict=True)
    expected_ids = {artifact.artifact_id for artifact in manifest.artifacts}
    if set(artifact_paths) != expected_ids:
        raise ValueError("resolved artifact IDs do not match the manifest")
    artifact_by_id = {artifact.artifact_id: artifact for artifact in manifest.artifacts}
    for artifact_id, candidate in artifact_paths.items():
        resolved = _resolve_artifact(root, candidate)
        if not resolved.is_file():
            raise ValueError("preview artifact is incomplete")
        artifact = artifact_by_id[artifact_id]
        expected_suffix = ".mp4" if artifact.media_type == "video/mp4" else ".webm"
        if resolved.suffix.casefold() != expected_suffix:
            raise ValueError("preview file extension differs from declared media type")
        if resolved.stat().st_size != artifact.byte_size:
            raise ValueError("preview artifact is incomplete or has a different size")
        if _file_sha256(resolved) != artifact.sha256:
            raise ValueError("preview artifact digest differs from manifest")


def _resolve_artifact(root: Path, candidate: Path) -> Path:
    if not isinstance(candidate, Path):
        raise TypeError("artifact paths must be pathlib.Path values")
    unresolved = candidate if candidate.is_absolute() else root / candidate
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("resolved preview artifact escapes configured root")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
