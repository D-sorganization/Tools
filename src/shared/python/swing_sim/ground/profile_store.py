"""Fail-closed cooperative persistence for strict ground profile libraries."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .profile_store_platform import (
    atomic_replace,
    is_link_like,
    validated_digest,
    validated_filename,
)
from .profile_types import GroundProfileLibrary
from .profile_wire import library_from_json

DEFAULT_PROFILE_LIBRARY_FILENAME = "ground-profile-library.v1.json"
DEFAULT_PROFILE_LIBRARY_MAX_BYTES = 4 * 1024 * 1024


class ProfileStoreError(RuntimeError):
    """Base class for typed persistence failures."""


class ProfileStorePathError(ProfileStoreError):
    """A configured or discovered path violates the storage boundary."""


class ProfileStoreConflictError(ProfileStoreError):
    """The compare-and-swap precondition does not match durable state."""


class ProfileStoreCorruptionError(ProfileStoreError):
    """Durable bytes cannot be read or validated without data loss risk."""


class ProfileStoreLockError(ProfileStoreError):
    """Another writer owns the explicit store lock."""


class ProfileStoreIndeterminateCommitError(ProfileStoreCorruptionError):
    """Replacement occurred but directory durability could not be confirmed."""

    def __init__(self, destination: Path, committed_sha256: str | None) -> None:
        self.destination = destination
        self.committed_sha256 = committed_sha256
        super().__init__(
            "atomic replacement completed but durable directory sync failed"
        )


@dataclass(frozen=True)
class StoredGroundProfileLibrary:
    """A validated durable library plus canonical identity and location."""

    library: GroundProfileLibrary
    sha256: str
    path: Path

    def __post_init__(self) -> None:
        if type(self.library) is not GroundProfileLibrary:
            raise TypeError("library must be an exact GroundProfileLibrary")
        digest = validated_digest(self.sha256, "sha256")
        if digest != self.library.canonical_sha256():
            raise ValueError("sha256 does not match library")
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib Path")
        if not self.path.is_absolute():
            raise ProfileStorePathError("stored library path must be absolute")


class GroundProfileLibraryStore:
    """Single-principal explicit-directory atomic compare-and-swap store."""

    def __init__(
        self,
        root: Path,
        *,
        filename: str = DEFAULT_PROFILE_LIBRARY_FILENAME,
        max_bytes: int = DEFAULT_PROFILE_LIBRARY_MAX_BYTES,
    ) -> None:
        self._root = self._validated_root(root)
        self._root_identity = self._capture_root_identity()
        self._filename = self._validated_filename(filename)
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")
        self._max_bytes = max_bytes

    @staticmethod
    def _is_link_like(path: Path) -> bool:
        return is_link_like(path)

    @staticmethod
    def _validated_root(root: Path) -> Path:
        candidate = Path(root)
        if not candidate.is_absolute():
            raise ProfileStorePathError("profile store root must be absolute")
        try:
            if any(
                GroundProfileLibraryStore._is_link_like(component)
                for component in (candidate, *candidate.parents)
            ):
                raise ProfileStorePathError(
                    "profile store root must not contain a symbolic link"
                )
            info = os.lstat(candidate)
            if not stat.S_ISDIR(info.st_mode):
                raise ProfileStorePathError(
                    "profile store root must be an existing directory"
                )
            return candidate.resolve(strict=True)
        except ProfileStorePathError:
            raise
        except OSError as exc:
            raise ProfileStorePathError(
                "profile store root must be an existing directory"
            ) from exc

    def _capture_root_identity(self) -> tuple[int, int]:
        try:
            info = os.lstat(self._root)
        except OSError as exc:
            raise ProfileStorePathError(
                "profile store root cannot be inspected"
            ) from exc
        return (info.st_dev, info.st_ino)

    def _assert_root_identity(self) -> None:
        try:
            info = os.lstat(self._root)
            link_like = any(
                self._is_link_like(item) for item in (self._root, *self._root.parents)
            )
        except OSError as exc:
            raise ProfileStorePathError("profile store root identity changed") from exc
        identity = (info.st_dev, info.st_ino)
        if (
            link_like
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or identity != self._root_identity
        ):
            raise ProfileStorePathError("profile store root identity changed")

    @staticmethod
    def _validated_filename(filename: str) -> str:
        try:
            return validated_filename(filename)
        except ValueError as exc:
            raise ProfileStorePathError(
                "profile store filename must be a safe plain filename"
            ) from exc

    @property
    def path(self) -> Path:
        """Return the fixed primary document path."""
        return self._root / self._filename

    @property
    def backup_path(self) -> Path:
        """Return the fixed last-known-good backup path."""
        return self._root / f"{self._filename}.bak"

    @property
    def lock_path(self) -> Path:
        """Return the fixed exclusive-writer lock path."""
        return self._root / f".{self._filename}.lock"

    def load(self) -> StoredGroundProfileLibrary:
        """Load only the primary; backup recovery is never implicit."""
        stored, _ = self._read_document(self.path)
        return stored

    def load_backup(self) -> StoredGroundProfileLibrary:
        """Read and validate the explicit last-known-good backup."""
        stored, _ = self._read_document(self.backup_path)
        return stored

    def save(
        self,
        library: GroundProfileLibrary,
        *,
        expected_sha256: str | None,
    ) -> StoredGroundProfileLibrary:
        """Create or CAS-replace the library and retain the previous bytes."""
        if type(library) is not GroundProfileLibrary:
            raise TypeError("library must be an exact GroundProfileLibrary")
        if expected_sha256 is not None:
            validated_digest(expected_sha256, "expected_sha256")
        payload = library.to_json().encode("utf-8")
        self._check_payload_size(payload)
        with self._exclusive_lock():
            current = self._current_for_save(expected_sha256)
            if current is not None:
                _, current_bytes = current
                self._atomic_write(self.backup_path, current_bytes)
            self._atomic_write(self.path, payload)
            stored, _ = self._read_document(self.path)
        if stored.sha256 != library.canonical_sha256():
            raise ProfileStoreCorruptionError(
                "durable library digest changed after write"
            )
        return stored

    def _current_for_save(
        self, expected_sha256: str | None
    ) -> tuple[StoredGroundProfileLibrary, bytes] | None:
        if not self._path_exists_or_link(self.path):
            if expected_sha256 is not None:
                raise ProfileStoreConflictError("primary library is absent")
            return None
        current = self._read_document(self.path)
        if expected_sha256 is None:
            raise ProfileStoreConflictError("primary library already exists")
        if current[0].sha256 != expected_sha256:
            raise ProfileStoreConflictError("primary library digest does not match")
        return current

    def recover_from_backup(
        self, *, expected_primary_sha256: str, expected_backup_sha256: str
    ) -> StoredGroundProfileLibrary:
        """Explicitly replace exact primary bytes with a validated backup."""
        validated_digest(expected_primary_sha256, "expected_primary_sha256")
        validated_digest(expected_backup_sha256, "expected_backup_sha256")
        with self._exclusive_lock():
            primary_bytes = self._read_bounded_bytes(self.path)
            primary_digest = hashlib.sha256(primary_bytes).hexdigest()
            if primary_digest != expected_primary_sha256:
                raise ProfileStoreConflictError("primary byte digest does not match")
            backup, backup_bytes = self._read_document(self.backup_path)
            if backup.sha256 != expected_backup_sha256:
                raise ProfileStoreConflictError("backup library digest does not match")
            self._atomic_write(self.path, backup_bytes)
            recovered, _ = self._read_document(self.path)
        return recovered

    def _read_document(self, path: Path) -> tuple[StoredGroundProfileLibrary, bytes]:
        payload = self._read_bounded_bytes(path)
        try:
            text = payload.decode("utf-8")
            library = library_from_json(text)
        except (TypeError, UnicodeDecodeError, ValueError) as exc:
            raise ProfileStoreCorruptionError(
                f"profile library at {path.name} is invalid"
            ) from exc
        digest = hashlib.sha256(payload).hexdigest()
        return StoredGroundProfileLibrary(library, digest, path), payload

    def _read_bounded_bytes(self, path: Path) -> bytes:
        self._assert_root_identity()
        self._reject_unsafe_existing_path(path)
        try:
            if not path.exists() or not path.is_file():
                raise ProfileStoreCorruptionError(
                    f"profile library at {path.name} is absent"
                )
            size = path.stat().st_size
            if size > self._max_bytes:
                raise ProfileStoreCorruptionError("profile library exceeds size limit")
            payload = path.read_bytes()
        except OSError as exc:
            raise ProfileStoreCorruptionError("profile library cannot be read") from exc
        self._check_payload_size(payload)
        return payload

    def _check_payload_size(self, payload: bytes) -> None:
        if len(payload) > self._max_bytes:
            raise ProfileStoreCorruptionError("profile library exceeds size limit")

    @staticmethod
    def _reject_unsafe_existing_path(path: Path) -> None:
        try:
            if GroundProfileLibraryStore._is_link_like(path):
                raise ProfileStorePathError(f"{path.name} must not be a reparse point")
        except ProfileStorePathError:
            raise
        except OSError as exc:
            raise ProfileStorePathError(f"{path.name} cannot be inspected") from exc

    def _path_exists_or_link(self, path: Path) -> bool:
        self._assert_root_identity()
        try:
            return path.exists() or path.is_symlink()
        except OSError as exc:
            raise ProfileStorePathError(f"{path.name} cannot be inspected") from exc

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self._assert_root_identity()
        self._reject_unsafe_existing_path(self.lock_path)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
        except FileExistsError as exc:
            raise ProfileStoreLockError("profile library store is locked") from exc
        except OSError as exc:
            raise ProfileStoreLockError(
                "profile library lock cannot be created"
            ) from exc
        try:
            try:
                os.write(descriptor, str(os.getpid()).encode("ascii"))
                os.fsync(descriptor)
                os.close(descriptor)
            except OSError as exc:
                raise ProfileStoreLockError(
                    "profile library lock cannot be initialized"
                ) from exc
            descriptor = -1
            self._assert_root_identity()
            yield
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError as exc:
                raise ProfileStoreLockError(
                    "profile library lock cannot be removed"
                ) from exc

    def _atomic_write(self, destination: Path, payload: bytes) -> None:
        self._assert_root_identity()
        self._reject_unsafe_existing_path(destination)
        temporary: Path | None = None
        replaced = False
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=self._root
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._replace_atomic(temporary, destination)
            replaced = True
            self._sync_directory()
        except OSError as exc:
            if replaced:
                committed_sha = self._committed_digest(destination)
                raise ProfileStoreIndeterminateCommitError(
                    destination, committed_sha
                ) from exc
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            raise ProfileStoreCorruptionError("atomic write failed") from exc

    @staticmethod
    def _replace_atomic(source: Path, destination: Path) -> None:
        atomic_replace(source, destination)

    def _committed_digest(self, destination: Path) -> str | None:
        try:
            payload = self._read_bounded_bytes(destination)
        except ProfileStoreError:
            return None
        return hashlib.sha256(payload).hexdigest()

    def _sync_directory(self) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(self._root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "DEFAULT_PROFILE_LIBRARY_FILENAME",
    "DEFAULT_PROFILE_LIBRARY_MAX_BYTES",
    "GroundProfileLibraryStore",
    "ProfileStoreConflictError",
    "ProfileStoreCorruptionError",
    "ProfileStoreError",
    "ProfileStoreIndeterminateCommitError",
    "ProfileStoreLockError",
    "ProfileStorePathError",
    "StoredGroundProfileLibrary",
]
