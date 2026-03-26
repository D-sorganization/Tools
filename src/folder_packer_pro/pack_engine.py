"""Core pack/unpack engine for Folder Packer Pro.

Provides non-UI pack and unpack operations including file collection,
JSON serialization, compression, and encryption.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.file_utils import safe_write_json

from .constants import COMPRESSION_LEVELS
from .encryption import EncryptionManager
from .file_ops import format_size, should_exclude

logger = logging.getLogger(__name__)


class PackResult:
    """Result of a pack operation."""

    def __init__(
        self,
        *,
        success: bool,
        output_path: Path | None = None,
        total_files: int = 0,
        package_size: int = 0,
        error: str | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Initialize pack result.

        Args:
            success: Whether the operation completed successfully.
            output_path: Path to the created package file.
            total_files: Number of files packed.
            package_size: Size of the package in bytes.
            error: Fatal error message if failed.
            errors: List of per-file errors encountered.
        """
        self.success = success
        self.output_path = output_path
        self.total_files = total_files
        self.package_size = package_size
        self.error = error
        self.errors = errors or []


class UnpackResult:
    """Result of an unpack operation."""

    def __init__(
        self,
        *,
        success: bool,
        dest_path: Path | None = None,
        total_files: int = 0,
        error: str | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Initialize unpack result.

        Args:
            success: Whether the operation completed successfully.
            dest_path: Destination folder.
            total_files: Number of files extracted.
            error: Fatal error message if failed.
            errors: List of per-file errors encountered.
        """
        self.success = success
        self.dest_path = dest_path
        self.total_files = total_files
        self.error = error
        self.errors = errors or []


def collect_files(
    source_path: Path,
    exclude_patterns: set[str],
    include_git: bool = False,
    cancel_check: Any = None,
) -> list[Path]:
    """Collect files from source folder, respecting exclusions.

    Args:
        source_path: Root folder to scan.
        exclude_patterns: Set of exclusion patterns.
        include_git: Whether to include .git directories.
        cancel_check: Callable returning True if operation is cancelled.

    Returns:
        List of file paths to pack.
    """
    if not (source_path is not None):
        raise ValueError("source_path must be provided")
    files_to_pack: list[Path] = []
    for root, dirs, filenames in os.walk(source_path):
        if cancel_check and cancel_check():
            break

        dirs[:] = [
            d for d in dirs if not should_exclude(Path(root) / d, exclude_patterns, include_git)
        ]

        for filename in filenames:
            file_path = Path(root) / filename
            if not should_exclude(file_path, exclude_patterns, include_git):
                files_to_pack.append(file_path)

    return files_to_pack


def _collect_file_contents(
    source_path: Path,
    files_to_pack: list[Path],
    package_data: dict[str, Any],
    progress_callback: Any = None,
    cancel_check: Any = None,
) -> tuple[list[str], bool]:
    """Read and base64-encode files into the package data dict.

    Args:
        source_path: Root source folder.
        files_to_pack: List of files to include.
        package_data: Package dict whose 'files' key will be populated.
        progress_callback: Optional progress callable.
        cancel_check: Optional cancellation callable.

    Returns:
        Tuple of (error_messages, was_cancelled).
    """
    if not (source_path is not None):
        raise ValueError("source_path must be provided")
    errors: list[str] = []
    total_files = len(files_to_pack)

    for i, file_path in enumerate(files_to_pack):
        if cancel_check and cancel_check():
            return errors, True

        try:
            rel_path = file_path.relative_to(source_path)
            with open(file_path, "rb") as f:
                content = f.read()

            package_data["files"][str(rel_path)] = base64.b64encode(
                content,
            ).decode("utf-8")

            if progress_callback:
                progress_callback(file_path.name, i + 1, total_files)

        except (OSError, UnicodeDecodeError) as e:
            error_msg = f"Error packing {file_path}: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)

    return errors, False


def _serialize_and_write(
    package_data: dict[str, Any],
    output_path: Path,
    compression: str,
    encrypt: bool,
    password: str,
) -> None:
    """Serialize package data to JSON, compress, optionally encrypt, and write.

    Args:
        package_data: The package dict to serialize.
        output_path: Output file path.
        compression: Compression level name.
        encrypt: Whether to encrypt.
        password: Encryption password.
    """
    if not (package_data is not None):
        raise ValueError("package_data must be provided")
    json_data = json.dumps(package_data, indent=2).encode("utf-8")

    compression_level = COMPRESSION_LEVELS[compression]
    if compression_level > 0:
        json_data = gzip.compress(json_data, compresslevel=compression_level)

    if encrypt:
        json_data = EncryptionManager.encrypt_data(json_data, password)

    with open(output_path, "wb") as f:  # type: ignore[assignment]
        f.write(json_data)


def _write_manifest(
    output_path: Path,
    source_path: Path,
    files_to_pack: list[Path],
    total_files: int,
) -> None:
    """Write a sidecar manifest JSON file.

    Args:
        output_path: Path to the package file.
        source_path: Root source folder.
        files_to_pack: List of packed files.
        total_files: Total file count.
    """
    if not (output_path is not None):
        raise ValueError("output_path must be provided")
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "package_file": str(output_path),
        "created_at": datetime.now().isoformat(),
        "files": [str(f.relative_to(source_path)) for f in files_to_pack],
        "total_files": total_files,
        "package_size": output_path.stat().st_size,
    }
    safe_write_json(manifest_path, manifest, indent=2)


def pack_files(
    source_path: Path,
    output_path: Path,
    files_to_pack: list[Path],
    compression: str = "balanced",
    encrypt: bool = False,
    password: str = "",
    create_manifest: bool = True,
    progress_callback: Any = None,
    cancel_check: Any = None,
) -> PackResult:
    """Pack files into a single archive.

    Orchestrates file collection, serialization, and manifest creation.

    Args:
        source_path: Root source folder.
        output_path: Output package file path.
        files_to_pack: List of files to include.
        compression: Compression level name.
        encrypt: Whether to encrypt the package.
        password: Encryption password (required if encrypt=True).
        create_manifest: Whether to create a manifest file.
        progress_callback: Callable(current_file, index, total) for progress.
        cancel_check: Callable returning True if operation is cancelled.

    Returns:
        PackResult with operation outcome.
    """
    try:
        total_files = len(files_to_pack)
        logger.info("Packing %d files...", total_files)

        package_data: dict[str, Any] = {
            "files": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source": str(source_path),
                "total_files": total_files,
                "compression": compression,
                "encrypted": encrypt,
            },
        }

        errors, cancelled = _collect_file_contents(
            source_path,
            files_to_pack,
            package_data,
            progress_callback,
            cancel_check,
        )
        if cancelled or (cancel_check and cancel_check()):
            return PackResult(success=False, error="Operation cancelled")

        _serialize_and_write(package_data, output_path, compression, encrypt, password)

        if create_manifest:
            _write_manifest(output_path, source_path, files_to_pack, total_files)

        package_size = output_path.stat().st_size
        logger.info(
            "Package created: %s (%s)",
            output_path,
            format_size(package_size),
        )

        return PackResult(
            success=True,
            output_path=output_path,
            total_files=total_files,
            package_size=package_size,
            errors=errors,
        )

    except (PermissionError, OSError) as e:
        logger.exception("Pack operation failed")
        return PackResult(success=False, error=str(e))


def unpack_files(
    package_path: Path,
    dest_path: Path,
    encrypted: bool = False,
    password: str = "",
    progress_callback: Any = None,
    cancel_check: Any = None,
) -> UnpackResult:
    """Unpack a package archive.

    Args:
        package_path: Path to the package file.
        dest_path: Destination folder.
        encrypted: Whether the package is encrypted.
        password: Decryption password.
        progress_callback: Callable(rel_path, index, total) for progress.
        cancel_check: Callable returning True if operation is cancelled.

    Returns:
        UnpackResult with operation outcome.
    """
    try:
        dest_path.mkdir(parents=True, exist_ok=True)

        # Read package file
        with open(package_path, "rb") as f:
            data = f.read()

        # Decrypt if needed
        if encrypted:
            try:
                data = EncryptionManager.decrypt_data(data, password)
            except (ValueError, TypeError) as e:
                return UnpackResult(
                    success=False,
                    error=f"Decryption failed - incorrect password? {e}",
                )

        # Decompress if needed
        try:
            data = gzip.decompress(data)
        except (gzip.BadGzipFile, OSError):
            pass  # Not compressed or already decompressed

        # Parse JSON
        package_data = json.loads(data.decode("utf-8"))
        files = package_data.get("files", {})
        total_files = len(files)

        logger.info("Extracting %d files...", total_files)

        errors: list[str] = []

        # Extract files
        for i, (rel_path, encoded_content) in enumerate(files.items()):
            if cancel_check and cancel_check():
                return UnpackResult(success=False, error="Operation cancelled")

            try:
                file_path = dest_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)

                content = base64.b64decode(encoded_content)
                with open(file_path, "wb") as f:
                    f.write(content)

                if progress_callback:
                    progress_callback(Path(rel_path).name, i + 1, total_files)

            except (OSError, ValueError) as e:
                error_msg = f"Error extracting {rel_path}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

        if cancel_check and cancel_check():
            return UnpackResult(success=False, error="Operation cancelled")

        logger.info("Package extracted to: %s", dest_path)

        return UnpackResult(
            success=True,
            dest_path=dest_path,
            total_files=total_files,
            errors=errors,
        )

    except (PermissionError, OSError) as e:
        logger.exception("Unpack operation failed")
        return UnpackResult(success=False, error=str(e))


def inspect_package(package_path: Path) -> dict[str, Any]:
    """Inspect a package file and return its metadata.

    Args:
        package_path: Path to the package file.

    Returns:
        Dictionary with package info (size, encrypted, metadata).
    """
    with open(package_path, "rb") as f:
        data = f.read()

    info: dict[str, Any] = {
        "file": Path(package_path).name,
        "size": Path(package_path).stat().st_size,
        "size_formatted": format_size(Path(package_path).stat().st_size),
        "encrypted": False,
        "metadata": {},
    }

    try:
        decompressed = gzip.decompress(data)
        package_data = json.loads(decompressed.decode("utf-8"))
        info["metadata"] = package_data.get("metadata", {})
    except (
        gzip.BadGzipFile,
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        info["encrypted"] = True

    return info
