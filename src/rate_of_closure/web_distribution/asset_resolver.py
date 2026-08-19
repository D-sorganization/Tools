"""Fail-closed immutable resolution of one declared web release bundle."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from .asset_manifest import (
    WEB_ASSET_MANIFEST_NAME,
    WebAssetManifest,
    WebAssetRecord,
    parse_web_asset_manifest,
)
from .runtime_descriptor import (
    WEB_RUNTIME_NAME,
    WebRuntimeDescriptor,
    parse_web_runtime_descriptor,
)

_REPARSE_POINT: Final = 0x400


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & _REPARSE_POINT)


def _assert_directory(path: Path) -> None:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or _is_link_or_reparse(metadata):
        raise ValueError(
            "web asset root and directories must not be links or reparse points"
        )


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _inventory(root: Path) -> tuple[str, ...]:
    root_identity = root.lstat()
    _assert_directory(root)
    paths: list[str] = []
    pending = [(root, _identity(root_identity))]
    while pending:
        directory, expected_identity = pending.pop()
        _assert_directory(directory)
        if _identity(directory.lstat()) != expected_identity:
            raise ValueError("web asset directory identity changed during traversal")
        for candidate in sorted(directory.iterdir(), key=lambda item: item.name):
            metadata = candidate.lstat()
            if _is_link_or_reparse(metadata):
                raise ValueError("web assets must not contain links or reparse points")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append((candidate, _identity(metadata)))
            elif stat.S_ISREG(metadata.st_mode):
                relative = candidate.relative_to(root).as_posix()
                if relative != WEB_ASSET_MANIFEST_NAME:
                    paths.append(relative)
            else:
                raise ValueError("web assets must contain only regular files")
    after = root.lstat()
    if _identity(root_identity) != _identity(after):
        raise ValueError("web asset root identity changed during traversal")
    return tuple(sorted(paths))


def _read_verified(root: Path, record: WebAssetRecord) -> bytes:
    candidate = root.joinpath(*record.path.split("/"))
    parents = [root]
    for part in record.path.split("/")[:-1]:
        parents.append(parents[-1] / part)
    parent_identities = tuple(_identity(parent.lstat()) for parent in parents)
    for parent in parents:
        _assert_directory(parent)
    before = candidate.lstat()
    if not stat.S_ISREG(before.st_mode) or _is_link_or_reparse(before):
        raise ValueError("declared web asset is not a regular non-link file")
    with candidate.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        source = stream.read(record.bytes + 1)
        after = os.fstat(stream.fileno())
    identities = {(item.st_dev, item.st_ino) for item in (before, opened, after)}
    if len(identities) != 1 or len(source) != record.bytes:
        raise ValueError("web asset changed during immutable resolution")
    if parent_identities != tuple(_identity(parent.lstat()) for parent in parents):
        raise ValueError("web asset parent changed during immutable resolution")
    if hashlib.sha256(source).hexdigest() != record.sha256:
        raise ValueError("web asset digest does not match its manifest")
    return source


@dataclass(frozen=True, slots=True)
class ResolvedWebAsset:
    """Immutable bytes and declared response media type."""

    source: bytes
    media_type: str


@dataclass(frozen=True, slots=True)
class ResolvedWebBundle:
    """Complete immutable release snapshot, safe from later path substitution."""

    release_revision: str
    runtime: WebRuntimeDescriptor
    assets: Mapping[str, ResolvedWebAsset]

    def asset(self, path: str) -> ResolvedWebAsset:
        try:
            return self.assets[path]
        except KeyError as exc:
            raise ValueError("web asset is not declared by the release") from exc


def resolve_web_assets(root: Path, manifest_source: bytes) -> ResolvedWebBundle:
    """Verify exact inventory, identity, size, and digest before returning bytes."""
    if not isinstance(root, Path) or not root.is_absolute():
        raise TypeError("web asset root must be an absolute Path")
    manifest: WebAssetManifest = parse_web_asset_manifest(manifest_source)
    inventory = _inventory(root)
    declared = tuple(asset.path for asset in manifest.assets)
    if inventory != declared:
        raise ValueError("web asset inventory does not match its manifest")
    resolved = {
        record.path: ResolvedWebAsset(_read_verified(root, record), record.media_type)
        for record in manifest.assets
    }
    try:
        runtime = parse_web_runtime_descriptor(resolved[WEB_RUNTIME_NAME].source)
    except KeyError as exc:
        raise ValueError(
            "web asset manifest must declare its runtime descriptor"
        ) from exc
    if runtime.mode != "static_inspection":
        raise ValueError("packaged web assets must use static-inspection mode")
    if runtime.release_revision != manifest.release_revision:
        raise ValueError("web runtime and asset manifest revisions differ")
    return ResolvedWebBundle(
        manifest.release_revision, runtime, MappingProxyType(resolved)
    )
