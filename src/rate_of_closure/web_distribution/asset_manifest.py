"""Strict v1 manifest contract for built Rate of Closure web assets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final

WEB_ASSET_MANIFEST_SCHEMA: Final = "rate-of-closure/web-asset-manifest/v1"
WEB_ASSET_MANIFEST_NAME: Final = "rate-of-closure-assets.v1.json"
MAX_WEB_ASSET_MANIFEST_BYTES: Final = 262_144
MAX_WEB_ASSETS: Final = 128
MAX_WEB_ASSET_BYTES: Final = 16 * 1024 * 1024
MAX_WEB_BUNDLE_BYTES: Final = 64 * 1024 * 1024
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PORTABLE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
_ROOT_FIELDS = {"schema_version", "release_revision", "total_bytes", "assets"}
_ASSET_FIELDS = {"path", "bytes", "sha256", "media_type", "executable"}
_MEDIA_TYPES: Final = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ico": "image/x-icon",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ttf": "font/ttf",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate web manifest field: {key}")
        result[key] = value
    return result


def _exact(item: dict[str, object], fields: set[str], name: str) -> None:
    if set(item) != fields:
        raise ValueError(f"{name} must contain only its exact fields")


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} lies outside its integer bound")
    return value


def _release_revision(value: object) -> str:
    if type(value) is not str:
        raise ValueError("web manifest release_revision must be text")
    if value != "development" and _COMMIT.fullmatch(value) is None:
        raise ValueError("web manifest release_revision is not an exact commit")
    return value


def _asset_path(value: object) -> str:
    if type(value) is not str or not value or len(value) > 512 or "\\" in value:
        raise ValueError("web asset path is not portable relative text")
    parts = value.split("/")
    if any(
        len(part) > 128
        or _PORTABLE_COMPONENT.fullmatch(part) is None
        or _WINDOWS_RESERVED.match(part)
        for part in parts
    ):
        raise ValueError("web asset path contains an unsafe component")
    if any(part.endswith((".", " ")) for part in parts):
        raise ValueError("web asset path contains an ambiguous component")
    return value


def _expected_media_type(path: str) -> str:
    suffix = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    try:
        return _MEDIA_TYPES[suffix]
    except KeyError as exc:
        raise ValueError("web asset path has an unsupported suffix") from exc


@dataclass(frozen=True, slots=True)
class WebAssetRecord:
    """One exact declared regular file."""

    path: str
    bytes: int
    sha256: str
    media_type: str
    executable: bool


@dataclass(frozen=True, slots=True)
class WebAssetManifest:
    """One bounded complete release-asset inventory."""

    release_revision: str
    total_bytes: int
    assets: tuple[WebAssetRecord, ...]


def _parse_asset(value: object) -> WebAssetRecord:
    if type(value) is not dict:
        raise ValueError("web manifest asset must be an object")
    _exact(value, _ASSET_FIELDS, "web manifest asset")
    path = _asset_path(value["path"])
    media_type = value["media_type"]
    if media_type != _expected_media_type(path):
        raise ValueError("web asset media_type does not match its suffix")
    digest = value["sha256"]
    if type(digest) is not str or _HEX_64.fullmatch(digest) is None:
        raise ValueError("web asset sha256 must be lowercase hexadecimal")
    if value["executable"] is not False:
        raise ValueError("web assets must not be executable")
    return WebAssetRecord(
        path=path,
        bytes=_integer(value["bytes"], "web asset bytes", 1, MAX_WEB_ASSET_BYTES),
        sha256=digest,
        media_type=media_type,
        executable=False,
    )


def parse_web_asset_manifest(source: bytes) -> WebAssetManifest:
    """Parse exact bounded UTF-8 JSON with duplicate-field rejection."""
    if not source or len(source) > MAX_WEB_ASSET_MANIFEST_BYTES:
        raise ValueError("web asset manifest violates its byte bound")
    if source.startswith(b"\xef\xbb\xbf"):
        raise ValueError("web asset manifest must not contain a UTF-8 BOM")
    try:
        text = source.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "web asset manifest must be valid unique-key UTF-8 JSON"
        ) from exc
    if type(value) is not dict:
        raise ValueError("web asset manifest must be an object")
    _exact(value, _ROOT_FIELDS, "web asset manifest")
    if value["schema_version"] != WEB_ASSET_MANIFEST_SCHEMA:
        raise ValueError("unsupported web asset manifest schema")
    raw_assets = value["assets"]
    if type(raw_assets) is not list or not 1 <= len(raw_assets) <= MAX_WEB_ASSETS:
        raise ValueError("web asset manifest asset count violates its bound")
    assets = tuple(_parse_asset(item) for item in raw_assets)
    paths = tuple(asset.path for asset in assets)
    if paths != tuple(sorted(paths)) or len(set(path.lower() for path in paths)) != len(
        paths
    ):
        raise ValueError("web asset paths must be sorted and case-distinct")
    total = _integer(
        value["total_bytes"], "web manifest total_bytes", 1, MAX_WEB_BUNDLE_BYTES
    )
    if total != sum(asset.bytes for asset in assets):
        raise ValueError("web manifest total_bytes does not match its assets")
    if "index.html" not in paths:
        raise ValueError("web asset manifest must declare index.html")
    return WebAssetManifest(_release_revision(value["release_revision"]), total, assets)
