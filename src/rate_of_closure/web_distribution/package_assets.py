"""Resolve the installed Rate of Closure web distribution from package data."""

from __future__ import annotations

from pathlib import Path

from .asset_manifest import WEB_ASSET_MANIFEST_NAME
from .asset_resolver import ResolvedWebBundle, resolve_web_assets


def packaged_web_root() -> Path:
    """Return the concrete installed web root without consulting the checkout."""
    return Path(__file__).resolve().parents[1] / "web" / "dist"


def resolve_packaged_web_assets() -> ResolvedWebBundle:
    """Fail closed unless installed package data is complete and immutable."""
    root = packaged_web_root()
    manifest_path = root / WEB_ASSET_MANIFEST_NAME
    try:
        source = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("installed web asset manifest is unavailable") from exc
    return resolve_web_assets(root, source)
