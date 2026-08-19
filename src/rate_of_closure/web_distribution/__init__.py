"""Verified immutable web-release assets for Rate of Closure."""

from .asset_resolver import ResolvedWebAsset, ResolvedWebBundle, resolve_web_assets
from .package_assets import resolve_packaged_web_assets

__all__ = [
    "ResolvedWebAsset",
    "ResolvedWebBundle",
    "resolve_packaged_web_assets",
    "resolve_web_assets",
]
