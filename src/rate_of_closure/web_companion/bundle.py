"""Verified in-memory web bundle with a fixed companion runtime overlay."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from rate_of_closure.web_distribution.asset_resolver import (
    ResolvedWebAsset,
    ResolvedWebBundle,
)
from rate_of_closure.web_distribution.runtime_descriptor import (
    WEB_AUTHORITY_PATH,
    WEB_RUNTIME_ELEMENT_ID,
    WEB_RUNTIME_SCHEMA,
    parse_web_runtime_descriptor,
)

_SCRIPT_START: Final = (
    f'<script id="{WEB_RUNTIME_ELEMENT_ID}" type="application/json">'
).encode("ascii")
_SCRIPT_END: Final = b"</script>"


@dataclass(frozen=True, slots=True)
class CompanionWebBundle:
    """Exact public index overlay plus immutable declared release assets."""

    release_revision: str
    index: ResolvedWebAsset
    assets: Mapping[str, ResolvedWebAsset]

    def asset(self, path: str) -> ResolvedWebAsset:
        """Return one exact public release asset without fallback routing."""
        try:
            return self.assets[path]
        except KeyError as exc:
            raise ValueError("companion asset is not declared") from exc


def _embedded_runtime(index: bytes) -> tuple[int, int, bytes]:
    first = index.find(_SCRIPT_START)
    if first < 0 or index.find(_SCRIPT_START, first + 1) >= 0:
        raise ValueError("verified index must embed exactly one runtime descriptor")
    source_start = first + len(_SCRIPT_START)
    source_end = index.find(_SCRIPT_END, source_start)
    if source_end < 0:
        raise ValueError("verified index runtime descriptor is not closed")
    return source_start, source_end, index[source_start:source_end]


def _companion_descriptor(release_revision: str) -> bytes:
    value = {
        "schema_version": WEB_RUNTIME_SCHEMA,
        "mode": "local_companion",
        "release_revision": release_revision,
        "authority_path": WEB_AUTHORITY_PATH,
    }
    return json.dumps(value, separators=(",", ":")).encode("ascii")


def _companion_index(bundle: ResolvedWebBundle) -> ResolvedWebAsset:
    original = bundle.asset("index.html")
    start, end, source = _embedded_runtime(original.source)
    embedded = parse_web_runtime_descriptor(source)
    if (
        embedded != bundle.runtime
        or embedded.release_revision != bundle.release_revision
    ):
        raise ValueError("verified index runtime does not match its bundle")
    overlay = (
        original.source[:start]
        + _companion_descriptor(bundle.release_revision)
        + original.source[end:]
    )
    return ResolvedWebAsset(overlay, original.media_type)


def build_companion_bundle(bundle: ResolvedWebBundle) -> CompanionWebBundle:
    """Derive one non-secret companion index from a verified static bundle."""
    if (
        type(bundle) is not ResolvedWebBundle
        or bundle.runtime.mode != "static_inspection"
    ):
        raise TypeError("companion source must be an exact static-inspection bundle")
    assets = {
        path: asset
        for path, asset in bundle.assets.items()
        if path.startswith("assets/")
    }
    if not assets:
        raise ValueError("companion bundle requires declared release assets")
    return CompanionWebBundle(
        bundle.release_revision,
        _companion_index(bundle),
        MappingProxyType(assets),
    )
