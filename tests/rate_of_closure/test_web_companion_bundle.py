"""Verified static-bundle overlay for the source production companion."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from rate_of_closure.web_companion.bundle import build_companion_bundle
from rate_of_closure.web_distribution.asset_resolver import (
    ResolvedWebAsset,
    ResolvedWebBundle,
)
from rate_of_closure.web_distribution.runtime_descriptor import (
    WEB_RUNTIME_ELEMENT_ID,
    WebRuntimeDescriptor,
)

_REVISION = "a" * 40
_STATIC_RUNTIME = {
    "schema_version": "rate-of-closure/web-runtime/v1",
    "mode": "static_inspection",
    "release_revision": _REVISION,
}
_SCRIPT_START = f'<script id="{WEB_RUNTIME_ELEMENT_ID}" type="application/json">'


def _bundle(index: bytes | None = None) -> ResolvedWebBundle:
    descriptor = json.dumps(_STATIC_RUNTIME, separators=(",", ":"))
    source = (
        index
        or (
            f"<!doctype html><html><head>{_SCRIPT_START}{descriptor}</script>"
            '</head><body><div id="root"></div></body></html>'
        ).encode()
    )
    assets = {
        "index.html": ResolvedWebAsset(source, "text/html; charset=utf-8"),
        "assets/index-AbCd_123.js": ResolvedWebAsset(
            b"export {};", "text/javascript; charset=utf-8"
        ),
    }
    return ResolvedWebBundle(
        release_revision=_REVISION,
        runtime=WebRuntimeDescriptor("static_inspection", _REVISION, None),
        assets=MappingProxyType(assets),
    )


def test_companion_bundle_overlays_only_fixed_nonsecret_runtime_data() -> None:
    bundle = build_companion_bundle(_bundle())
    index = bundle.index.source.decode("utf-8")
    start = index.index(_SCRIPT_START) + len(_SCRIPT_START)
    end = index.index("</script>", start)
    assert json.loads(index[start:end]) == {
        "schema_version": "rate-of-closure/web-runtime/v1",
        "mode": "local_companion",
        "release_revision": _REVISION,
        "authority_path": "/api/rate-of-closure/v1",
    }
    assert "Bearer" not in index
    assert "127.0.0.1" not in index
    assert bundle.asset("assets/index-AbCd_123.js").source == b"export {};"


@pytest.mark.parametrize(
    "index",
    [
        b"<html></html>",
        (
            f"{_SCRIPT_START}{json.dumps(_STATIC_RUNTIME)}</script>"
            f"{_SCRIPT_START}{json.dumps(_STATIC_RUNTIME)}</script>"
        ).encode(),
        f"{_SCRIPT_START}{{}}</script>".encode(),
    ],
)
def test_companion_bundle_rejects_noncanonical_index(index: bytes) -> None:
    with pytest.raises(ValueError):
        build_companion_bundle(_bundle(index))
