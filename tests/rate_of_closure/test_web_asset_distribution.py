"""Strict package-side contract for Rate of Closure web release assets."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from build_hooks import _summarize_dirty_status
from rate_of_closure.web_distribution.asset_manifest import (
    WEB_ASSET_MANIFEST_SCHEMA,
    parse_web_asset_manifest,
)
from rate_of_closure.web_distribution.asset_resolver import resolve_web_assets
from rate_of_closure.web_distribution.runtime_descriptor import (
    WEB_RUNTIME_NAME,
    WEB_RUNTIME_SCHEMA,
    parse_web_runtime_descriptor,
)
from scripts.check_rate_web_wheel import verify_rate_web_wheel

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_dirty_release_checkout_evidence_is_bounded_and_actionable() -> None:
    status = "\n".join(f" M path-{index}.txt" for index in range(23))

    summary = _summarize_dirty_status(status)

    assert " M path-0.txt" in summary
    assert " M path-19.txt" in summary
    assert "path-20.txt" not in summary
    assert summary.endswith("... (3 more)")


def _entry(path: str, source: bytes, media_type: str) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(source),
        "sha256": hashlib.sha256(source).hexdigest(),
        "media_type": media_type,
        "executable": False,
    }


def _manifest(entries: list[dict[str, object]], revision: str = "development") -> bytes:
    return (
        json.dumps(
            {
                "schema_version": WEB_ASSET_MANIFEST_SCHEMA,
                "release_revision": revision,
                "total_bytes": sum(int(entry["bytes"]) for entry in entries),
                "assets": entries,
            },
            indent=2,
        )
        + "\n"
    ).encode()


def _runtime(revision: str = "development") -> bytes:
    return (
        json.dumps(
            {
                "schema_version": WEB_RUNTIME_SCHEMA,
                "mode": "static_inspection",
                "release_revision": revision,
            },
            indent=2,
        )
        + "\n"
    ).encode()


def _write_runtime(root: Path, entries: list[dict[str, object]]) -> None:
    source = _runtime()
    (root / WEB_RUNTIME_NAME).write_bytes(source)
    entries.append(_entry(WEB_RUNTIME_NAME, source, "application/json; charset=utf-8"))
    entries.sort(key=lambda item: str(item["path"]))


def test_resolver_returns_an_immutable_verified_byte_snapshot(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    index = b"<main>Rate of Closure</main>\n"
    script = b"export const ready = true;\n"
    (tmp_path / "index.html").write_bytes(index)
    (assets / "app-123.js").write_bytes(script)
    entries = [
        _entry("assets/app-123.js", script, "text/javascript; charset=utf-8"),
        _entry("index.html", index, "text/html; charset=utf-8"),
    ]
    _write_runtime(tmp_path, entries)
    manifest = _manifest(entries)

    resolved = resolve_web_assets(tmp_path, manifest)
    (tmp_path / "index.html").write_bytes(b"substituted")

    assert resolved.asset("index.html").source == index
    assert resolved.asset("assets/app-123.js").source == script
    assert resolved.release_revision == "development"
    assert resolved.runtime.mode == "static_inspection"


@pytest.mark.parametrize("mutation", ["digest", "extra", "missing"])
def test_resolver_fails_closed_on_inventory_or_integrity_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = b"<main>Rate of Closure</main>\n"
    (tmp_path / "index.html").write_bytes(source)
    entry = _entry("index.html", source, "text/html; charset=utf-8")
    entries = [entry]
    _write_runtime(tmp_path, entries)
    manifest = _manifest(entries)
    if mutation == "digest":
        entry["sha256"] = "0" * 64
        manifest = _manifest(entries)
    elif mutation == "extra":
        (tmp_path / "unlisted.js").write_text("export {};\n", encoding="utf-8")
    else:
        (tmp_path / "index.html").unlink()

    with pytest.raises(ValueError, match="asset|manifest|inventory"):
        resolve_web_assets(tmp_path, manifest)


def test_manifest_rejects_duplicate_fields_and_unsafe_paths() -> None:
    source = b"{}\n"
    entry = _entry("../escape.json", source, "application/json; charset=utf-8")
    with pytest.raises(ValueError, match="path"):
        parse_web_asset_manifest(_manifest([entry]))
    duplicate = _manifest(
        [
            _entry(
                "index.html",
                source,
                "text/html; charset=utf-8",
            )
        ]
    ).replace(b'"total_bytes": 3,', b'"total_bytes": 3,\n  "total_bytes": 3,')
    with pytest.raises(ValueError, match="duplicate"):
        parse_web_asset_manifest(duplicate)


@pytest.mark.parametrize(
    "source",
    [
        _runtime().decode().encode("utf-16"),
        _runtime().decode().encode("utf-32"),
        b"\xef\xbb\xbf" + _runtime(),
    ],
)
def test_runtime_descriptor_rejects_noncanonical_text_encodings(source: bytes) -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        parse_web_runtime_descriptor(source)


def test_manifest_rejects_nonportable_names_and_non_utf8() -> None:
    for path in (
        ".hidden.json",
        "CON.json",
        "assets/caf\N{LATIN SMALL LETTER E WITH ACUTE}.json",
    ):
        with pytest.raises(ValueError, match="path"):
            parse_web_asset_manifest(
                _manifest([_entry(path, b"{}\n", "application/json; charset=utf-8")])
            )
    for source in (
        _manifest([_entry("index.html", b"x", "text/html; charset=utf-8")])
        .decode()
        .encode("utf-16"),
        b"\xef\xbb\xbf"
        + _manifest([_entry("index.html", b"x", "text/html; charset=utf-8")]),
    ):
        with pytest.raises(ValueError, match="UTF-8|BOM"):
            parse_web_asset_manifest(source)


@pytest.mark.parametrize("mutation", ["missing", "mode", "revision"])
def test_resolver_requires_one_matching_static_runtime(
    tmp_path: Path, mutation: str
) -> None:
    index = b"<main>Rate of Closure</main>\n"
    (tmp_path / "index.html").write_bytes(index)
    entries = [_entry("index.html", index, "text/html; charset=utf-8")]
    if mutation != "missing":
        value: dict[str, object] = {
            "schema_version": WEB_RUNTIME_SCHEMA,
            "mode": "static_inspection",
            "release_revision": "development",
        }
        if mutation == "mode":
            value["mode"] = "local_companion"
            value["authority_path"] = "/api/rate-of-closure/v1"
        else:
            value["release_revision"] = "0" * 40
        runtime = (json.dumps(value, indent=2) + "\n").encode()
        (tmp_path / WEB_RUNTIME_NAME).write_bytes(runtime)
        entries.append(
            _entry(
                WEB_RUNTIME_NAME,
                runtime,
                "application/json; charset=utf-8",
            )
        )
        entries.sort(key=lambda item: str(item["path"]))
    with pytest.raises(ValueError, match="runtime|static|revision"):
        resolve_web_assets(tmp_path, _manifest(entries))


def test_resolver_rejects_symbolic_assets_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "target.html"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "index.html"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links unavailable to this Windows token")
    entries = [
        _entry(
            "index.html",
            target.read_bytes(),
            "text/html; charset=utf-8",
        )
    ]
    _write_runtime(tmp_path, entries)
    manifest = _manifest(entries)
    with pytest.raises(ValueError, match="link|reparse"):
        resolve_web_assets(tmp_path, manifest)


def test_wheel_verifier_rejects_duplicate_archive_members(tmp_path: Path) -> None:
    revision = "0" * 40
    runtime = _runtime(revision)
    descriptor = json.dumps(json.loads(runtime), separators=(",", ":"))
    index = (
        '<script id="rate-of-closure-web-runtime" type="application/json">'
        f"{descriptor}</script>\n"
    ).encode()
    entries = [
        _entry(WEB_RUNTIME_NAME, runtime, "application/json; charset=utf-8"),
        _entry("index.html", index, "text/html; charset=utf-8"),
    ]
    wheel = tmp_path / "duplicate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "rate_of_closure/web/dist/rate-of-closure-assets.v1.json",
            _manifest(entries, revision),
        )
        archive.writestr(f"rate_of_closure/web/dist/{WEB_RUNTIME_NAME}", runtime)
        archive.writestr("rate_of_closure/web/dist/index.html", index)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("rate_of_closure/web/dist/index.html", index)
    with pytest.raises(ValueError, match="duplicate"):
        verify_rate_web_wheel(wheel, revision)
