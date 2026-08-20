"""Verify an exact built wheel contains one closed Rate of Closure web bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from rate_of_closure.web_distribution.asset_manifest import (
    WEB_ASSET_MANIFEST_NAME,
    parse_web_asset_manifest,
)
from rate_of_closure.web_distribution.runtime_descriptor import (
    WEB_RUNTIME_ELEMENT_ID,
    WEB_RUNTIME_NAME,
    parse_web_runtime_descriptor,
)

_PREFIX = "rate_of_closure/web/dist/"
_FORBIDDEN_TEXT = (
    "sourceMappingURL",
    "C:/Users/",
    "C:\\Users\\",
    "127.0.0.1",
    "localhost",
)


def _member_bytes(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.is_dir() or info.file_size <= 0:
        raise ValueError(f"wheel web member is not a non-empty file: {name}")
    return archive.read(info)


def verify_rate_web_wheel(wheel: Path, expected_revision: str) -> int:
    """Verify inventory, revision, bytes, and release-only content in one wheel."""
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("an exact wheel path is required")
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith(_PREFIX) and not name.endswith("/")
        ]
        if len(members) != len(set(members)):
            raise ValueError("wheel contains duplicate web members")
        actual = {name.removeprefix(_PREFIX) for name in members}
        manifest_source = _member_bytes(archive, _PREFIX + WEB_ASSET_MANIFEST_NAME)
        manifest = parse_web_asset_manifest(manifest_source)
        declared = {record.path for record in manifest.assets}
        if actual != declared | {WEB_ASSET_MANIFEST_NAME}:
            raise ValueError("wheel web inventory is not exactly manifest-bound")
        if (
            manifest.release_revision != expected_revision
            or expected_revision == "development"
        ):
            raise ValueError("wheel web revision is not the required release commit")
        for record in manifest.assets:
            source = _member_bytes(archive, _PREFIX + record.path)
            if (
                len(source) != record.bytes
                or hashlib.sha256(source).hexdigest() != record.sha256
            ):
                raise ValueError(f"wheel web asset failed integrity: {record.path}")
            if record.path.endswith((".map", ".ts", ".tsx")):
                raise ValueError("wheel contains a source or source-map web asset")
            if record.media_type.startswith(("text/", "application/json")):
                text = source.decode("utf-8", errors="strict")
                if any(item in text for item in _FORBIDDEN_TEXT):
                    raise ValueError(
                        f"wheel web asset leaks development transport data: {record.path}"
                    )
        runtime_source = _member_bytes(archive, _PREFIX + WEB_RUNTIME_NAME)
        runtime = parse_web_runtime_descriptor(runtime_source)
        if (
            runtime.mode != "static_inspection"
            or runtime.release_revision != expected_revision
        ):
            raise ValueError("wheel runtime is not bound to static release identity")
        embedded = json.dumps(json.loads(runtime_source), separators=(",", ":"))
        index = _member_bytes(archive, _PREFIX + "index.html").decode("utf-8")
        marker = f'<script id="{WEB_RUNTIME_ELEMENT_ID}" type="application/json">{embedded}</script>'
        if index.count(marker) != 1:
            raise ValueError("wheel index is not bound to its exact runtime descriptor")
    return len(declared)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    arguments = parser.parse_args()
    count = verify_rate_web_wheel(
        arguments.wheel.resolve(), arguments.expected_revision
    )
    print(f"verified wheel web distribution: {count} declared assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
