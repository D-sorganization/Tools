"""Byte-equality gate for the JSON files vendored into ``web/src/vendored``.

``src/rate_of_closure/web`` must be self-contained so the public mirror
(rate-of-closure-explorer) builds standalone, but the canonical JSON files stay
owned by the monorepo. Copies of them are vendored into ``web/src/vendored``
per ``web/src/vendored/vendored_map.json``. This test (monorepo-only, so it
never ships with the mirror) makes silent forks impossible: every vendored
copy must be byte-identical to its canonical source.

If it fails, refresh the copies from the canonical files:

    node src/rate_of_closure/web/scripts/refresh-vendored.mjs
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "src" / "rate_of_closure" / "web"
VENDORED_MAP = WEB_ROOT / "src" / "vendored" / "vendored_map.json"


def _load_mappings() -> dict[str, str]:
    document = json.loads(VENDORED_MAP.read_text(encoding="utf-8"))
    mappings = document["mappings"]
    assert isinstance(mappings, dict)
    return mappings


def test_vendored_map_exists_and_is_complete() -> None:
    assert VENDORED_MAP.is_file(), f"missing vendored map: {VENDORED_MAP}"
    mappings = _load_mappings()
    assert len(mappings) >= 10, (
        "vendored map lost entries; expected the ten vendored files"
    )


def test_every_vendored_copy_matches_canonical_bytes() -> None:
    mismatches: list[str] = []
    for canonical_rel, vendored_rel in _load_mappings().items():
        canonical = REPO_ROOT / canonical_rel
        vendored = WEB_ROOT / vendored_rel
        if not canonical.is_file():
            mismatches.append(f"canonical file missing: {canonical_rel}")
            continue
        if not vendored.is_file():
            mismatches.append(f"vendored copy missing: {vendored_rel}")
            continue
        if canonical.read_bytes() != vendored.read_bytes():
            mismatches.append(f"{vendored_rel} != {canonical_rel}")
    assert not mismatches, (
        "Vendored web/ copies drifted from their canonical sources:\n  "
        + "\n  ".join(mismatches)
        + "\nRefresh with: node src/rate_of_closure/web/scripts/refresh-vendored.mjs"
    )
