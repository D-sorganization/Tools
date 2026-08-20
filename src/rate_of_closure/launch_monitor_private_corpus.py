"""Authorized loader for the private, source-partitioned shot corpus."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rate_of_closure.launch_monitor_linked_scatter import MAX_RETAINED_ROWS

PRIVATE_DATA_ENV = "LAUNCH_MONITOR_DATA_ROOT"
CORPUS_RELATIVE_PATH = Path("data/authority/database/shot_corpus_parquet")


@dataclass(frozen=True)
class LoadedPrivateCorpus:
    """Validated full-corpus frame and immutable display provenance."""

    frame: pd.DataFrame
    parquet_path: Path
    manifest_sha256: str
    source_count: int

    @property
    def source_name(self) -> str:
        """Return a privacy-safe source label for the desktop UI."""
        return (
            f"Private Corpus ({self.source_count} sources; manifest "
            f"{self.manifest_sha256[:12]}...)"
        )


def resolve_private_corpus_path(root: Path | None = None) -> Path:
    """Resolve an explicitly selected root or the authorized environment root."""
    candidate = root
    if candidate is None:
        configured = os.environ.get(PRIVATE_DATA_ENV, "").strip()
        if not configured:
            raise ValueError(
                f"Select the private authority root or set {PRIVATE_DATA_ENV}."
            )
        candidate = Path(configured)
    resolved = candidate.expanduser().resolve()
    choices = (resolved, resolved / CORPUS_RELATIVE_PATH)
    for choice in choices:
        if (choice / "_MANIFEST.json").is_file():
            return choice
    raise FileNotFoundError(
        "Private corpus manifest not found. Select either the authority repository "
        "root or its shot_corpus_parquet directory."
    )


def _read_manifest(parquet_path: Path) -> tuple[dict[str, Any], str]:
    manifest_path = parquet_path / "_MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    payload = json.loads(manifest_bytes)
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("sources"), dict
    ):
        raise ValueError("Private corpus manifest schema is unsupported")
    return payload, hashlib.sha256(manifest_bytes).hexdigest()


def load_private_corpus(root: Path | None = None) -> LoadedPrivateCorpus:
    """Load all normalized partitions after manifest row/source validation."""
    parquet_path = resolve_private_corpus_path(root)
    manifest, manifest_sha256 = _read_manifest(parquet_path)
    expected_rows = int(manifest.get("total_rows", -1))
    if not 0 <= expected_rows <= MAX_RETAINED_ROWS:
        raise ValueError(
            "Private corpus manifest row count is outside the desktop retained-"
            f"data limit of {MAX_RETAINED_ROWS}"
        )
    frame = pd.read_parquet(parquet_path)
    sources = manifest["sources"]
    if len(frame) != expected_rows:
        raise ValueError(
            f"Private corpus row count mismatch: expected {expected_rows}, "
            f"loaded {len(frame)}"
        )
    if "source_id" not in frame:
        raise ValueError("Private corpus partitions did not expose source_id")
    observed_sources = set(frame["source_id"].astype(str).unique())
    if observed_sources != set(sources):
        raise ValueError("Private corpus source IDs do not match the manifest")
    return LoadedPrivateCorpus(
        frame=frame,
        parquet_path=parquet_path,
        manifest_sha256=manifest_sha256,
        source_count=len(sources),
    )


__all__ = [
    "CORPUS_RELATIVE_PATH",
    "PRIVATE_DATA_ENV",
    "LoadedPrivateCorpus",
    "load_private_corpus",
    "resolve_private_corpus_path",
]
