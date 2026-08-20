"""Tests for the authorized full private-corpus desktop loader."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_private_corpus import (
    CORPUS_RELATIVE_PATH,
    load_private_corpus,
    resolve_private_corpus_path,
)


def _authority(root: Path) -> Path:
    parquet = root / CORPUS_RELATIVE_PATH
    for source_id, speed in (("a", 100.0), ("b", 110.0)):
        partition = parquet / f"source_id={source_id}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"ball_speed_mph": [speed, speed + 1], "carry_yd": [150.0, 151.0]}
        ).to_parquet(partition / "part-0.parquet", index=False)
    manifest = {
        "schema_version": 1,
        "sources": {"a": {"rows": 2}, "b": {"rows": 2}},
        "total_rows": 4,
    }
    (parquet / "_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return parquet


def test_full_private_corpus_loads_all_manifested_partitions(tmp_path: Path) -> None:
    parquet = _authority(tmp_path)

    loaded = load_private_corpus(tmp_path)

    assert loaded.parquet_path == parquet.resolve()
    assert len(loaded.frame) == 4
    assert set(loaded.frame["source_id"].astype(str)) == {"a", "b"}
    assert loaded.source_count == 2
    assert "manifest" in loaded.source_name


def test_private_corpus_fails_closed_on_row_mismatch(tmp_path: Path) -> None:
    parquet = _authority(tmp_path)
    manifest_path = parquet / "_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_rows"] = 5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="row count mismatch"):
        load_private_corpus(parquet)


def test_private_corpus_requires_explicit_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LAUNCH_MONITOR_DATA_ROOT", raising=False)

    with pytest.raises(ValueError, match="LAUNCH_MONITOR_DATA_ROOT"):
        resolve_private_corpus_path()
