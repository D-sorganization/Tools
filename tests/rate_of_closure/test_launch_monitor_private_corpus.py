"""Tests for the authorized full private-corpus desktop loader."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow", reason="pyarrow required for parquet corpus test")

from rate_of_closure.launch_monitor_private_corpus import (
    CORPUS_RELATIVE_PATH,
    load_private_corpus,
    resolve_private_corpus_path,
)

LOCAL_CAMPAIGN_ROOT = Path(
    r"c:\Users\diete\Repositories\Launch-Monitor-Flight-Model-Campaign"
)


def _authority(root: Path) -> Path:
    parquet: Path = root / CORPUS_RELATIVE_PATH
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


def test_full_private_corpus_loads_all_manifested_partitions(
    tmp_path: Path,
) -> None:
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


def test_private_corpus_fails_closed_when_sources_do_not_match(
    tmp_path: Path,
) -> None:
    parquet = _authority(tmp_path)
    manifest_path = parquet / "_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"] = {"a": {"rows": 2}, "c": {"rows": 2}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source IDs do not match"):
        load_private_corpus(parquet)


def test_private_corpus_fails_closed_on_row_limit_overflow(
    tmp_path: Path,
) -> None:
    parquet = _authority(tmp_path)
    manifest_path = parquet / "_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_rows"] = 350_000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the desktop retained-data limit"):
        load_private_corpus(parquet)


def test_private_corpus_fails_closed_on_unsupported_manifest_schema(
    tmp_path: Path,
) -> None:
    parquet = _authority(tmp_path)
    manifest_path = parquet / "_MANIFEST.json"
    manifest = {"schema_version": 999, "sources": "invalid", "total_rows": 4}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest schema is unsupported"):
        load_private_corpus(parquet)


def test_private_corpus_requires_explicit_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LAUNCH_MONITOR_DATA_ROOT", raising=False)

    with pytest.raises(ValueError, match="LAUNCH_MONITOR_DATA_ROOT"):
        resolve_private_corpus_path()


def test_private_corpus_resolves_via_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = _authority(tmp_path)
    monkeypatch.setenv("LAUNCH_MONITOR_DATA_ROOT", str(tmp_path))

    resolved = resolve_private_corpus_path()
    assert resolved == parquet.resolve()


def test_private_corpus_fails_closed_on_missing_manifest(
    tmp_path: Path,
) -> None:
    empty_dir = tmp_path / "empty_authority"
    empty_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Private corpus manifest not found"):
        resolve_private_corpus_path(empty_dir)


@pytest.mark.skipif(
    not (LOCAL_CAMPAIGN_ROOT / CORPUS_RELATIVE_PATH / "_MANIFEST.json").is_file(),
    reason="Local private campaign repository not present",
)
def test_full_governed_261k_corpus_loads_exact_campaign_partitions() -> None:
    """Verify private authority loads all 261,666 rows across 27 sources."""
    loaded = load_private_corpus(LOCAL_CAMPAIGN_ROOT)

    assert len(loaded.frame) == 261_666
    assert loaded.source_count == 27
    assert (
        loaded.manifest_sha256
        == "b45fd9100e6786d32dce229224ed901f02c20ef5c44962769faf6cc94700c299"
    )
    assert "Private Corpus (27 sources; manifest b45fd9100e67...)" in loaded.source_name
    assert "source_id" in loaded.frame.columns
    assert set(loaded.frame["source_id"].unique())
    assert loaded.frame["ball_speed_mph"].notna().sum() > 0
