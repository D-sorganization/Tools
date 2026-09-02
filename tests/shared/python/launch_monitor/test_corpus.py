"""The private-corpus loader over synthetic Parquet fixtures.

Travels with step **P19** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``) from UpstreamDrift's
``tests/unit/launch_monitor/test_corpus.py`` — but P19 is a **merge**, so the
suite it travels with is the union of two suites, not one.

Three groups of cases live here.

*UpstreamDrift's five cases* travel with one necessary change: every synthetic
checkout now writes a ``_MANIFEST.json``. That is not incidental test
maintenance, it is the merge itself. UpstreamDrift's loader accepted a corpus
with no manifest at all; the canonical loader refuses one, and
``test_merge_d30_a_corpus_with_no_manifest_is_now_refused`` pins the refusal so
the change is visible rather than implied.

*The ``rate_of_closure`` half's guarantees* arrive as the five D30 refusals
(missing manifest, unsupported ``schema_version``, a ``total_rows`` above the
desktop cap, a row-count mismatch, a source-set mismatch), the manifest digest
and the privacy-safe ``source_name`` label. These are the cases
``tests/rate_of_closure/test_launch_monitor_private_corpus.py`` asserts against
the legacy loader; here they are asserted against the canonical one, over the
same synthetic corpus shape G0.1's drift gate uses.

*The merge's own seams* are the last group: that validation runs against the
whole dataset rather than the caller's slice (otherwise selection pushdown
would silently defeat the row-count and source-set checks), that
``MAX_RETAINED_ROWS`` still equals the ``rate_of_closure`` constant it was
folded in from, and that this module does not import ``rate_of_closure`` to get
it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from rate_of_closure.launch_monitor_linked_scatter import (  # noqa: E402
    MAX_RETAINED_ROWS as LEGACY_MAX_RETAINED_ROWS,
)
from rate_of_closure.launch_monitor_private_corpus import (  # noqa: E402
    CORPUS_RELATIVE_PATH as LEGACY_CORPUS_RELATIVE_PATH,
)
from rate_of_closure.launch_monitor_private_corpus import (  # noqa: E402
    PRIVATE_DATA_ENV as LEGACY_PRIVATE_DATA_ENV,
)
from shared.python.launch_monitor.corpus import (  # noqa: E402
    CORPUS_RELATIVE_PATH,
    MANIFEST_FILENAME,
    MAX_RETAINED_ROWS,
    PRIVATE_DATA_ENV,
    corpus_dataset_path,
    load_private_corpus,
    load_private_corpus_with_provenance,
    read_corpus_manifest,
    resolve_private_corpus_path,
)

pytestmark = pytest.mark.unit

EXPECTED_DESKTOP_ROW_CAP = 300_000


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "monitor": ["TrackMan", "FlightScope Mevo+"],
            "file": ["a.csv", "b.csv"],
            "row_index": [0, 0],
            "club": ["Driver", "7 Iron"],
            "club_speed_mph": [100.0, 80.0],
            "ball_speed_mph": [150.0, 110.0],
            "smash_factor": [1.5, 1.375],
            "launch_angle_deg": [12.0, 18.0],
            "launch_direction_deg": [1.0, -0.5],
            "spin_rate_rpm": [2700.0, 6500.0],
            "back_spin_rpm": [2600.0, 6400.0],
            "side_spin_rpm": [300.0, -200.0],
            "spin_axis_deg": [4.0, -2.0],
            "attack_angle_deg": [-1.2, -4.0],
            "club_path_deg": [0.5, 1.5],
            "face_angle_deg": [0.2, 0.8],
            "carry_yd": [250.0, 165.0],
            "total_yd": [270.0, 172.0],
            "apex_native": [95.0, 28.0],
            "descent_angle_deg": [38.0, 45.0],
            "native_json": ["{}", "{}"],
        }
    )


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_corpus(
    checkout: Path,
    partitions: dict[str, pd.DataFrame],
    *,
    manifest: dict[str, Any] | None = None,
    write_manifest: bool = True,
) -> Path:
    """Materialise a hive-partitioned corpus with the manifest that governs it."""
    dataset = checkout / CORPUS_RELATIVE_PATH
    for source_id, group in partitions.items():
        partition = dataset / f"source_id={source_id}"
        partition.mkdir(parents=True, exist_ok=True)
        group.to_parquet(partition / "part-0.parquet", index=False)
    if write_manifest:
        payload = manifest
        if payload is None:
            payload = {
                "schema_version": 1,
                "sources": {
                    name: {"rows": len(group)} for name, group in partitions.items()
                },
                "total_rows": sum(len(group) for group in partitions.values()),
            }
        (dataset / MANIFEST_FILENAME).write_bytes(_manifest_bytes(payload))
    return checkout


def _synthetic_checkout(
    tmp_path: Path,
    *,
    manifest: dict[str, Any] | None = None,
    write_manifest: bool = True,
) -> Path:
    rows = _rows()
    return _write_corpus(
        tmp_path / "checkout",
        {"synthetic_trackman": rows.iloc[:1], "synthetic_mevo": rows.iloc[1:]},
        manifest=manifest,
        write_manifest=write_manifest,
    )


# ── UpstreamDrift's cases, travelling with the module ────────────────────


def test_load_private_corpus_converts_to_canonical_units(tmp_path: Path) -> None:
    frame = load_private_corpus(root=_synthetic_checkout(tmp_path))

    assert len(frame) == 2
    row = frame.set_index("session_id").loc["synthetic_trackman"]
    assert row["ball_speed"] == pytest.approx(150.0 * 0.44704)
    assert row["launch_angle"] == pytest.approx(math.radians(12.0))
    assert row["spin_rate"] == pytest.approx(2700.0 * math.pi / 30.0)
    assert row["carry_distance"] == pytest.approx(250.0 * 0.9144)
    assert row["monitor_vendor"] == "TrackMan"
    assert row["observation_kind"] == "shot"
    assert "apex_native" not in frame.columns
    assert frame["shot_id"].nunique() == 2


def test_source_and_metric_selection(tmp_path: Path) -> None:
    checkout = _synthetic_checkout(tmp_path)
    frame = load_private_corpus(
        root=checkout,
        sources=["synthetic_mevo"],
        metrics=["ball_speed", "carry_distance"],
    )
    assert set(frame["session_id"].astype(str)) == {"synthetic_mevo"}
    assert "ball_speed" in frame.columns
    assert "spin_rate" not in frame.columns
    with pytest.raises(ValueError, match="Unknown corpus sources"):
        load_private_corpus(root=checkout, sources=["nope"])
    with pytest.raises(ValueError, match="Unknown corpus metrics"):
        load_private_corpus(root=checkout, metrics=["warp_speed"])


def test_missing_root_and_missing_dataset_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PRIVATE_DATA_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="LAUNCH_MONITOR_DATA_ROOT"):
        corpus_dataset_path()
    with pytest.raises(FileNotFoundError, match="shot corpus dataset not found"):
        load_private_corpus(root=tmp_path / "empty")


def test_lateral_flight_and_capture_columns_reach_canonical_schema(
    tmp_path: Path,
) -> None:
    """The #18/#19 corpus columns convert into the canonical schema."""
    rows = pd.DataFrame(
        {
            "monitor": ["TrackMan"],
            "file": ["a.csv"],
            "row_index": [0],
            "club": ["Driver"],
            "club_speed_mph": [100.0],
            "ball_speed_mph": [150.0],
            "smash_factor": [1.5],
            "launch_angle_deg": [12.0],
            "launch_direction_deg": [1.0],
            "spin_rate_rpm": [2700.0],
            "back_spin_rpm": [2600.0],
            "side_spin_rpm": [300.0],
            "spin_axis_deg": [4.0],
            "attack_angle_deg": [-1.2],
            "club_path_deg": [0.5],
            "face_angle_deg": [0.2],
            "carry_yd": [250.0],
            "total_yd": [270.0],
            "apex_native": [95.0],
            "descent_angle_deg": [38.0],
            "lateral_carry_yd": [-12.5],
            "flight_time_s": [6.2],
            "captured_at": ["2023-08-07T00:00:00"],
            "native_json": ["{}"],
        }
    )
    checkout = _write_corpus(tmp_path / "checkout", {"synthetic_new": rows})

    frame = load_private_corpus(root=checkout)

    row = frame.iloc[0]
    assert row["lateral_carry"] == pytest.approx(-12.5 * 0.9144)  # yards -> m
    assert row["flight_time"] == pytest.approx(6.2)
    assert row["captured_at"] == "2023-08-07T00:00:00"


def test_corpus_predating_the_new_columns_still_loads(tmp_path: Path) -> None:
    """An older pinned corpus lacks the columns; the loader must not fail."""
    frame = load_private_corpus(root=_synthetic_checkout(tmp_path))

    assert len(frame) == 2
    assert "lateral_carry" not in frame.columns
    assert "captured_at" not in frame.columns


# ── the merge: D30's governance hole, closed ─────────────────────────────


def test_merge_d30_a_corpus_with_no_manifest_is_now_refused(tmp_path: Path) -> None:
    """The headline behaviour change: UpstreamDrift loaded this; canonical does not."""
    checkout = _synthetic_checkout(tmp_path, write_manifest=False)

    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_private_corpus(root=checkout)


@pytest.mark.parametrize(
    ("label", "manifest", "match"),
    [
        (
            "unsupported_schema",
            {"schema_version": 2, "sources": {}, "total_rows": 2},
            "manifest schema is unsupported",
        ),
        (
            "sources_not_a_mapping",
            {"schema_version": 1, "sources": [], "total_rows": 2},
            "manifest schema is unsupported",
        ),
        (
            "row_cap_exceeded",
            {
                "schema_version": 1,
                "sources": {"synthetic_trackman": {}, "synthetic_mevo": {}},
                "total_rows": EXPECTED_DESKTOP_ROW_CAP + 1,
            },
            "outside the desktop retained-",
        ),
        (
            "negative_row_count",
            {
                "schema_version": 1,
                "sources": {"synthetic_trackman": {}, "synthetic_mevo": {}},
                "total_rows": -1,
            },
            "outside the desktop retained-",
        ),
        (
            "row_count_mismatch",
            {
                "schema_version": 1,
                "sources": {"synthetic_trackman": {}, "synthetic_mevo": {}},
                "total_rows": 1,
            },
            "row count mismatch",
        ),
        (
            "source_set_mismatch",
            {"schema_version": 1, "sources": {"other": {}}, "total_rows": 2},
            "source IDs do not match",
        ),
    ],
)
def test_merge_d30_manifest_refusals_all_reach_the_canonical_loader(
    tmp_path: Path, label: str, manifest: dict[str, Any], match: str
) -> None:
    """Every corpus ``rate_of_closure`` refuses, the canonical layer refuses."""
    checkout = _synthetic_checkout(tmp_path / label, manifest=manifest)

    with pytest.raises(ValueError, match=match):
        load_private_corpus(root=checkout)


def test_merge_d30_provenance_matches_the_legacy_reporting_surface(
    tmp_path: Path,
) -> None:
    """The manifest digest and the desktop label are folded in unchanged."""
    checkout = _synthetic_checkout(tmp_path)
    manifest_path = checkout / CORPUS_RELATIVE_PATH / MANIFEST_FILENAME
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    corpus = load_private_corpus_with_provenance(root=checkout)

    assert corpus.manifest_sha256 == expected
    assert len(corpus.manifest_sha256) == 64
    assert corpus.source_count == 2
    assert corpus.parquet_path == checkout / CORPUS_RELATIVE_PATH
    assert corpus.source_name == (
        f"Private Corpus (2 sources; manifest {expected[:12]}...)"
    )
    # The digest is over the manifest bytes, not the data: no corpus row is
    # ever hashed into the identity the UI displays.
    assert corpus.manifest_sha256 != hashlib.sha256(b"").hexdigest()


def test_merge_d29_and_d31_are_carried_over_intact(tmp_path: Path) -> None:
    """Validation is additive: the UpstreamDrift capabilities are unchanged."""
    corpus = load_private_corpus_with_provenance(
        root=_synthetic_checkout(tmp_path),
        sources=["synthetic_mevo"],
        metrics=["carry_distance"],
    )

    # D31: selection pushdown survives the gate.
    assert len(corpus.frame) == 1
    assert set(corpus.frame["session_id"]) == {"synthetic_mevo"}
    assert "carry_distance" in corpus.frame.columns
    assert "ball_speed" not in corpus.frame.columns
    # D29: the ADR-0031 factor is still applied to what survives.
    assert float(corpus.frame.iloc[0]["carry_distance"]) == pytest.approx(
        165.0 * 0.9144, rel=1e-12
    )
    # ...and the provenance still describes the whole corpus, not the slice.
    assert corpus.source_count == 2


# ── the merge's own seams ────────────────────────────────────────────────


def test_validation_runs_against_the_whole_corpus_not_the_selection(
    tmp_path: Path,
) -> None:
    """Selection must not be a way to slip past the row-count/source checks.

    ``rate_of_closure`` compares ``total_rows`` against the frame it loaded,
    which is always the whole corpus because it has no selection. Under
    UpstreamDrift's pushdown that basis would make the check vacuous, so the
    canonical loader counts the unfiltered dataset instead. Requesting exactly
    the one partition a bad manifest describes must still be refused.
    """
    checkout = _synthetic_checkout(
        tmp_path,
        manifest={
            "schema_version": 1,
            "sources": {"synthetic_mevo": {}},
            "total_rows": 1,
        },
    )

    with pytest.raises(ValueError, match="row count mismatch"):
        load_private_corpus(root=checkout, sources=["synthetic_mevo"])


def test_either_the_authority_root_or_the_corpus_directory_resolves(
    tmp_path: Path,
) -> None:
    """The union of both halves' root resolution."""
    checkout = _synthetic_checkout(tmp_path)
    dataset_dir = checkout / CORPUS_RELATIVE_PATH

    assert resolve_private_corpus_path(checkout) == dataset_dir
    assert resolve_private_corpus_path(dataset_dir) == dataset_dir
    assert len(load_private_corpus(root=dataset_dir)) == 2
    # UpstreamDrift's pure path convention is unchanged and still agrees.
    assert corpus_dataset_path(checkout) == dataset_dir


def test_environment_variable_selects_the_corpus_for_the_canonical_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _synthetic_checkout(tmp_path)
    monkeypatch.setenv(PRIVATE_DATA_ENV, str(checkout))

    assert len(load_private_corpus()) == 2
    assert resolve_private_corpus_path() == checkout / CORPUS_RELATIVE_PATH


def test_manifest_reader_content_addresses_and_schema_checks(tmp_path: Path) -> None:
    dataset_dir = _synthetic_checkout(tmp_path) / CORPUS_RELATIVE_PATH

    manifest = read_corpus_manifest(dataset_dir)

    assert manifest.schema_version == 1
    assert manifest.total_rows == 2
    assert manifest.source_count == 2
    assert set(manifest.sources) == {"synthetic_trackman", "synthetic_mevo"}
    assert (
        manifest.manifest_sha256
        == hashlib.sha256((dataset_dir / MANIFEST_FILENAME).read_bytes()).hexdigest()
    )


def test_folded_in_constants_still_agree_with_the_legacy_half() -> None:
    """The seam left behind by not importing ``rate_of_closure``."""
    assert MAX_RETAINED_ROWS == LEGACY_MAX_RETAINED_ROWS == EXPECTED_DESKTOP_ROW_CAP
    assert PRIVATE_DATA_ENV == LEGACY_PRIVATE_DATA_ENV == "LAUNCH_MONITOR_DATA_ROOT"
    assert CORPUS_RELATIVE_PATH == LEGACY_CORPUS_RELATIVE_PATH
    assert CORPUS_RELATIVE_PATH == Path("data/authority/database/shot_corpus_parquet")


def test_corpus_module_does_not_import_rate_of_closure() -> None:
    """The merge folds capabilities in by value, never by dependency."""
    import ast
    from importlib.util import find_spec

    spec = find_spec("shared.python.launch_monitor.corpus")
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(name.split(".")[0] == "rate_of_closure" for name in modules)
