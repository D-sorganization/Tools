"""Immutable, aggregate-only dataset-reference job contract tests.

Travels with step **P20** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``) — the ladder's final row — from
UpstreamDrift's ``tests/unit/launch_monitor/test_dataset_reference_jobs.py``
plus the portable logic of ``tests/api/test_launch_monitor_dataset_jobs.py``.

What travels and what does not
------------------------------
UpstreamDrift drives every one of these cases through ``DatasetJobService``, a
worker-pool job runner in ``src/api/services/launch_monitor_dataset_jobs.py``.
That service is an app-local API concern, not one of P20's four modules, and it
does not exist in this repository — the same situation P18 met with the
covariation FastAPI router. So the orchestration does not travel and the
*claims* do, as direct calls to ``verify_dataset_reference`` and
``execute_dataset_operation``: the fail-closed codes, the aggregate-only
outputs, the sub-minimum suppression, the path-escape refusal and the bounded
result surface are all properties of these four modules, while the job
lifecycle (queued/running/completed, paging offsets, worker shutdown) is not.

``test_published_dataset_job_schema_matches_python_authority`` likewise does not
travel — it compares against UpstreamDrift's committed
``docs/api/contracts/launch-monitor-dataset-job-v1.schema.json``, which is that
repository's published API surface, and a second copy here would be a second
thing to drift. Its obligations are asserted directly against
``dataset_job_contract_json_schema()`` instead.

Every case that *does* travel asserts on the same synthetic authority checkout
UpstreamDrift builds: a real git repository, a hive-partitioned Parquet corpus,
a ``_MANIFEST.json``, an acquisition manifest, a source summary CSV and a
qualification manifest whose digests tie them together.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from shared.python.launch_monitor.dataset_reference import (  # noqa: E402
    DATASET_JOB_CONTRACT_VERSION,
    MAX_PAGE_SIZE,
    DatasetJobRequestV1,
    DatasetOperationV1,
    DatasetReferenceV1,
    DatasetUnavailableError,
    dataset_content_sha256,
    dataset_job_contract_json_schema,
    execute_dataset_operation,
    verify_dataset_reference,
)

pytestmark = pytest.mark.unit

LARGE_CORPUS_ROWS = 261_666


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _write_authority(
    root: Path, *, rows: int = LARGE_CORPUS_ROWS
) -> DatasetReferenceV1:
    dataset = root / "data/authority/database/shot_corpus_parquet"
    partition = dataset / "source_id=synthetic_trackman"
    partition.mkdir(parents=True)
    table = pa.table(
        {
            "monitor": pa.array(["TrackMan"] * rows),
            "file": pa.array(["synthetic.csv"] * rows),
            "row_index": pa.array(range(rows), type=pa.int64()),
            "club": pa.array(["7-iron"] * rows),
            "club_speed_mph": pa.array([80.0 + (index % 20) for index in range(rows)]),
            "ball_speed_mph": pa.array(
                [122.0 + 1.4 * (index % 20) for index in range(rows)]
            ),
        },
    )
    pq.write_table(table, partition / "part-0.parquet", compression="zstd")
    manifest = {
        "schema_version": 1,
        "sources": {
            "synthetic_trackman": {
                "rows": rows,
                "bytes": (partition / "part-0.parquet").stat().st_size,
                "columns": table.column_names,
            }
        },
        "total_rows": rows,
        "total_bytes": (partition / "part-0.parquet").stat().st_size,
    }
    manifest_path = dataset / "_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    database = root / "data/authority/database"
    acquisition = {
        "schema_version": 1,
        "source_count": 1,
        "parsed_row_count": rows,
        "sources": [
            {
                "source_id": "synthetic_trackman",
                "repository": "https://github.com/example/synthetic.git",
                "resolved_commit": "a" * 40,
                "parsed_rows": rows,
                "files": [{"path": "synthetic.csv", "sha256": "b" * 64, "bytes": 123}],
            }
        ],
    }
    acquisition_path = database / "acquisition_manifest.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    results = root / "results/v2"
    results.mkdir(parents=True)
    source_summary = results / "source_summary.csv"
    source_summary.write_text(
        "source_id,monitor,vendor_key,rows,redistribution_status,license_spdx\n"
        f"synthetic_trackman,TrackMan,trackman,{rows},reference_only,MIT\n",
        encoding="utf-8",
    )
    qualification = {
        "schema": "launch-monitor-data-qualification-manifest/v1",
        "source_rows": rows,
        "source_count": 1,
        "parquet_manifest_sha256": _sha256(manifest_path),
        "acquisition_manifest_sha256": _sha256(acquisition_path),
        "output_sha256": {"source_summary.csv": _sha256(source_summary)},
    }
    (results / "qualification_manifest.json").write_text(
        json.dumps(qualification), encoding="utf-8"
    )

    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Dataset Contract Test")
    _git(root, "config", "user.email", "dataset-contract@example.invalid")
    _git(
        root,
        "remote",
        "add",
        "origin",
        "https://github.com/D-sorganization/Launch-Monitor-Flight-Model-Campaign.git",
    )
    _git(root, "add", "data", "results")
    _git(root, "commit", "-qm", "synthetic authority")
    commit = _git(root, "rev-parse", "HEAD")
    return DatasetReferenceV1(
        root_id="test-authority",
        repository="D-sorganization/Launch-Monitor-Flight-Model-Campaign",
        commit=commit,
        manifest_sha256=_sha256(manifest_path),
        content_sha256=dataset_content_sha256(dataset),
        expected_row_count=rows,
    )


def _run(root: Path, reference: DatasetReferenceV1, operation: DatasetOperationV1):
    """The portable half of the job service: verify, then execute."""
    return execute_dataset_operation(
        verify_dataset_reference(root, reference), operation
    )


# ── UpstreamDrift's cases, travelling with the modules ───────────────────


def test_large_corpus_runs_by_reference_without_inline_rows(tmp_path: Path) -> None:
    reference = _write_authority(tmp_path)
    request = DatasetJobRequestV1(
        dataset=reference,
        operation=DatasetOperationV1(
            kind="correlation", metrics=["club_speed", "ball_speed"]
        ),
    )

    items = _run(tmp_path, request.dataset, request.operation)

    assert len(items) == 1
    assert items[0]["left_metric"] == "club_speed"
    assert items[0]["right_metric"] == "ball_speed"
    assert items[0]["n"] == LARGE_CORPUS_ROWS
    assert items[0]["correlation"] == pytest.approx(1.0)
    # The request that produced a quarter-million-row result carries no rows.
    assert "records" not in request.model_dump_json()
    assert str(tmp_path) not in request.model_dump_json()


def test_source_summary_joins_content_addressed_backing_without_rows(
    tmp_path: Path,
) -> None:
    reference = _write_authority(tmp_path, rows=100)

    items = _run(tmp_path, reference, DatasetOperationV1(kind="source_summary"))

    assert items == [
        {
            "source_id": "synthetic_trackman",
            "row_count": 100,
            "vendor_key": "trackman",
            "redistribution_status": "reference_only",
            "license_spdx": "MIT",
            "backing_repository": "example/synthetic",
            "backing_commit": "a" * 40,
            "backing_object_digests": [{"sha256": "b" * 64, "bytes": 123}],
        }
    ]
    serialized = json.dumps(items)
    assert "synthetic.csv" not in serialized
    assert "github.com" not in serialized
    assert str(tmp_path) not in serialized


def test_source_summary_suppresses_subminimum_sources(tmp_path: Path) -> None:
    reference = _write_authority(tmp_path, rows=9)

    items = _run(tmp_path, reference, DatasetOperationV1(kind="source_summary"))

    assert items == []


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("commit", "f" * 40, "commit_mismatch"),
        ("manifest_sha256", "f" * 64, "manifest_mismatch"),
        ("content_sha256", "f" * 64, "content_mismatch"),
        ("expected_row_count", 99, "row_count_mismatch"),
    ],
)
def test_reference_mismatch_fails_closed(
    tmp_path: Path, field: str, value: object, code: str
) -> None:
    reference = _write_authority(tmp_path, rows=100).model_copy(update={field: value})

    with pytest.raises(DatasetUnavailableError) as excinfo:
        verify_dataset_reference(tmp_path, reference)

    assert excinfo.value.state.code == code
    assert str(tmp_path) not in excinfo.value.state.message


def test_root_alias_cannot_be_used_as_a_path_escape(tmp_path: Path) -> None:
    reference = _write_authority(tmp_path, rows=100).model_copy(
        update={"root_id": "../../private"}
    )
    with pytest.raises(ValueError, match="root_id"):
        DatasetJobRequestV1(
            dataset=reference, operation=DatasetOperationV1(kind="source_summary")
        )


def test_repository_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    """The service reports ``root_not_authorized``; the model layer's own
    equivalent is the repository-identity refusal, which does travel."""
    reference = _write_authority(tmp_path, rows=100).model_copy(
        update={"repository": "someone-else/not-the-authority"}
    )

    with pytest.raises(DatasetUnavailableError) as excinfo:
        verify_dataset_reference(tmp_path, reference)

    assert excinfo.value.state.code == "repository_mismatch"
    assert str(tmp_path) not in excinfo.value.state.message


def test_client_cannot_bless_content_modified_after_the_exact_commit(
    tmp_path: Path,
) -> None:
    reference = _write_authority(tmp_path, rows=100)
    parquet_path = next(
        (tmp_path / "data/authority/database/shot_corpus_parquet").rglob("*.parquet")
    )
    parquet_path.write_bytes(parquet_path.read_bytes() + b"dirty")
    dirty_hash = dataset_content_sha256(parquet_path.parents[1])

    with pytest.raises(DatasetUnavailableError) as excinfo:
        verify_dataset_reference(
            tmp_path, reference.model_copy(update={"content_sha256": dirty_hash})
        )

    # Blessing the dirty content does not help: the tree is no longer what the
    # commit says it is, so the committed-layout check refuses first.
    assert excinfo.value.state.code in {"content_mismatch", "manifest_mismatch"}


def test_metric_summary_returns_one_aggregate_per_metric(tmp_path: Path) -> None:
    reference = _write_authority(tmp_path, rows=100)

    items = _run(
        tmp_path,
        reference,
        DatasetOperationV1(kind="metric_summary", metrics=["club_speed", "ball_speed"]),
    )

    assert len(items) == 2
    assert {item["metric"] for item in items} == {"club_speed", "ball_speed"}
    for item in items:
        assert item["n"] == 100
        assert set(item) == {
            "group_by",
            "group",
            "metric",
            "n",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
        }


# ── replacing the published-artifact comparison with direct assertions ───


def test_generated_schema_carries_the_full_job_obligation() -> None:
    """Replaces UpstreamDrift's committed-schema comparison."""
    schema = dataset_job_contract_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["contract_version"]["const"] == (
        DATASET_JOB_CONTRACT_VERSION
    )
    assert set(schema["required"]) == {"dataset", "operation"}

    reference = schema["$defs"]["DatasetReferenceV1"]
    assert reference["additionalProperties"] is False
    assert reference["properties"]["commit"]["pattern"] == r"^[0-9a-f]{40}$"
    assert reference["properties"]["manifest_sha256"]["pattern"] == r"^[0-9a-f]{64}$"
    assert reference["properties"]["content_sha256"]["pattern"] == r"^[0-9a-f]{64}$"
    assert reference["properties"]["root_id"]["pattern"] == r"^[a-z][a-z0-9-]{0,62}$"

    operation = schema["$defs"]["DatasetOperationV1"]
    assert operation["additionalProperties"] is False
    assert operation["properties"]["kind"]["enum"] == [
        "source_summary",
        "metric_summary",
        "correlation",
    ]
    assert operation["properties"]["metrics"]["maxItems"] == 12
    # There is nowhere in the request to put an observation or a query string:
    # the operation is an enum plus an allow-listed metric tuple, and no
    # property anywhere in the request accepts free text.
    assert "records" not in json.dumps(schema)
    assert all(
        "query" not in properties
        for definition in schema["$defs"].values()
        for properties in [definition.get("properties", {})]
    )
    assert set(operation["properties"]) == {
        "kind",
        "metrics",
        "group_by",
        "minimum_group_rows",
    }


# ── refusal pins the port earns ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        ({"kind": "metric_summary", "metrics": ()}, "requires at least one metric"),
        (
            {"kind": "correlation", "metrics": ("club_speed",)},
            "requires at least two metrics",
        ),
        (
            {"kind": "source_summary", "metrics": ("club_speed",)},
            "does not accept metrics or group_by",
        ),
        (
            {"kind": "metric_summary", "metrics": ("club_speed", "club_speed")},
            "must be unique",
        ),
        (
            {"kind": "metric_summary", "metrics": ("not_a_corpus_metric",)},
            "unsupported metrics",
        ),
    ],
)
def test_operation_shapes_are_refused_before_any_read(
    operation: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        DatasetOperationV1(**operation)


def test_metric_allow_list_is_the_p19_corpus_column_map() -> None:
    """The set of nameable metrics is defined by the merged corpus module."""
    from shared.python.launch_monitor.corpus import CORPUS_COLUMN_MAP

    allowed = {name for name, _ in CORPUS_COLUMN_MAP.values()}
    for metric in sorted(allowed):
        DatasetOperationV1(kind="metric_summary", metrics=(metric,))
    with pytest.raises(ValueError, match="unsupported metrics"):
        DatasetOperationV1(kind="metric_summary", metrics=("apex_native",))


def test_result_paging_bound_reaches_the_contract() -> None:
    """The bounded-page limit the job service enforces is contract-owned."""
    assert MAX_PAGE_SIZE == 200


def test_dataset_reference_modules_do_not_import_rate_of_closure() -> None:
    """The canonical layer never depends on the legacy package."""
    import ast
    from importlib.util import find_spec

    for name in (
        "dataset_reference",
        "dataset_reference_contract",
        "dataset_reference_operations",
        "dataset_reference_verification",
    ):
        spec = find_spec(f"shared.python.launch_monitor.{name}")
        assert spec is not None and spec.origin is not None
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        assert not any(item.split(".")[0] == "rate_of_closure" for item in modules), (
            name
        )
