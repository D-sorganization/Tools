"""Qualify a clean installed wheel's Windows authority-state boundary."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import rate_of_closure
from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobSnapshot,
    AuthorityJobStatus,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
    build_regional_ground_execution_job,
    regional_ground_execution_job_from_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
    regional_ground_execution_result_from_json,
)
from rate_of_closure.web_authority.job_store import (
    AuthorityJobStore,
    RetainedAuthorityJob,
)
from rate_of_closure.web_authority.jobs import AuthorityJobManager
from rate_of_closure.web_authority.state_security import (
    verify_state_file,
    verify_state_root,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("seed", "recover"), required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--job-fixture", type=Path, required=True)
    parser.add_argument("--result-fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _assert_installed_module() -> None:
    module = Path(rate_of_closure.__file__).resolve()
    if not module.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("qualification imported a non-installed source tree")


def _fixture(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload[field]
    if type(selected) is not dict:
        raise RuntimeError("qualification fixture has an invalid root")
    return selected


def _records(arguments: argparse.Namespace) -> tuple[RetainedAuthorityJob, ...]:
    job_payload = _fixture(arguments.job_fixture, "job")
    job = regional_ground_execution_job_from_json(
        json.dumps(job_payload, ensure_ascii=False, separators=(",", ":"))
    )
    result_payload = _fixture(arguments.result_fixture, "result")
    result = regional_ground_execution_result_from_json(
        json.dumps(result_payload, ensure_ascii=False, separators=(",", ":")),
        expected_job=job,
    )
    interrupted = _clone_job(job, "installed-restart-interrupted")
    return (
        RetainedAuthorityJob(job, _snapshot(job, succeeded=True), result),
        RetainedAuthorityJob(
            interrupted,
            _snapshot(interrupted, succeeded=False),
            None,
        ),
    )


def _clone_job(
    source: RegionalGroundExecutionJob,
    job_id: str,
) -> RegionalGroundExecutionJob:
    return build_regional_ground_execution_job(
        job_id=job_id,
        launch=source.launch.launch,
        flight=source.flight,
        transfer=source.transfer,
        capture_speed_m_s=source.capture_speed_m_s,
        execution_options=source.execution_options,
        regional_execution_options=source.regional_execution_options,
        variation_request=source.variation_request,
        producer=source.provenance.producer,
        producer_version=source.provenance.producer_version,
        source_revision=source.provenance.source_revision,
    )


def _snapshot(
    job: RegionalGroundExecutionJob,
    *,
    succeeded: bool,
) -> AuthorityJobSnapshot:
    return AuthorityJobSnapshot(
        job_id=job.job_id,
        job_sha256=job.job_sha256,
        status=(
            AuthorityJobStatus.SUCCEEDED if succeeded else AuthorityJobStatus.RUNNING
        ),
        completed=(job.execution_options.max_trials if succeeded else 1),
        total=job.execution_options.max_trials,
        result_available=succeeded,
    )


def _database_snapshot(
    path: Path,
) -> tuple[
    tuple[tuple[int, ...] | None, tuple[int, ...] | None],
    set[str],
    list[tuple[object, ...]],
]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        identity = (
            connection.execute("PRAGMA application_id").fetchone(),
            connection.execute("PRAGMA user_version").fetchone(),
        )
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
        rows = connection.execute("SELECT * FROM retained_jobs").fetchall()
    if identity != ((0x524F4341,), (1,)):
        raise RuntimeError("installed authority store identity is invalid")
    if tables != {"store_metadata", "retained_jobs"}:
        raise RuntimeError("installed authority store schema is invalid")
    return (identity, tables, rows)


def _verify_live_store(root: Path, path: Path) -> None:
    verify_state_root(root)
    for candidate in (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}.lock"),
    ):
        verify_state_file(root, candidate)


def _seed(arguments: argparse.Namespace, root: Path, path: Path) -> None:
    root.mkdir(parents=False)
    store = AuthorityJobStore(path, max_retained_jobs=4)
    records = _records(arguments)
    store.replace(records)
    _verify_live_store(root, path)
    store.close()
    if len(_database_snapshot(path)[2]) != 2:
        raise RuntimeError("installed authority seed was not durable")
    arguments.report.write_text('{"seeded":true}', encoding="utf-8")


def _recover(arguments: argparse.Namespace, root: Path, path: Path) -> None:
    expected_success, expected_interrupted = _records(arguments)
    runner_invoked = False

    def forbidden_runner(*_args: object) -> RegionalGroundExecutionResult:
        nonlocal runner_invoked
        runner_invoked = True
        raise RuntimeError("recovery replayed physics")

    store = AuthorityJobStore(path, max_retained_jobs=4)
    manager = AuthorityJobManager(
        runner=forbidden_runner,
        max_retained_jobs=4,
        store=store,
    )
    _verify_live_store(root, path)
    success = manager.status(expected_success.job.job_id)
    interrupted = manager.status(expected_interrupted.job.job_id)
    if success != expected_success.status:
        raise RuntimeError("completed installed result status changed across restart")
    if manager.result(expected_success.job.job_id) != expected_success.result:
        raise RuntimeError("completed installed result changed across restart")
    if (
        interrupted.status is not AuthorityJobStatus.FAILED
        or interrupted.failure is None
    ):
        raise RuntimeError("interrupted installed job did not fail closed")
    if interrupted.failure.stage != "authority_restart" or runner_invoked:
        raise RuntimeError("interrupted installed job was replayed")
    manager.close()
    arguments.report.write_text(
        json.dumps(
            {
                "exact_private_acl": True,
                "installed_module": True,
                "interrupted_no_replay": True,
                "restart_recovery": True,
                "schema_version": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    arguments = _arguments()
    if os.name != "nt":
        raise RuntimeError("qualification requires native Windows")
    _assert_installed_module()
    root = arguments.state_root.resolve()
    path = root / "authority.v1.sqlite3"
    if arguments.phase == "seed":
        _seed(arguments, root, path)
    else:
        _recover(arguments, root, path)


if __name__ == "__main__":
    main()
