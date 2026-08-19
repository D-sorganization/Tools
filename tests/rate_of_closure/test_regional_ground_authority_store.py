"""Durable restart-recovery contract for regional-ground authority jobs."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobFailure,
    AuthorityJobSnapshot,
    AuthorityJobStatus,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
)
from rate_of_closure.web_authority.job_store import (
    AuthorityJobStore,
    RetainedAuthorityJob,
)
from rate_of_closure.web_authority.jobs import (
    AuthorityExecutionUnavailable,
    AuthorityJobConflict,
    AuthorityJobManager,
)
from rate_of_closure.web_authority.state_security import (
    StateSecurityCode,
    StateSecurityError,
    verify_state_file,
    verify_state_root,
)
from tests.rate_of_closure.test_regional_ground_authority_jobs import _job, _result

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _snapshot(
    status: AuthorityJobStatus,
    job: RegionalGroundExecutionJob | None = None,
) -> AuthorityJobSnapshot:
    selected = _job() if job is None else job
    result_available = status is AuthorityJobStatus.SUCCEEDED
    failure = (
        AuthorityJobFailure("execution_failed", "authority_restart")
        if status is AuthorityJobStatus.FAILED
        else None
    )
    completed = selected.execution_options.max_trials if result_available else 0
    return AuthorityJobSnapshot(
        job_id=selected.job_id,
        job_sha256=selected.job_sha256,
        status=status,
        completed=completed,
        total=selected.execution_options.max_trials,
        result_available=result_available,
        failure=failure,
    )


def test_store_round_trips_only_validated_job_bound_records(tmp_path: Path) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    job = _job()
    record = RetainedAuthorityJob(
        job, _snapshot(AuthorityJobStatus.SUCCEEDED), _result(job)
    )

    store = AuthorityJobStore(path, max_retained_jobs=4)
    store.replace((record,))

    assert store.load() == (record,)
    assert b"token" not in path.read_bytes().lower()
    store.close()


def test_store_replace_removes_only_unretained_records(tmp_path: Path) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    first_job = _job("first-ground-job")
    second_job = _job("second-ground-job")
    first = RetainedAuthorityJob(
        first_job,
        _snapshot(AuthorityJobStatus.SUCCEEDED, first_job),
        _result(first_job),
    )
    second = RetainedAuthorityJob(
        second_job,
        _snapshot(AuthorityJobStatus.SUCCEEDED, second_job),
        _result(second_job),
    )
    store = AuthorityJobStore(path, max_retained_jobs=4)
    store.replace((first, second))

    store.replace((second,))

    assert store.load() == (second,)
    store.close()


def test_store_rejects_corruption_without_replacing_last_good_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    store = AuthorityJobStore(path, max_retained_jobs=4)
    job = _job()
    retained = RetainedAuthorityJob(
        job, _snapshot(AuthorityJobStatus.SUCCEEDED), _result(job)
    )
    store.replace((retained,))
    store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE retained_jobs SET job_digest = ?", ("0" * 64,))
        connection.commit()
    corrupted = AuthorityJobStore(path, max_retained_jobs=4)
    with pytest.raises(ValueError, match="digest"):
        corrupted.load()
    corrupted.close()


def test_store_rolls_back_deletion_when_upsert_fails_mid_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    store = AuthorityJobStore(path, max_retained_jobs=4)
    first_job = _job("first-ground-job")
    second_job = _job("second-ground-job")
    initial = (
        RetainedAuthorityJob(
            first_job,
            _snapshot(AuthorityJobStatus.SUCCEEDED, first_job),
            _result(first_job),
        ),
        RetainedAuthorityJob(
            second_job,
            _snapshot(AuthorityJobStatus.SUCCEEDED, second_job),
            _result(second_job),
        ),
    )
    store.replace(initial)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_retained_update BEFORE UPDATE ON retained_jobs "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )

    replacement = RetainedAuthorityJob(
        second_job,
        _snapshot(AuthorityJobStatus.FAILED, second_job),
        None,
    )
    with pytest.raises(RuntimeError, match="transaction"):
        store.replace((replacement,))

    assert store.load() == initial
    store.close()


def test_store_rejects_result_substitution_and_impossible_state(
    tmp_path: Path,
) -> None:
    store = AuthorityJobStore(
        tmp_path / "authority-jobs.v1.sqlite3", max_retained_jobs=4
    )
    job = _job()
    wrong_result = _result(_job("different-ground-job"))

    with pytest.raises(ValueError, match="job"):
        RetainedAuthorityJob(job, _snapshot(AuthorityJobStatus.SUCCEEDED), wrong_result)
    with pytest.raises(ValueError, match="result"):
        RetainedAuthorityJob(job, _snapshot(AuthorityJobStatus.RUNNING), _result(job))
    store.close()


def test_manager_recovers_complete_result_and_rejects_duplicate_submission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    first = AuthorityJobManager(
        runner=lambda job, _hooks: _result(job),
        store=AuthorityJobStore(path, max_retained_jobs=4),
    )
    job = _job()
    first.submit(job)
    first.wait_for_terminal(job.job_id, timeout_s=2.0)
    first.close()

    recovered = AuthorityJobManager(
        runner=lambda job, _hooks: _result(job),
        store=AuthorityJobStore(path, max_retained_jobs=4),
    )

    assert recovered.status(job.job_id).status is AuthorityJobStatus.SUCCEEDED
    assert recovered.result(job.job_id) == _result(job)
    with pytest.raises(AuthorityJobConflict, match="retained"):
        recovered.submit(job)
    recovered.close()


@pytest.mark.parametrize(
    "interrupted",
    [
        AuthorityJobStatus.QUEUED,
        AuthorityJobStatus.RUNNING,
    ],
)
def test_manager_marks_interrupted_records_failed_without_resuming_physics(
    tmp_path: Path,
    interrupted: AuthorityJobStatus,
) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    job = _job()
    store = AuthorityJobStore(path, max_retained_jobs=4)
    store.replace((RetainedAuthorityJob(job, _snapshot(interrupted), None),))
    ran = threading.Event()

    manager = AuthorityJobManager(
        runner=lambda job, _hooks: (ran.set(), _result(job))[1],
        store=store,
    )

    recovered = manager.status(job.job_id)
    assert recovered.status is AuthorityJobStatus.FAILED
    assert recovered.failure == AuthorityJobFailure(
        "execution_failed", "authority_restart"
    )
    assert recovered.result_available is False
    assert ran.is_set() is False
    assert store.load()[0].status.status is AuthorityJobStatus.FAILED
    manager.close()


def test_manager_recovers_cancel_requested_as_cancelled_without_runner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    job = _job()
    store = AuthorityJobStore(path, max_retained_jobs=4)
    store.replace(
        (
            RetainedAuthorityJob(
                job, _snapshot(AuthorityJobStatus.CANCEL_REQUESTED), None
            ),
        )
    )
    ran = threading.Event()
    manager = AuthorityJobManager(
        runner=lambda job, _hooks: (ran.set(), _result(job))[1], store=store
    )

    assert manager.status(job.job_id).status is AuthorityJobStatus.CANCELLED
    assert ran.is_set() is False
    manager.close()


def test_corrupt_store_fails_manager_construction_closed(tmp_path: Path) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    path.write_text("{not-sqlite", encoding="utf-8")

    with pytest.raises(ValueError):
        AuthorityJobManager(
            runner=lambda job, _hooks: _result(job),
            store=AuthorityJobStore(path, max_retained_jobs=4),
        )


def test_store_rejects_concurrent_process_ownership(tmp_path: Path) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    owner = AuthorityJobStore(path, max_retained_jobs=4)

    with pytest.raises(RuntimeError, match="owned"):
        AuthorityJobStore(path, max_retained_jobs=4)

    owner.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not authoritative")
def test_store_hardens_directory_database_and_sqlite_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    store = AuthorityJobStore(path, max_retained_jobs=4)

    assert path.parent.stat().st_mode & 0o777 == 0o700
    for candidate in (
        path,
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        Path(str(path) + ".lock"),
    ):
        assert candidate.stat().st_mode & 0o777 == 0o600

    store.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows security descriptors")
def test_store_hardens_exact_windows_database_sidecar_and_lock_dacls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.v1.sqlite3"
    store = AuthorityJobStore(path, max_retained_jobs=4)

    verify_state_root(tmp_path)
    for candidate in (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}.lock"),
    ):
        verify_state_file(tmp_path, candidate)

    store.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows state-root lease")
def test_store_rejects_unknown_root_entry_without_mutating_it(tmp_path: Path) -> None:
    sentinel = tmp_path / "unexpected.txt"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(StateSecurityError) as captured:
        AuthorityJobStore(tmp_path / "authority.v1.sqlite3", max_retained_jobs=4)

    assert captured.value.code is StateSecurityCode.UNEXPECTED_ENTRY
    assert sentinel.read_bytes() == b"preserve"
    assert not (tmp_path / "authority.v1.sqlite3").exists()


@pytest.mark.parametrize("linked_suffix", ["", ".lock"])
def test_store_rejects_symbolic_store_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_suffix: str,
) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    linked_path = Path(str(path) + linked_suffix)
    original_is_symlink = Path.is_symlink

    def selective_is_symlink(candidate: Path) -> bool:
        return candidate == linked_path or original_is_symlink(candidate)

    monkeypatch.setattr(Path, "is_symlink", selective_is_symlink)

    with pytest.raises(ValueError, match="symbolic"):
        AuthorityJobStore(path, max_retained_jobs=4)


def test_store_rejects_unknown_database_version_without_recreating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    store = AuthorityJobStore(path, max_retained_jobs=4)
    store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(ValueError, match="version"):
        AuthorityJobStore(path, max_retained_jobs=4)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)


def test_manager_persists_lifecycle_before_exposing_transitions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority-jobs.v1.sqlite3"
    store = AuthorityJobStore(path, max_retained_jobs=4)
    entered = threading.Event()
    progressed = threading.Event()
    release = threading.Event()

    def controlled_runner(
        job: RegionalGroundExecutionJob, hooks: GroundRegionalVariationHooks
    ) -> RegionalGroundExecutionResult:
        entered.set()
        callback = hooks.progress_callback
        assert callback is not None
        callback(GroundRegionalVariationProgress(1, 4))
        progressed.set()
        assert release.wait(2.0)
        return _result(job)

    manager = AuthorityJobManager(runner=controlled_runner, store=store)
    submitted = manager.submit(_job())
    assert submitted.status is AuthorityJobStatus.QUEUED
    assert entered.wait(2.0)
    assert progressed.wait(2.0)
    persisted = store.load()[0].status
    assert persisted.status is AuthorityJobStatus.RUNNING
    assert persisted.completed == 1

    cancelled = manager.cancel(submitted.job_id)
    assert cancelled.status is AuthorityJobStatus.CANCEL_REQUESTED
    assert store.load()[0].status.status is AuthorityJobStatus.CANCEL_REQUESTED
    release.set()
    manager.wait_for_terminal(submitted.job_id, timeout_s=2.0)
    manager.close()


def test_acceptance_transaction_failure_prevents_worker_and_disables_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AuthorityJobStore(
        tmp_path / "authority-jobs.v1.sqlite3", max_retained_jobs=4
    )
    ran = threading.Event()

    def fail_replace(_records: object) -> None:
        raise RuntimeError("injected private storage failure")

    monkeypatch.setattr(store, "replace", fail_replace)
    manager = AuthorityJobManager(
        runner=lambda job, _hooks: (ran.set(), _result(job))[1], store=store
    )

    with pytest.raises(AuthorityExecutionUnavailable, match="durable job acceptance"):
        manager.submit(_job())

    assert ran.is_set() is False
    assert manager.retained_job_count == 0
    assert manager.execution_available is False
    manager.close()
