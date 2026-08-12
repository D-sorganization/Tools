"""Locked transactional persistence for regional-ground authority jobs."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from filelock import FileLock, Timeout

from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobSnapshot,
    AuthorityJobStatus,
    regional_ground_authority_job_status_from_json,
    regional_ground_authority_job_status_to_json,
)
from rate_of_closure.application.regional_ground_execution_job import (
    MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
    RegionalGroundExecutionJob,
    regional_ground_execution_job_from_json,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
    RegionalGroundExecutionResult,
    regional_ground_execution_result_from_json,
    regional_ground_execution_result_to_json,
)

AUTHORITY_JOB_STORE_SCHEMA_VERSION: Final = (
    "rate-of-closure/regional-ground-authority-store/v1"
)
MAX_AUTHORITY_JOB_STORE_BYTES: Final = 201_326_592
_PAGE_SIZE: Final = 4_096
_MAX_PAGES: Final = MAX_AUTHORITY_JOB_STORE_BYTES // _PAGE_SIZE
_APPLICATION_ID: Final = 0x524F4341
_USER_VERSION: Final = 1
_CREATE = """
CREATE TABLE IF NOT EXISTS store_metadata (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  schema_version TEXT NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS retained_jobs (
  ordinal INTEGER NOT NULL UNIQUE CHECK (ordinal >= 0),
  job_id TEXT PRIMARY KEY,
  job_json TEXT NOT NULL,
  job_digest TEXT NOT NULL,
  status_json TEXT NOT NULL,
  status_digest TEXT NOT NULL,
  result_json TEXT,
  result_digest TEXT,
  CHECK ((result_json IS NULL) = (result_digest IS NULL))
) STRICT;
"""


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _bounded(source: object, maximum: int, name: str) -> str:
    if type(source) is not str:
        raise ValueError(f"stored {name} must be text")
    try:
        size = len(source.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"stored {name} must be valid UTF-8") from exc
    if not 0 < size <= maximum:
        raise ValueError(f"stored {name} exceeds its byte bound")
    return source


@dataclass(frozen=True, slots=True)
class RetainedAuthorityJob:
    """One exact retained job, lifecycle snapshot, and optional complete result."""

    job: RegionalGroundExecutionJob
    status: AuthorityJobSnapshot
    result: RegionalGroundExecutionResult | None

    def __post_init__(self) -> None:
        if type(self.job) is not RegionalGroundExecutionJob:
            raise TypeError("retained job must be an exact execution job")
        self.job.__post_init__()
        if type(self.status) is not AuthorityJobSnapshot:
            raise TypeError("retained status must be an exact authority snapshot")
        self.status.__post_init__()
        if self.status.job_id != self.job.job_id:
            raise ValueError("retained status job_id must match the job")
        if self.status.job_sha256 != self.job.job_sha256:
            raise ValueError("retained status job_sha256 must match the job")
        if self.status.total != self.job.execution_options.max_trials:
            raise ValueError("retained status total must match the job")
        succeeded = self.status.status is AuthorityJobStatus.SUCCEEDED
        if succeeded != (self.result is not None):
            raise ValueError("retained result is required only for succeeded status")
        if self.result is not None:
            if type(self.result) is not RegionalGroundExecutionResult:
                raise TypeError("retained result must be an exact execution result")
            self.result.assert_matches_job(self.job)


class AuthorityJobStore:
    """Own one SQLite store and its process-lifetime advisory lock."""

    def __init__(self, path: Path, *, max_retained_jobs: int) -> None:
        if not isinstance(path, Path) or not path.name:
            raise TypeError("authority store path must be a named Path")
        if not path.parent.is_dir():
            raise FileNotFoundError("authority store parent directory does not exist")
        lock_path = Path(str(path) + ".lock")
        if path.is_symlink() or lock_path.is_symlink():
            raise ValueError("authority store paths must not be symbolic links")
        if type(max_retained_jobs) is not int or not 1 <= max_retained_jobs <= 16:
            raise ValueError(
                "authority store retention lies outside the supported bound"
            )
        self._path = path
        self._max_retained_jobs = max_retained_jobs
        self._lock = FileLock(lock_path)
        self._connection: sqlite3.Connection | None = None
        try:
            if os.name != "nt":
                path.parent.chmod(0o700)
            if path.exists() and path.stat().st_size > MAX_AUTHORITY_JOB_STORE_BYTES:
                raise ValueError("authority state store exceeds its file-size bound")
            self._lock.acquire(timeout=0)
            self._connection = sqlite3.connect(path, check_same_thread=False)
            self._configure()
            self._initialize()
            self._validate_database()
            self._harden_files()
        except Timeout as exc:
            self.close()
            raise RuntimeError("authority state store is already owned") from exc
        except sqlite3.Error as exc:
            self.close()
            raise ValueError("authority state store is corrupt or unsupported") from exc
        except Exception:
            self.close()
            raise

    @property
    def max_retained_jobs(self) -> int:
        return self._max_retained_jobs

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
        if self._lock.is_locked:
            self._lock.release()

    def load(self) -> tuple[RetainedAuthorityJob, ...]:
        connection = self._require_connection()
        self._validate_database()
        rows = connection.execute(
            "SELECT job_id, job_json, job_digest, status_json, status_digest, "
            "result_json, result_digest FROM retained_jobs ORDER BY ordinal"
        ).fetchall()
        if len(rows) > self._max_retained_jobs + 1:
            raise ValueError("authority store exceeds the retained-job bound")
        records = tuple(self._parse_row(row) for row in rows)
        self._validate_collection(records)
        return records

    def replace(self, records: Sequence[RetainedAuthorityJob]) -> None:
        retained = tuple(records)
        self._validate_collection(retained)
        connection = self._require_connection()
        rows = [self._row(index, record) for index, record in enumerate(retained)]
        try:
            with connection:
                self._delete_missing(connection, retained)
                connection.executemany(
                    "INSERT INTO retained_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(job_id) DO UPDATE SET ordinal=excluded.ordinal, "
                    "job_json=excluded.job_json, job_digest=excluded.job_digest, "
                    "status_json=excluded.status_json, "
                    "status_digest=excluded.status_digest, "
                    "result_json=excluded.result_json, "
                    "result_digest=excluded.result_digest "
                    "WHERE retained_jobs.ordinal != excluded.ordinal OR "
                    "retained_jobs.job_digest != excluded.job_digest OR "
                    "retained_jobs.status_digest != excluded.status_digest OR "
                    "retained_jobs.result_digest IS NOT excluded.result_digest",
                    rows,
                )
            self._harden_files()
        except sqlite3.Error as exc:
            raise RuntimeError("authority state transaction failed") from exc

    def _harden_files(self) -> None:
        if os.name == "nt":
            return
        for candidate in (
            self._path,
            Path(str(self._path) + "-wal"),
            Path(str(self._path) + "-shm"),
            Path(str(self._path) + ".lock"),
        ):
            if candidate.exists():
                candidate.chmod(0o600)

    @staticmethod
    def _delete_missing(
        connection: sqlite3.Connection,
        retained: tuple[RetainedAuthorityJob, ...],
    ) -> None:
        if not retained:
            connection.execute("DELETE FROM retained_jobs")
            return
        placeholders = ",".join("?" for _record in retained)
        connection.execute(
            f"DELETE FROM retained_jobs WHERE job_id NOT IN ({placeholders})",
            tuple(record.job.job_id for record in retained),
        )

    def _configure(self) -> None:
        connection = self._require_connection()
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        if journal_mode != ("wal",):
            raise ValueError("authority state store requires WAL journal mode")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(f"PRAGMA max_page_count = {_MAX_PAGES}")

    def _initialize(self) -> None:
        connection = self._require_connection()
        application_id = connection.execute("PRAGMA application_id").fetchone()
        user_version = connection.execute("PRAGMA user_version").fetchone()
        identity = (application_id, user_version)
        if identity not in {
            ((0,), (0,)),
            ((_APPLICATION_ID,), (_USER_VERSION,)),
        }:
            raise ValueError("unsupported authority store identity or version")
        with connection:
            connection.executescript(_CREATE)
            connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_USER_VERSION}")
            row = connection.execute(
                "SELECT schema_version FROM store_metadata WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO store_metadata VALUES (1, ?)",
                    (AUTHORITY_JOB_STORE_SCHEMA_VERSION,),
                )
            elif row[0] != AUTHORITY_JOB_STORE_SCHEMA_VERSION:
                raise ValueError("unsupported authority store schema")

    def _validate_database(self) -> None:
        connection = self._require_connection()
        try:
            row = connection.execute("PRAGMA quick_check").fetchone()
            schema = connection.execute(
                "SELECT schema_version FROM store_metadata WHERE singleton = 1"
            ).fetchone()
        except sqlite3.Error as exc:
            raise ValueError("authority state store is corrupt") from exc
        if row != ("ok",):
            raise ValueError("authority state store integrity check failed")
        if schema != (AUTHORITY_JOB_STORE_SCHEMA_VERSION,):
            raise ValueError("unsupported authority store schema")
        self._validate_schema()

    def _validate_schema(self) -> None:
        connection = self._require_connection()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
        if tables != {"store_metadata", "retained_jobs"}:
            raise ValueError("authority state store tables do not match v1")
        metadata = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(store_metadata)")
        )
        jobs = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(retained_jobs)")
        )
        if metadata != ("singleton", "schema_version") or jobs != (
            "ordinal",
            "job_id",
            "job_json",
            "job_digest",
            "status_json",
            "status_digest",
            "result_json",
            "result_digest",
        ):
            raise ValueError("authority state store columns do not match v1")

    def _parse_row(self, row: tuple[object, ...]) -> RetainedAuthorityJob:
        (
            job_id,
            job_text,
            job_hash,
            status_text,
            status_hash,
            result_text,
            result_hash,
        ) = row
        job_source = _bounded(job_text, MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES, "job")
        if job_hash != _digest(job_source):
            raise ValueError("stored job digest mismatch")
        job = regional_ground_execution_job_from_json(job_source)
        if (
            job.job_id != job_id
            or regional_ground_execution_job_to_json(job) != job_source
        ):
            raise ValueError("stored job identity or canonical bytes mismatch")
        status_source = _bounded(status_text, 4_096, "status")
        if status_hash != _digest(status_source):
            raise ValueError("stored status digest mismatch")
        status = regional_ground_authority_job_status_from_json(status_source, job)
        if regional_ground_authority_job_status_to_json(status, job) != status_source:
            raise ValueError("stored status canonical bytes mismatch")
        result = self._parse_result(result_text, result_hash, job)
        return RetainedAuthorityJob(job, status, result)

    @staticmethod
    def _parse_result(
        source: object, digest: object, job: RegionalGroundExecutionJob
    ) -> RegionalGroundExecutionResult | None:
        if source is None and digest is None:
            return None
        text = _bounded(source, MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES, "result")
        if digest != _digest(text):
            raise ValueError("stored result digest mismatch")
        result = regional_ground_execution_result_from_json(text, expected_job=job)
        if regional_ground_execution_result_to_json(result) != text:
            raise ValueError("stored result canonical bytes mismatch")
        return result

    @staticmethod
    def _row(index: int, record: RetainedAuthorityJob) -> tuple[object, ...]:
        record.__post_init__()
        job = regional_ground_execution_job_to_json(record.job)
        status = regional_ground_authority_job_status_to_json(record.status, record.job)
        result = (
            None
            if record.result is None
            else regional_ground_execution_result_to_json(record.result)
        )
        return (
            index,
            record.job.job_id,
            job,
            _digest(job),
            status,
            _digest(status),
            result,
            None if result is None else _digest(result),
        )

    def _validate_collection(self, records: tuple[RetainedAuthorityJob, ...]) -> None:
        if len(records) > self._max_retained_jobs + 1:
            raise ValueError("authority store exceeds the retained-job bound")
        identities: set[str] = set()
        active = 0
        for record in records:
            if type(record) is not RetainedAuthorityJob:
                raise TypeError("authority store records must be exact retained jobs")
            record.__post_init__()
            if record.job.job_id in identities:
                raise ValueError("authority store contains duplicate job_id values")
            identities.add(record.job.job_id)
            if record.status.status not in {
                AuthorityJobStatus.SUCCEEDED,
                AuthorityJobStatus.FAILED,
                AuthorityJobStatus.CANCELLED,
            }:
                active += 1
        if active > 1:
            raise ValueError("authority store contains multiple active jobs")

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("authority state store is closed")
        return self._connection


__all__ = [
    "AUTHORITY_JOB_STORE_SCHEMA_VERSION",
    "MAX_AUTHORITY_JOB_STORE_BYTES",
    "AuthorityJobStore",
    "RetainedAuthorityJob",
]
