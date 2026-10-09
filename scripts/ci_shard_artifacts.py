"""Attempt-bound status and coverage artifact selection for CI shard gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

STATUS_SCHEMA = "tools-ci-shard-status/1"
__all__ = [
    "ShardRun",
    "ShardStatus",
    "STATUS_SCHEMA",
    "add_artifact_arguments",
    "coverage_artifact_name",
    "handle_artifact_command",
    "select_coverage_artifacts",
    "status_artifact_name",
    "verify_status_artifacts",
    "write_status_manifest",
]
_STATUS_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "run_attempt",
        "python_version",
        "shard",
        "outcome",
        "artifact_name",
        "payload_sha256",
    }
)
_OUTCOMES = frozenset({"success", "failure", "cancelled", "skipped", "not_run"})


@dataclass(frozen=True)
class ShardRun:
    """Identity of one workflow run and the latest attempt available to a gate."""

    run_id: str
    max_attempt: int
    python_version: str


@dataclass(frozen=True)
class ShardStatus:
    """One validated shard outcome bound to its actual workflow attempt."""

    run_id: str
    run_attempt: int
    python_version: str
    shard: str
    outcome: str
    artifact_name: str
    payload_sha256: str


def status_artifact_name(run: ShardRun, attempt: int, shard: str) -> str:
    """Return the unique artifact identity for one run-attempt shard record."""
    return f"shard-status-{run.run_id}-attempt-{attempt}-{run.python_version}-{shard}"


def coverage_artifact_name(run: ShardRun, attempt: int, shard: str) -> str:
    """Return the unique artifact identity for one run-attempt coverage shard."""
    return f"coverage-data-{run.run_id}-attempt-{attempt}-{run.python_version}-{shard}"


def _payload_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_status_manifest(
    output_dir: Path, run: ShardRun, shard: str, outcome: str
) -> Path:
    """Write one canonical status manifest for the current workflow attempt."""
    if not run.run_id.isdigit() or run.max_attempt < 1:
        raise ValueError("run identity must contain a numeric ID and positive attempt")
    if not shard or outcome not in _OUTCOMES:
        raise ValueError("shard and a recognized step outcome are required")
    payload: dict[str, object] = {
        "schema_version": STATUS_SCHEMA,
        "run_id": run.run_id,
        "run_attempt": run.max_attempt,
        "python_version": run.python_version,
        "shard": shard,
        "outcome": outcome,
        "artifact_name": status_artifact_name(run, run.max_attempt, shard),
    }
    payload["payload_sha256"] = _payload_digest(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run.python_version}-{shard}.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_manifest(path: Path) -> tuple[ShardStatus | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{path}: invalid status manifest ({exc})"
    if not isinstance(payload, dict) or set(payload) != _STATUS_KEYS:
        return None, f"{path}: status fields do not match {STATUS_SCHEMA}"
    if payload["schema_version"] != STATUS_SCHEMA:
        return None, f"{path}: unsupported status schema"
    if (
        not isinstance(payload["run_id"], str)
        or not payload["run_id"].isdigit()
        or type(payload["run_attempt"]) is not int
        or payload["run_attempt"] < 1
        or not isinstance(payload["python_version"], str)
        or not isinstance(payload["shard"], str)
        or not isinstance(payload["outcome"], str)
        or payload["outcome"] not in _OUTCOMES
        or not isinstance(payload["artifact_name"], str)
        or not isinstance(payload["payload_sha256"], str)
    ):
        return None, f"{path}: invalid status identity or outcome"
    digest = payload["payload_sha256"]
    digest_payload = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    if len(digest) != 64 or digest != _payload_digest(digest_payload):
        return None, f"{path}: status payload digest mismatch"
    status = ShardStatus(
        run_id=payload["run_id"],
        run_attempt=payload["run_attempt"],
        python_version=payload["python_version"],
        shard=payload["shard"],
        outcome=payload["outcome"],
        artifact_name=payload["artifact_name"],
        payload_sha256=digest,
    )
    if path.parent.name != status.artifact_name:
        return None, f"{path}: artifact directory does not match declared identity"
    return status, None


def _load_latest_statuses(
    status_dir: Path, run: ShardRun, required_shards: tuple[str, ...]
) -> tuple[dict[str, ShardStatus], list[str]]:
    problems: list[str] = []
    candidates: dict[str, dict[int, ShardStatus]] = {}
    for path in sorted(status_dir.rglob("*.json")):
        status, error = _parse_manifest(path)
        if error:
            problems.append(error)
            continue
        assert status is not None
        if status.run_id != run.run_id:
            problems.append(f"{path}: run_id mismatch")
            continue
        if status.python_version != run.python_version:
            problems.append(f"{path}: Python lane mismatch")
            continue
        if status.shard not in required_shards:
            problems.append(f"{path}: unexpected shard {status.shard}")
            continue
        if status.run_attempt > run.max_attempt:
            problems.append(f"{path}: artifact is from a future run attempt")
            continue
        expected_name = status_artifact_name(run, status.run_attempt, status.shard)
        if status.artifact_name != expected_name:
            problems.append(f"{path}: artifact name does not match shard identity")
            continue
        attempts = candidates.setdefault(status.shard, {})
        if status.run_attempt in attempts:
            problems.append(
                f"{status.shard}: ambiguous duplicate for run attempt "
                f"{status.run_attempt}"
            )
            continue
        attempts[status.run_attempt] = status

    latest: dict[str, ShardStatus] = {}
    for shard in required_shards:
        attempts = candidates.get(shard, {})
        if not attempts:
            problems.append(f"{shard}: no status recorded (shard did not run?)")
            continue
        latest_attempt = max(attempts)
        status = attempts[latest_attempt]
        latest[shard] = status
        if status.outcome != "success":
            problems.append(
                f"{shard}: {status.outcome} (latest run attempt {latest_attempt})"
            )
    return latest, problems


def verify_status_artifacts(
    status_dir: Path, run: ShardRun, required_shards: tuple[str, ...]
) -> list[str]:
    """Require one valid, successful latest-attempt status for every shard."""
    if not status_dir.is_dir():
        return ["status artifact directory is missing"]
    if run.max_attempt < 1 or not run.run_id.isdigit():
        return ["run identity must contain a numeric ID and positive attempt"]
    _, problems = _load_latest_statuses(status_dir, run, required_shards)
    return problems


def select_coverage_artifacts(
    status_dir: Path,
    coverage_dir: Path,
    output_dir: Path,
    run: ShardRun,
    required_shards: tuple[str, ...],
    expected_files: Mapping[str, tuple[str, ...]],
) -> list[str]:
    """Copy only coverage files paired with each shard's selected status attempt."""
    latest, problems = _load_latest_statuses(status_dir, run, required_shards)
    if problems:
        return problems
    if not coverage_dir.is_dir():
        return ["coverage artifact directory is missing"]
    if set(expected_files) != set(required_shards):
        return ["expected coverage files must name every required shard exactly once"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for shard in required_shards:
        status = latest[shard]
        artifact = coverage_dir / coverage_artifact_name(run, status.run_attempt, shard)
        filenames = expected_files[shard]
        if not _valid_coverage_filenames(filenames):
            problems.append(f"{shard}: expected coverage filenames are invalid")
            continue
        if not artifact.is_dir():
            problems.append(f"{shard}: selected-attempt coverage artifact is missing")
            continue
        sources = {path.name: path for path in artifact.iterdir() if path.is_file()}
        expected = set(filenames)
        if set(sources) != expected:
            problems.append(
                f"{shard}: expected {len(expected)} coverage files, found "
                f"{len(set(sources) & expected)}"
            )
            continue
        for filename in filenames:
            shutil.copyfile(sources[filename], output_dir / filename)
    return problems


def _valid_coverage_filenames(filenames: tuple[str, ...]) -> bool:
    return (
        bool(filenames)
        and all(
            name.startswith(".coverage.py")
            and "/" not in name
            and "\\" not in name
            and Path(name).name == name
            for name in filenames
        )
        and len(set(filenames)) == len(filenames)
    )


def add_artifact_arguments(
    parser: argparse.ArgumentParser, action_group: argparse._MutuallyExclusiveGroup
) -> None:
    """Add the shared status/coverage artifact CLI options to the shard command."""
    action_group.add_argument(
        "--verify-status",
        metavar="DIR",
        help="fail unless every shard has a valid successful latest-attempt record",
    )
    action_group.add_argument(
        "--record-status",
        metavar="DIR",
        help="write the current run-attempt status record into DIR",
    )
    action_group.add_argument(
        "--select-coverage",
        metavar="DIR",
        help="copy coverage artifacts matching the latest shard attempts in DIR",
    )
    parser.add_argument(
        "--python-version", default="", help="Python lane used by artifact operations"
    )
    parser.add_argument("--run-id", default="", help="GitHub Actions workflow run ID")
    parser.add_argument(
        "--run-attempt", type=int, help="GitHub Actions workflow run attempt"
    )
    parser.add_argument("--shard", default="", help="matrix shard label")
    parser.add_argument("--outcome", default="", help="test-step outcome to record")
    parser.add_argument(
        "--status-dir", type=Path, help="downloaded shard-status artifact root"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="selected coverage output directory"
    )
    parser.add_argument(
        "--matrix-result",
        default="success",
        help="aggregate tests matrix result; verification requires success",
    )


def _run_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> ShardRun:
    if not args.run_id or args.run_attempt is None or not args.python_version:
        parser.error("run ID, run attempt, and Python version are required")
    if not args.run_id.isdigit() or args.run_attempt < 1:
        parser.error("run ID must be numeric and run attempt must be positive")
    return ShardRun(args.run_id, args.run_attempt, args.python_version)


def handle_artifact_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    required_shards: tuple[str, ...],
    expected_coverage_files: Mapping[str, tuple[str, ...]],
) -> int | None:
    """Run one artifact command, returning ``None`` for non-artifact modes."""
    if args.verify_status:
        if args.matrix_result != "success":
            print(
                f"Tests matrix result was {args.matrix_result}, expected success.",
                file=sys.stderr,
            )
            return 1
        run = _run_from_args(args, parser)
        problems = verify_status_artifacts(
            Path(args.verify_status), run, required_shards
        )
        if problems:
            print(f"Shards failed for Python {args.python_version}:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(
            f"All {len(required_shards)} shards passed for Python {args.python_version}."
        )
        return 0
    if args.record_status:
        if not args.shard or not args.outcome:
            parser.error("--record-status requires --shard and --outcome")
        run = _run_from_args(args, parser)
        write_status_manifest(Path(args.record_status), run, args.shard, args.outcome)
        return 0
    if args.select_coverage:
        if not args.status_dir or not args.output_dir:
            parser.error("--select-coverage requires --status-dir and --output-dir")
        run = _run_from_args(args, parser)
        problems = select_coverage_artifacts(
            args.status_dir,
            Path(args.select_coverage),
            args.output_dir,
            run,
            required_shards,
            expected_coverage_files,
        )
        if problems:
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        return 0
    return None
