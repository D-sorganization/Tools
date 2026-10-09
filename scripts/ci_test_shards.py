#!/usr/bin/env python3
"""Partition the whole Tools test tree into CI shards (Tools #4913).

Every test under ``tests/`` and ``src/``, including embedded apps, belongs to one
named shard in the ``ci-standard.yml`` matrix.

Design rules:

* Every shard is a list of pytest invocations. Suites that ship their own
  ``pyproject.toml`` ``[tool.pytest.ini_options]`` (the embedded sub-apps) run
  as their own invocation so pytest picks up their rootdir/markers/conftest,
  exactly as a developer running ``pytest src/<app>/tests`` gets.
* Catch-all shards (``tests-rest``, ``src-rest``) use ``--ignore`` for the
  directories other shards own, so a new test directory is collected by the
  next CI run without anyone editing this file.
* ``--check`` proves the partition: every test file is claimed by exactly one
  shard, and every quarantined path still exists.
* Quarantine (``config/test_quarantine.json``) is the only sanctioned way to
  keep a test module out of the PR lane; each entry names an owner and a
  tracked issue. Directory exclusions are not allowed.

Usage::

    python scripts/ci_test_shards.py --list
    python scripts/ci_test_shards.py --check
    python scripts/ci_test_shards.py --run tests-shared --fanout 0 --coverage-data .coverage.py311.tests-shared
    python scripts/ci_test_shards.py --verify-status shard-status/ \
        --run-id 12345 --run-attempt 2 --python-version 3.11
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ci_shard_artifacts import add_artifact_arguments, handle_artifact_command

REPO_ROOT = Path(__file__).resolve().parents[1]
QUARANTINE_FILE = REPO_ROOT / "config" / "test_quarantine.json"

# Mirrors ``[tool.pytest.ini_options] norecursedirs`` plus directories pytest
# never enters (VCS, virtualenvs, node_modules). Keep in sync with pyproject.
_SKIP_DIRS = frozenset(
    {
        "replicants",
        "archive",
        "legacy",
        "experimental",
        ".git",
        ".tox",
        ".nox",
        ".eggs",
        "__pycache__",
        "build",
        "dist",
        ".pytest_cache",
        "htmlcov",
        "node_modules",
        ".venv",
        "venv",
        ".hypothesis",
    }
)

# Files that match pytest's ``test_*.py`` / ``*_test.py`` globs but are not
# test modules: stand-alone scripts kept for manual profiling/signal checks.
# They live at the app root, outside the ``tests/`` package every shard
# targets, so they are deliberately unclaimed. Add an entry only with a
# comment saying why the file is not a test.
_NOT_TEST_MODULES = frozenset(
    {
        # manual perf harness, not a pytest module (see tests/test_gh1655_print_to_logging.py)
        "src/pendulum_simulator/perf_test.py",
        # manual signal harness, same
        "src/pendulum_simulator/signal_test.py",
        # manual simulation smoke script, same
        "src/pendulum_simulator/test_sim.py",
    }
)

PYTEST_MARKER_EXPR = "not live_simulation and not e2e and not requires_network"
CLUB_TESTER_SUITE = "tests/rate_of_closure/test_club_tester_tab.py"


@dataclass(frozen=True)
class Invocation:
    """One ``python -m pytest`` call inside a shard."""

    paths: tuple[str, ...]
    ignores: tuple[str, ...] = ()
    # ``True`` when the target ships its own pytest configuration (rootdir is
    # the sub-app, not the repo). Such suites must be invoked on their own so
    # their markers, ``pythonpath`` and conftest apply.
    own_config: bool = False
    # Numerically intensive convergence studies retain their per-test deadline
    # but run without competing xdist workers (Tools #5130).
    serial: bool = False

    def claims(self, rel_path: str) -> bool:
        if not any(_under(rel_path, root) for root in self.paths):
            return False
        return not any(_under(rel_path, ignored) for ignored in self.ignores)


@dataclass(frozen=True)
class Shard:
    name: str
    invocations: tuple[Invocation, ...] = field(default_factory=tuple)

    def claims(self, rel_path: str) -> bool:
        return any(inv.claims(rel_path) for inv in self.invocations)


def _under(rel_path: str, root: str) -> bool:
    return rel_path == root or rel_path.startswith(root.rstrip("/") + "/")


_TESTS_OWNED_ELSEWHERE = (
    "tests/shared",
    "tests/rate_of_closure",
    "tests/unit",
    "tests/architecture",
    "tests/scripts",
    "tests/ops",
)
_SRC_OWNED_ELSEWHERE = (
    "src/shared",
    "src/pendulum_simulator",
    "src/movement_optimizer",
)

SHARDS: tuple[Shard, ...] = (
    Shard(
        "tests-shared",
        (
            Invocation(("tests/shared",), ignores=("tests/shared/python/golf_club",)),
            Invocation(("tests/shared/python/golf_club",), serial=True),
        ),
    ),
    Shard(
        "tests-rate",
        (
            Invocation(("tests/rate_of_closure",), ignores=(CLUB_TESTER_SUITE,)),
            Invocation((CLUB_TESTER_SUITE,), serial=True),
        ),
    ),
    Shard(
        "tests-unit",
        (
            Invocation(
                ("tests/unit", "tests/architecture", "tests/scripts", "tests/ops")
            ),
        ),
    ),
    Shard("tests-rest", (Invocation(("tests",), ignores=_TESTS_OWNED_ELSEWHERE),)),
    Shard("src-shared", (Invocation(("src/shared",)),)),
    Shard(
        "src-embedded",
        (
            Invocation(
                (
                    "src/pendulum_simulator/tests",
                    "src/pendulum_simulator/src/double_pendulum_golf/tests",
                ),
                own_config=True,
            ),
            Invocation(("src/movement_optimizer/tests",), own_config=True),
        ),
    ),
    Shard("src-rest", (Invocation(("src",), ignores=_SRC_OWNED_ELSEWHERE),)),
)

SHARD_NAMES: tuple[str, ...] = tuple(shard.name for shard in SHARDS)


def shard_by_name(name: str) -> Shard:
    for shard in SHARDS:
        if shard.name == name:
            return shard
    raise SystemExit(f"unknown shard {name!r}; known: {', '.join(SHARD_NAMES)}")


def coverage_data_filenames(shard_name: str, python_version: str) -> tuple[str, ...]:
    """Return the exact coverage files emitted by every invocation in a shard."""
    shard = shard_by_name(shard_name)
    base = f".coverage.py{python_version}.{shard_name}"
    return _coverage_files(base, len(shard.invocations))


def _is_test_file(name: str) -> bool:
    return name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py")
    )


def iter_test_files(repo_root: Path = REPO_ROOT) -> list[str]:
    """Every repo-relative test module pytest could collect under tests/ and src/."""
    found: list[str] = []
    for base in ("tests", "src"):
        for dirpath, dirnames, filenames in os.walk(repo_root / base):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in _SKIP_DIRS and not d.endswith(".egg-info")
            )
            rel_dir = Path(dirpath).relative_to(repo_root).as_posix()
            found.extend(
                f"{rel_dir}/{f}" for f in sorted(filenames) if _is_test_file(f)
            )
    return found


def load_quarantine(path: Path = QUARANTINE_FILE) -> list[dict[str, str]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: 'entries' must be a list")
    return [dict(entry) for entry in entries]


def quarantined_paths(path: Path = QUARANTINE_FILE) -> list[str]:
    return [entry["path"] for entry in load_quarantine(path)]


def check_partition(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return human-readable problems; empty list means the partition is sound."""
    problems: list[str] = []
    for rel in iter_test_files(repo_root):
        if rel in _NOT_TEST_MODULES:
            continue
        owners = [shard.name for shard in SHARDS if shard.claims(rel)]
        if len(owners) != 1:
            problems.append(f"{rel}: claimed by {owners or 'no shard'}")
    for rel in sorted(_NOT_TEST_MODULES):
        if not (repo_root / rel).is_file():
            problems.append(f"{rel}: listed in _NOT_TEST_MODULES but does not exist")
    for entry in load_quarantine(repo_root / "config" / "test_quarantine.json"):
        for key in ("path", "owner", "issue", "reason"):
            if not entry.get(key):
                problems.append(f"quarantine entry {entry!r} is missing {key!r}")
        rel = entry.get("path", "")
        if rel and not (repo_root / rel).exists():
            problems.append(f"quarantine entry {rel}: path no longer exists (drop it)")
        if (
            rel
            and not any(shard.claims(rel) for shard in SHARDS)
            and (repo_root / rel).is_file()
        ):
            problems.append(f"quarantine entry {rel}: no shard would run it anyway")
    return problems


def pytest_command(
    invocation: Invocation,
    *,
    fanout: str,
    extra: tuple[str, ...] = (),
    quarantine: tuple[str, ...] = (),
) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", *invocation.paths]
    for ignored in invocation.ignores:
        cmd.append(f"--ignore={ignored}")
    for rel in quarantine:
        if invocation.claims(rel):
            cmd.append(f"--ignore={rel}")
    cmd += ["-m", PYTEST_MARKER_EXPR, "-n", "0" if invocation.serial else fanout]
    if not invocation.own_config:
        # Root addopts already carry --strict-markers/--durations; only the
        # xdist fan-out and marker expression are overridden per lane.
        cmd += ["--dist", "loadscope"]
    cmd += [
        # Measure only. The single coverage floor is
        # ``[tool.coverage.report] fail_under`` in pyproject.toml, applied by
        # ``coverage report`` on the *combined* data in the tests-gate job; a
        # per-shard floor would reject every shard for the code it does not
        # exercise. The literal 0 disables pytest-cov's per-run copy of that
        # floor and is not a coverage target.
        "--cov",
        "--cov-report=",
        "--cov-fail-under=0",
        *extra,
    ]
    return cmd


def _coverage_files(base: str, invocation_count: int) -> tuple[str, ...]:
    if invocation_count < 1:
        raise ValueError("a shard must contain at least one invocation")
    if invocation_count == 1:
        return (base,)
    return tuple(f"{base}.{index}" for index in range(invocation_count))


def run_shard(
    name: str,
    *,
    fanout: str,
    coverage_data: str | None,
    dry_run: bool = False,
) -> int:
    shard = shard_by_name(name)
    quarantine = tuple(quarantined_paths())
    rc = 0
    for index, invocation in enumerate(shard.invocations):
        env = dict(os.environ)
        if coverage_data:
            filenames = _coverage_files(coverage_data, len(shard.invocations))
            env["COVERAGE_FILE"] = filenames[index]
        cmd = pytest_command(invocation, fanout=fanout, quarantine=quarantine)
        print("+", " ".join(cmd), flush=True)
        if dry_run:
            continue
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        # pytest exit 5 == "no tests collected": for a shard that is a real
        # failure (the partition promised tests here).
        if result.returncode != 0:
            rc = result.returncode
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list", action="store_true", help="print shard names, one per line"
    )
    group.add_argument(
        "--list-json", action="store_true", help="print shard names as a JSON list"
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="verify the partition covers every test file exactly once",
    )
    group.add_argument(
        "--run", metavar="SHARD", help="run one shard's pytest invocations"
    )
    add_artifact_arguments(parser, group)
    parser.add_argument(
        "--fanout", default="0", help="pytest-xdist -n value (default 0)"
    )
    parser.add_argument(
        "--coverage-data", default=None, help="COVERAGE_FILE base name for the run"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands without running them"
    )
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(SHARD_NAMES))
        return 0
    if args.list_json:
        print(json.dumps(list(SHARD_NAMES)))
        return 0
    if args.check:
        problems = check_partition()
        if problems:
            print("Test-shard partition is broken:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        total = len(iter_test_files())
        print(f"Partition OK: {total} test files across {len(SHARDS)} shards.")
        return 0
    expected_coverage_files = (
        {
            shard: coverage_data_filenames(shard, args.python_version)
            for shard in SHARD_NAMES
        }
        if args.select_coverage
        else {}
    )
    artifact_result: int | None = handle_artifact_command(
        args,
        parser,
        tuple(SHARD_NAMES),
        expected_coverage_files,
    )
    if artifact_result is not None:
        return artifact_result
    if args.run:
        return run_shard(
            args.run,
            fanout=args.fanout,
            coverage_data=args.coverage_data,
            dry_run=args.dry_run,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
