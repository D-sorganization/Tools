#!/usr/bin/env python3
"""Partition the whole Tools test tree into CI shards (Tools #4913).

The PR lane used to run a hand-curated ``core_tests`` allowlist plus the tests
changed by the diff, with branch-name special cases and a blanket exclusion of
the two largest embedded suites (``src/pendulum_simulator`` and
``src/movement_optimizer``). This module replaces all of that with one
deterministic partition of *every* test file under ``tests/`` and ``src/`` into
named shards that ``ci-standard.yml`` fans out as a matrix.

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
    python scripts/ci_test_shards.py --verify-status shard-status/ --python-version 3.11
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(frozen=True)
class Invocation:
    """One ``python -m pytest`` call inside a shard."""

    paths: tuple[str, ...]
    ignores: tuple[str, ...] = ()
    # ``True`` when the target ships its own pytest configuration (rootdir is
    # the sub-app, not the repo). Such suites must be invoked on their own so
    # their markers, ``pythonpath`` and conftest apply.
    own_config: bool = False

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
    Shard("tests-shared", (Invocation(("tests/shared",)),)),
    Shard("tests-rate", (Invocation(("tests/rate_of_closure",)),)),
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
    cmd += ["-m", PYTEST_MARKER_EXPR, "-n", fanout]
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
            suffix = f".{index}" if len(shard.invocations) > 1 else ""
            env["COVERAGE_FILE"] = f"{coverage_data}{suffix}"
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


def verify_status(status_dir: Path, python_version: str) -> list[str]:
    """Every shard must have recorded ``success`` for this Python lane."""
    problems: list[str] = []
    for name in SHARD_NAMES:
        status_file = status_dir / f"{python_version}-{name}"
        if not status_file.is_file():
            problems.append(f"{name}: no status recorded (shard did not run?)")
            continue
        outcome = status_file.read_text(encoding="utf-8").strip()
        if outcome != "success":
            problems.append(f"{name}: {outcome}")
    return problems


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
    group.add_argument(
        "--verify-status",
        metavar="DIR",
        help="fail unless every shard recorded success in DIR",
    )
    parser.add_argument(
        "--fanout", default="0", help="pytest-xdist -n value (default 0)"
    )
    parser.add_argument(
        "--coverage-data", default=None, help="COVERAGE_FILE base name for the run"
    )
    parser.add_argument(
        "--python-version", default="", help="lane label used by --verify-status"
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
    if args.run:
        return run_shard(
            args.run,
            fanout=args.fanout,
            coverage_data=args.coverage_data,
            dry_run=args.dry_run,
        )
    if args.verify_status:
        if not args.python_version:
            parser.error("--verify-status requires --python-version")
        problems = verify_status(Path(args.verify_status), args.python_version)
        if problems:
            print(f"Shards failed for Python {args.python_version}:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"All {len(SHARD_NAMES)} shards passed for Python {args.python_version}.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
