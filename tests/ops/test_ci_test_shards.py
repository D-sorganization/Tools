"""Contracts for the sharded full-suite PR lane (Tools #4913).

``ci-standard.yml`` no longer carries a hand-curated ``core_tests`` allowlist,
branch-name special cases, or a directory exclusion for the embedded suites.
Every test file is claimed by exactly one shard of
``scripts/ci_test_shards.py`` and the workflow matrix fans those shards out.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
SHARDS_SCRIPT = REPO_ROOT / "scripts" / "ci_test_shards.py"
QUARANTINE = REPO_ROOT / "config" / "test_quarantine.json"

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _load_shards_module() -> Any:
    spec = importlib.util.spec_from_file_location("ci_test_shards", SHARDS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve postponed annotations through sys.modules[__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workflow() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8")))


def test_partition_claims_every_test_file_exactly_once() -> None:
    shards = _load_shards_module()
    assert shards.check_partition(REPO_ROOT) == []


def test_previously_excluded_embedded_suites_are_claimed() -> None:
    """#3975: the two largest suites were excluded from every PR run."""
    shards = _load_shards_module()
    embedded = shards.shard_by_name("src-embedded")
    for rel in (
        "src/pendulum_simulator/tests/test_constants.py",
        "src/movement_optimizer/tests/test_import.py",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
        assert embedded.claims(rel), rel
    for shard in shards.SHARDS:
        for invocation in shard.invocations:
            for path in invocation.paths:
                assert "movement_optimizer" not in invocation.ignores or path == "src"
    # Nothing in the partition may exclude either suite wholesale.
    assert "src/movement_optimizer" not in shards._TESTS_OWNED_ELSEWHERE
    assert not any(
        "movement_optimizer" in rel or "pendulum_simulator" in rel
        for rel in shards._NOT_TEST_MODULES
        if not rel.startswith("src/pendulum_simulator/")
        or rel.count("/") != 2  # only the three root-level scripts are allowed
    )


def test_workflow_matrix_lists_exactly_the_script_shards() -> None:
    shards = _load_shards_module()
    matrix = _workflow()["jobs"]["tests"]["strategy"]["matrix"]
    assert list(matrix["shard"]) == list(shards.SHARD_NAMES)
    assert _workflow()["jobs"]["tests"]["strategy"]["fail-fast"] is False


def test_shard_workers_bound_native_math_threads_before_imports() -> None:
    """#5130: xdist must not multiply native BLAS/OpenMP thread pools."""
    environment = _workflow()["jobs"]["tests"]["env"]
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        assert environment.get(variable) == "1"


def test_shared_science_has_one_serial_invocation_without_dropping_tests() -> None:
    """Long convergence studies need a core each, not competing worker pools."""
    shards = _load_shards_module()
    shared = shards.shard_by_name("tests-shared")
    target = "tests/shared/python/golf_club/test_shaft_rotating_disturbance.py"
    owners = [
        invocation for invocation in shared.invocations if invocation.claims(target)
    ]
    assert len(owners) == 1
    command = shards.pytest_command(owners[0], fanout="8", quarantine=())
    assert command[command.index("-n") + 1] == "0"
    remaining = next(
        invocation for invocation in shared.invocations if invocation is not owners[0]
    )
    assert remaining.claims("tests/shared/python/theme/test_theme_colors_derivation.py")
    command = shards.pytest_command(remaining, fanout="8", quarantine=())
    assert command[command.index("-n") + 1] == "8"


def test_club_tester_is_serial_without_dropping_its_complete_gui_workflow() -> None:
    """#5114: isolate the coupled solver while preserving all GUI assertions."""
    shards = _load_shards_module()
    rate = shards.shard_by_name("tests-rate")
    target = "tests/rate_of_closure/test_club_tester_tab.py"
    owners = [
        invocation for invocation in rate.invocations if invocation.claims(target)
    ]
    assert len(owners) == 1
    command = shards.pytest_command(owners[0], fanout="4", quarantine=())
    assert command[command.index("-n") + 1] == "0"
    assert owners[0].paths == (target,)
    remaining = next(
        invocation for invocation in rate.invocations if invocation is not owners[0]
    )
    assert remaining.claims("tests/rate_of_closure/test_viewers_gui.py")
    command = shards.pytest_command(remaining, fanout="4", quarantine=())
    assert command[command.index("-n") + 1] == "4"


@pytest.mark.parametrize("shard_name", ["tests-shared", "tests-rate"])
def test_serial_invocations_keep_distinct_coverage_outputs(
    monkeypatch: pytest.MonkeyPatch,
    shard_name: str,
) -> None:
    shards = _load_shards_module()
    outputs = []

    def run(command: list[str], **kwargs: Any) -> Any:
        outputs.append(kwargs["env"]["COVERAGE_FILE"])
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(shards.subprocess, "run", run)
    assert shards.run_shard(shard_name, fanout="8", coverage_data="coverage-shard") == 0
    assert outputs == ["coverage-shard.0", "coverage-shard.1"]


@pytest.mark.parametrize("outcomes", [(1, 0), (0, 1), (0, 5)])
def test_rate_shard_preserves_failures_from_either_invocation(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: tuple[int, int],
) -> None:
    """Neither a serial failure nor empty collection can produce a green shard."""
    shards = _load_shards_module()
    results = iter(outcomes)

    def run(command: list[str], **kwargs: Any) -> Any:
        return type("Result", (), {"returncode": next(results)})()

    monkeypatch.setattr(shards.subprocess, "run", run)
    assert shards.run_shard("tests-rate", fanout="4", coverage_data=None) != 0


def test_workflow_has_no_test_allowlist_or_branch_conditionals() -> None:
    text = CI_STANDARD.read_text(encoding="utf-8")
    assert "core_tests" not in text
    assert "changed_test_files" not in text
    assert "large_consolidation_branch" not in text
    assert "codex/tools-3316-import-canonicalization" not in text
    assert "consolidate/open-prs-20260620" not in text
    assert "codex/src-assert-contracts-3674" not in text
    assert "grep -v -E '^src/movement_optimizer/|^src/pendulum_simulator/'" not in text


def test_shard_job_runs_the_partition_and_records_status() -> None:
    steps = _workflow()["jobs"]["tests"]["steps"]
    run = next(
        step
        for step in steps
        if step.get("name")
        == "Run Tests with Coverage (Python ${{ matrix.python-version }})"
    )
    assert "scripts/ci_test_shards.py --check" in run["run"]
    assert 'scripts/ci_test_shards.py --run "${{ matrix.shard }}"' in run["run"]
    status = next(step for step in steps if step.get("name") == "Record shard status")
    assert status["if"] == "always()"
    assert "steps.run_tests.outcome" in status["run"]


def test_gate_job_keeps_the_required_tests_context() -> None:
    """Branch protection requires ``tests (3.11)``; the gate must own that name."""
    jobs = _workflow()["jobs"]
    gate = jobs["tests-gate"]
    assert gate["name"] == "tests (${{ matrix.python-version }})"
    assert gate["if"] == "always()"
    assert "tests" in gate["needs"]
    assert list(gate["strategy"]["matrix"]["python-version"]) == list(
        jobs["tests"]["strategy"]["matrix"]["python-version"]
    )
    verify = next(
        step
        for step in gate["steps"]
        if step.get("name") == "Require every shard to pass"
    )
    assert "--verify-status" in verify["run"]
    combine = next(
        step
        for step in gate["steps"]
        if step.get("name") == "Combine coverage and apply the floor"
    )
    assert "coverage combine" in combine["run"]
    assert "coverage report" in combine["run"]
    assert "--fail-under" not in combine["run"], "the floor lives in pyproject only"


def test_quarantine_entries_are_owned_and_tracked() -> None:
    data = json.loads(QUARANTINE.read_text(encoding="utf-8"))
    for entry in data["entries"]:
        assert (REPO_ROOT / entry["path"]).exists(), entry["path"]
        assert entry["owner"], entry["path"]
        assert entry["issue"].startswith("#"), entry["path"]
        assert entry["reason"], entry["path"]
        assert entry["path"].endswith(".py"), (
            f"{entry['path']}: quarantine individual modules, not directories"
        )


def test_run_dry_run_lists_one_command_per_invocation() -> None:
    shards = _load_shards_module()
    for shard in shards.SHARDS:
        for invocation in shard.invocations:
            cmd = shards.pytest_command(invocation, fanout="0")
            assert cmd[1:3] == ["-m", "pytest"]
            assert "--cov" in cmd
            assert "-n" in cmd
            for ignored in invocation.ignores:
                assert f"--ignore={ignored}" in cmd
