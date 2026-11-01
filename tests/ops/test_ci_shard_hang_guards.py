"""A crashed xdist worker must fail its shard fast, never stall it (Tools #5899).

pytest-timeout's ``thread`` method ends a stuck test by calling ``os._exit`` on
the xdist worker. xdist then restarts the worker, and in ``tests-rate`` the
controller never finished: the job sat silent until its 90-minute timeout and
ejected merge-queue entries. With no worker restarts allowed, the crash ends
the session at once with a "worker crashed while running <test>" error, and a
step deadline bounds any other stall well inside the job timeout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
SHARDS_SCRIPT = REPO_ROOT / "scripts" / "ci_test_shards.py"
RUN_TESTS_STEP = "Run Tests with Coverage (Python ${{ matrix.python-version }})"
# The slowest green shard step (tests-shared on 3.12) takes about 21 minutes;
# the deadline leaves twice that while freeing the runner long before the
# 90-minute job timeout, which also has to absorb a cold-cache setup.
MAX_TEST_STEP_MINUTES = 45

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _load_shards_module() -> Any:
    scripts_dir = str(SHARDS_SCRIPT.parent)
    inserted = scripts_dir not in sys.path
    if inserted:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("ci_test_shards", SHARDS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if inserted:
        sys.path.remove(scripts_dir)
    return module


def _tests_job() -> dict[str, Any]:
    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    return cast(dict[str, Any], workflow["jobs"]["tests"])


def test_every_parallel_invocation_forbids_worker_restarts() -> None:
    shards = _load_shards_module()
    parallel = [
        invocation
        for shard in shards.SHARDS
        for invocation in shard.invocations
        if not invocation.serial
    ]
    assert parallel, "expected at least one xdist invocation"
    for invocation in parallel:
        command = shards.pytest_command(invocation, fanout="4")
        assert "--max-worker-restart=0" in command, invocation.paths


def test_test_step_deadline_sits_well_inside_the_job_timeout() -> None:
    job = _tests_job()
    step = next(s for s in job["steps"] if s.get("name") == RUN_TESTS_STEP)
    deadline = step.get("timeout-minutes")
    assert isinstance(deadline, int), "the test step needs its own timeout-minutes"
    assert 0 < deadline <= MAX_TEST_STEP_MINUTES
    assert deadline < job["timeout-minutes"]
