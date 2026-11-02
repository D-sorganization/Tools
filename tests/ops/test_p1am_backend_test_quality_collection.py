"""Collection ownership for P1AM backend test-quality epic items (#5924)."""

from __future__ import annotations

from tests.ops._ci_test_shards_loader import load_ci_test_shards_module

_P1AM_QUALITY_MODULES = (
    "src/p1am_control_system/backend/tests/test_alarm_ack_engine.py",
    "src/p1am_control_system/backend/tests/test_p1am_app_import_guard.py",
    "tests/p1am_control_system/test_alicat_manager.py",
)


def test_p1am_backend_quality_modules_claimed_by_ci_shards() -> None:
    shard_by_name = load_ci_test_shards_module().shard_by_name
    embedded = shard_by_name("src-rest")
    tests_rest = shard_by_name("tests-rest")
    assert embedded.claims(_P1AM_QUALITY_MODULES[0])
    assert embedded.claims(_P1AM_QUALITY_MODULES[1])
    assert tests_rest.claims(_P1AM_QUALITY_MODULES[2])
