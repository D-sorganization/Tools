"""Collection ownership for pendulum golfer test-quality epic items (#5923)."""

from __future__ import annotations

from tests.ops._ci_test_shards_loader import load_ci_test_shards_module

_PENDULUM_QUALITY_MODULES = (
    "src/pendulum_simulator/tests/test_counterfactual_golfer.py",
    "src/pendulum_simulator/tests/test_jacobians_golfer.py",
)


def test_pendulum_golfer_quality_modules_claimed_by_src_embedded_shard() -> None:
    shard_by_name = load_ci_test_shards_module().shard_by_name
    shard = shard_by_name("src-embedded")
    for module_path in _PENDULUM_QUALITY_MODULES:
        assert shard.claims(module_path), (
            f"{module_path} must be collected by the src-embedded CI shard"
        )
