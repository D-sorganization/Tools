"""Synthetic contracts for bounded replay resources and preview artifacts."""

from __future__ import annotations

import pytest
from sidekick.lab.mocap import (
    ExperimentCancellationToken,
    ExperimentCancelledError,
    ExperimentResourceBudget,
    ExperimentTiming,
    make_experiment_cache_key,
)


def test_budget_rejects_requests_exceeding_available_resources() -> None:
    budget = ExperimentResourceBudget(
        max_workers=2,
        max_memory_bytes=1024,
        max_disk_bytes=4096,
        max_preview_artifacts=3,
        max_preview_bytes=2048,
    )

    with pytest.raises(ValueError, match="available disk"):
        budget.validate_request(
            workers=1,
            memory_bytes=512,
            disk_bytes=1024,
            available_disk_bytes=512,
        )

    with pytest.raises(ValueError, match="worker"):
        budget.validate_request(
            workers=3,
            memory_bytes=512,
            disk_bytes=1024,
            available_disk_bytes=4096,
        )
    with pytest.raises(ValueError, match="memory"):
        budget.validate_request(
            workers=1,
            memory_bytes=2048,
            disk_bytes=1024,
            available_disk_bytes=4096,
        )


def test_cache_key_changes_when_any_bound_replay_identity_changes() -> None:
    baseline = _bundle()
    changed = _bundle(provider_sha256="4" * 64)

    assert make_experiment_cache_key(baseline) == make_experiment_cache_key(baseline)
    assert make_experiment_cache_key(baseline) != make_experiment_cache_key(changed)


def test_cancellation_is_observed_cooperatively() -> None:
    token = ExperimentCancellationToken()
    assert not token.cancelled
    token.request_cancel()
    assert token.cancelled
    with pytest.raises(ExperimentCancelledError, match="cancellation requested"):
        token.raise_if_cancelled()


def test_cold_and_warm_timings_bind_the_cache_key() -> None:
    cold = ExperimentTiming("c" * 64, 2.5, cache_hit=False)
    warm = ExperimentTiming("c" * 64, 0.25, cache_hit=True)

    assert cold.elapsed_seconds > warm.elapsed_seconds
    assert not cold.cache_hit and warm.cache_hit
    with pytest.raises(ValueError, match="cannot be negative"):
        ExperimentTiming("c" * 64, -0.1, cache_hit=False)


def _bundle(
    *, experiment_id: str = "synthetic-experiment", provider_sha256: str = "2" * 64
):
    from sidekick.lab.mocap import (
        ActuationInputKind,
        CapabilityAvailability,
        CapabilityDeclaration,
        CapabilitySupport,
        InitialStateSchema,
        InputChannel,
        InputInterpolation,
        ModelIdentity,
        ReplayExecutionPolicy,
        ReplayMode,
        StateComponentRole,
        StateComponentSpec,
        build_experiment_replay_bundle,
    )

    schema = InitialStateSchema(
        "synthetic-state",
        "1.0.0",
        (
            StateComponentSpec("q", StateComponentRole.POSITION, 1, "rad", "scalar"),
            StateComponentSpec("v", StateComponentRole.VELOCITY, 1, "rad/s", "scalar"),
        ),
    )
    channel = InputChannel("torque", "actuator:hip", "N*m")
    model = ModelIdentity(
        "engine",
        "model",
        "variant",
        "1.0.0",
        "1" * 64,
        "synthetic-provider",
        "1.0.0",
        provider_sha256,
        schema,
        ("torque",),
    )
    capability = CapabilityDeclaration(
        "forward_dynamics",
        False,
        CapabilitySupport.SUPPORTED,
        CapabilityAvailability.AVAILABLE,
    )
    policy = ReplayExecutionPolicy(
        ReplayMode.SHARED_RIGID_BODY_EMULATION,
        "synthetic-solver",
        "1.0.0",
        "rk4",
        "fixed",
        0.01,
        "synthetic-init",
        "1.0.0",
        "time-player",
        "1.0.0",
        False,
        False,
        False,
        "synthetic-contact",
        "1.0.0",
        "3" * 64,
    )
    return build_experiment_replay_bundle(
        experiment_id,
        model,
        (capability,),
        (("q", (0.0,)), ("v", (0.0,))),
        (channel,),
        ActuationInputKind.ACTUATOR_TORQUE,
        InputInterpolation.ZERO_ORDER_HOLD,
        (0.0, 0.01),
        ((0.0,), (0.0,)),
        policy,
    )
