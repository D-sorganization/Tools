"""Contracts for protected, immutable configuration revision workflows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_workflow import (  # noqa: E402
    ConfigurationState,
    ConfigurationWorkflow,
    InMemoryRevisionRepository,
)
from identity import Principal, Role  # noqa: E402
from models import InterlockConfig, RoutingConfig  # noqa: E402


def _principal(subject: str, role: Role = Role.ENGINEER) -> Principal:
    return Principal(subject=subject, display_name=subject.title(), role=role)


def _routing(high: float = 90) -> RoutingConfig:
    return RoutingConfig(
        input_routing=["TAG_0"],
        output_routing=[],
        pids=[],
        interlocks={
            "TAG_0": InterlockConfig(
                lolo_limit=0,
                low_limit=10,
                high_limit=high,
                hihi_limit=100,
            )
        },
    )


async def _approved_revision(
    workflow: ConfigurationWorkflow,
    high: float = 90,
):
    author = _principal("author")
    reviewer = _principal("reviewer")
    draft = workflow.create_draft(_routing(high), author, "Synthetic test change")
    validated = workflow.validate(draft.revision_id, author)
    in_review = workflow.submit_for_review(validated.revision_id, author)
    return workflow.approve(in_review.revision_id, reviewer, "Reviewed synthetic diff")


@pytest.mark.asyncio
async def test_protected_revision_requires_every_transition_before_activation() -> None:
    deployed: list[RoutingConfig] = []

    async def deploy(config: RoutingConfig) -> None:
        deployed.append(config)

    workflow = ConfigurationWorkflow(InMemoryRevisionRepository(), deploy)
    draft = workflow.create_draft(
        _routing(), _principal("author"), "Synthetic test change"
    )

    with pytest.raises(ValueError, match="approved"):
        await workflow.activate(draft.revision_id, _principal("admin", Role.ADMIN))

    approved = await _approved_revision(workflow)
    active = await workflow.activate(
        approved.revision_id, _principal("admin", Role.ADMIN)
    )

    assert active.state is ConfigurationState.ACTIVE
    assert active.activated_by == "admin"
    assert active.activation_identity == approved.revision_id
    assert active.activation_identity.startswith("cfg-")
    assert deployed == [_routing()]


@pytest.mark.asyncio
async def test_failed_deployment_does_not_claim_an_active_revision() -> None:
    async def fail_deploy(_config: RoutingConfig) -> None:
        raise RuntimeError("synthetic adapter refused deployment")

    repository = InMemoryRevisionRepository()
    workflow = ConfigurationWorkflow(repository, fail_deploy)
    approved = await _approved_revision(workflow)

    with pytest.raises(RuntimeError, match="refused"):
        await workflow.activate(approved.revision_id, _principal("admin", Role.ADMIN))

    assert workflow.get(approved.revision_id).state is ConfigurationState.APPROVED
    assert workflow.active() is None


@pytest.mark.asyncio
async def test_rollback_clones_history_into_a_new_identified_revision() -> None:
    deployed: list[RoutingConfig] = []

    async def deploy(config: RoutingConfig) -> None:
        deployed.append(config)

    workflow = ConfigurationWorkflow(InMemoryRevisionRepository(), deploy)
    first = await _approved_revision(workflow, high=80)
    first_active = await workflow.activate(
        first.revision_id, _principal("admin", Role.ADMIN)
    )
    second = await _approved_revision(workflow, high=90)
    await workflow.activate(second.revision_id, _principal("admin", Role.ADMIN))

    rollback = await workflow.rollback(
        first_active.revision_id,
        _principal("admin", Role.ADMIN),
        "Synthetic recovery exercise",
    )

    assert rollback.state is ConfigurationState.ACTIVE
    assert rollback.revision_id not in {first.revision_id, second.revision_id}
    assert rollback.source_revision_id == first.revision_id
    assert rollback.payload == first.payload
    assert len(deployed) == 3


def test_validation_and_diff_are_semantic_and_machine_readable() -> None:
    workflow = ConfigurationWorkflow(InMemoryRevisionRepository(), lambda _config: None)
    baseline = workflow.create_draft(
        _routing(80), _principal("author"), "Synthetic baseline"
    )
    workflow.validate(baseline.revision_id, _principal("author"))
    changed = workflow.create_draft(
        _routing(90), _principal("author"), "Synthetic setpoint change"
    )

    diff = workflow.diff(changed.revision_id, baseline.revision_id)
    assert any(
        item.path == "interlocks.TAG_0.high_limit"
        and item.before == 80
        and item.after == 90
        for item in diff
    )

    changed.payload.interlocks["TAG_0"].high_limit = 5
    with pytest.raises(ValueError, match="ordered"):
        workflow.validate(changed.revision_id, _principal("author"))
