"""Recovery-package contracts: verified, bounded, and de-energized."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
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
from recovery_package import RecoveryPackageService  # noqa: E402

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _routing() -> RoutingConfig:
    return RoutingConfig(
        input_routing=["TAG_0"],
        output_routing=[],
        pids=[],
        interlocks={
            "TAG_0": InterlockConfig(
                lolo_limit=0,
                low_limit=10,
                high_limit=90,
                hihi_limit=100,
            )
        },
    )


async def _active_workflow() -> tuple[ConfigurationWorkflow, list[RoutingConfig]]:
    deployed: list[RoutingConfig] = []

    async def deploy(config: RoutingConfig) -> None:
        deployed.append(config)

    workflow = ConfigurationWorkflow(InMemoryRevisionRepository(), deploy)
    engineer = Principal("engineer", "Engineer", Role.ENGINEER)
    admin = Principal("admin", "Admin", Role.ADMIN)
    revision = workflow.create_draft(_routing(), engineer, "Synthetic baseline")
    workflow.validate(revision.revision_id, engineer)
    workflow.submit_for_review(revision.revision_id, engineer)
    workflow.approve(revision.revision_id, engineer, "Synthetic approval")
    await workflow.activate(revision.revision_id, admin)
    return workflow, deployed


@pytest.mark.asyncio
async def test_backup_round_trip_restores_only_as_a_draft() -> None:
    workflow, deployed = await _active_workflow()
    service = RecoveryPackageService(
        workflow,
        software_revision="software-test-1",
        clock=lambda: datetime(2026, 8, 3, tzinfo=UTC),
    )
    artifact = service.create()

    verified = service.verify(artifact.payload, artifact.sha256)
    restored = service.restore_as_draft(
        artifact.payload,
        Principal("restore-engineer", "Restore Engineer", Role.ENGINEER),
        "Synthetic restore exercise",
        artifact.sha256,
    )

    assert verified.manifest.data_classification == "configuration_backup"
    assert verified.manifest.not_for_live_control is True
    assert verified.manifest.energized_state_included is False
    assert restored.state is ConfigurationState.DRAFT
    assert restored.source_revision_id is None
    assert len(deployed) == 1  # restore did not invoke the deployment adapter


@pytest.mark.asyncio
async def test_tampered_or_wrongly_identified_package_is_rejected() -> None:
    workflow, _deployed = await _active_workflow()
    service = RecoveryPackageService(workflow, software_revision="software-test-1")
    artifact = service.create()
    tampered = artifact.payload[:-1] + bytes([artifact.payload[-1] ^ 1])

    with pytest.raises(ValueError, match="checksum"):
        service.verify(tampered, artifact.sha256)

    with pytest.raises(ValueError, match="checksum"):
        service.verify(artifact.payload, "0" * 64)


def test_backup_requires_an_identified_active_revision() -> None:
    async def deploy(_config: RoutingConfig) -> None:
        return None

    workflow = ConfigurationWorkflow(InMemoryRevisionRepository(), deploy)
    service = RecoveryPackageService(workflow, software_revision="software-test-1")

    with pytest.raises(ValueError, match="active"):
        service.create()
