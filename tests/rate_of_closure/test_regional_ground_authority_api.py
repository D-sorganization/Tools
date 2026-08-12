"""Fail-closed contract tests for the regional-ground browser authority."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rate_of_closure.web_authority.api import create_authority_app
from rate_of_closure.web_authority.capability import (
    AUTHORITY_CAPABILITY_SCHEMA_VERSION,
    AuthorityCapability,
)
from rate_of_closure.web_authority.runtime import build_authority_process_spec


def test_capability_defaults_to_non_executable() -> None:
    capability = AuthorityCapability.unavailable(
        reason_code="execution_profile_unqualified",
        detail="Exact flight and ground execution profile is not qualified.",
    )

    assert capability.to_wire() == {
        "schema_version": AUTHORITY_CAPABILITY_SCHEMA_VERSION,
        "authority_id": "rate-of-closure-python-authority",
        "authority_version": "1",
        "available": False,
        "regional_ground_execution": False,
        "reason_code": "execution_profile_unqualified",
        "detail": "Exact flight and ground execution profile is not qualified.",
    }


def test_app_requires_nonempty_ephemeral_token() -> None:
    with pytest.raises(ValueError, match="token"):
        create_authority_app(token="")


def test_capability_endpoint_requires_exact_bearer_token() -> None:
    client = TestClient(create_authority_app(token="test-ephemeral-token"))

    missing = client.get("/api/rate-of-closure/v1/capabilities")
    wrong = client.get(
        "/api/rate-of-closure/v1/capabilities",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"


def test_capability_endpoint_returns_injected_fail_closed_state() -> None:
    capability = AuthorityCapability.unavailable(
        reason_code="runner_not_started",
        detail="Qualified execution runner is not started.",
    )
    client = TestClient(
        create_authority_app(token="test-ephemeral-token", capability=capability)
    )

    response = client.get(
        "/api/rate-of-closure/v1/capabilities",
        headers={"Authorization": "Bearer test-ephemeral-token"},
    )

    assert response.status_code == 200
    assert response.json() == capability.to_wire()
    assert response.headers["cache-control"] == "no-store"


def test_authority_process_spec_keeps_token_out_of_command(tmp_path) -> None:
    spec = build_authority_process_spec(
        token="test-ephemeral-token",
        port=54321,
        source_root=tmp_path,
    )

    assert spec.command[-2:] == ("--no-access-log", "--log-level=warning")
    assert "test-ephemeral-token" not in " ".join(spec.command)
    assert spec.environment["ROC_AUTHORITY_TOKEN"] == "test-ephemeral-token"
    assert spec.environment["PYTHONPATH"].split(";")[0] == str(tmp_path)
