"""F11 isolated connector/plugin and diagnostic contracts."""

from __future__ import annotations

from connector_plugins import (
    CommandDisposition,
    ConnectorDescriptor,
    ConnectorManager,
    ConnectorSample,
)


class HealthyConnector:
    descriptor = ConnectorDescriptor(
        connector_id="SYNTHETIC.CONNECTOR.HEALTHY",
        version="1.0.0",
        tags=("SYNTHETIC.HEALTHY.PV",),
        writable_tags=("SYNTHETIC.HEALTHY.SP",),
    )

    def read(self) -> dict[str, float]:
        return {"SYNTHETIC.HEALTHY.PV": 42.0}

    def write(self, tag: str, value: float) -> None:
        assert tag == "SYNTHETIC.HEALTHY.SP"
        assert value == 10

    def diagnostics(self) -> dict[str, object]:
        return {
            "endpoint": "synthetic://healthy",
            "api_token": "do-not-expose",  # pragma: allowlist secret
        }


class FailedConnector:
    descriptor = ConnectorDescriptor(
        connector_id="SYNTHETIC.CONNECTOR.FAILED",
        version="1.0.0",
        tags=("SYNTHETIC.FAILED.PV",),
        writable_tags=("SYNTHETIC.FAILED.SP",),
    )

    def read(self) -> dict[str, float]:
        raise ConnectionError("secret=field-password")  # pragma: allowlist secret

    def write(self, tag: str, value: float) -> None:
        raise ConnectionError("token=field-token")  # pragma: allowlist secret

    def diagnostics(self) -> dict[str, object]:
        return {
            "password": "field-password",  # pragma: allowlist secret
            "state": "offline",
        }


def test_failed_connector_degrades_only_its_tags_without_crashing_poll() -> None:
    manager = ConnectorManager((HealthyConnector(), FailedConnector()))

    samples = manager.poll()

    assert samples["SYNTHETIC.HEALTHY.PV"] == ConnectorSample(
        value=42.0,
        quality="good",
        diagnostic="",
        connector_id="SYNTHETIC.CONNECTOR.HEALTHY",
    )
    assert samples["SYNTHETIC.FAILED.PV"].value is None
    assert samples["SYNTHETIC.FAILED.PV"].quality == "bad"
    assert "SYNTHETIC.CONNECTOR.FAILED" in samples["SYNTHETIC.FAILED.PV"].diagnostic
    assert "field-password" not in samples["SYNTHETIC.FAILED.PV"].diagnostic


def test_failed_and_unknown_commands_fail_closed() -> None:
    manager = ConnectorManager((HealthyConnector(), FailedConnector()))

    accepted = manager.command("SYNTHETIC.HEALTHY.SP", 10)
    failed = manager.command("SYNTHETIC.FAILED.SP", 10)
    unknown = manager.command("SYNTHETIC.UNKNOWN.SP", 10)

    assert accepted.disposition is CommandDisposition.ACCEPTED
    assert failed.disposition is CommandDisposition.REJECTED
    assert unknown.disposition is CommandDisposition.REJECTED
    assert failed.fail_closed is True
    assert "field-token" not in failed.diagnostic


def test_diagnostics_identify_connector_and_redact_secrets() -> None:
    manager = ConnectorManager((HealthyConnector(), FailedConnector()))

    diagnostics = manager.diagnostics()

    assert diagnostics[0].connector_id == "SYNTHETIC.CONNECTOR.HEALTHY"
    assert diagnostics[0].details["api_token"] == "[REDACTED]"
    assert diagnostics[1].details["password"] == "[REDACTED]"
