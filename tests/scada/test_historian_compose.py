"""Contract and unit tests for historian deployment topology (issue #5547, H8).

Acceptance criteria:
1. Compose parses, defines no literal secrets, and binds no 0.0.0.0 port.
2. Settings validation rejects half-configured remote with clear error.
3. Unreachable remote at startup does not prevent control startup.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from p1am_control_system.backend.settings import P1AMSettings

ROOT = Path(__file__).resolve().parents[2]
HISTORIAN_DIR = ROOT / "deploy" / "historian"
COMPOSE_FILE = HISTORIAN_DIR / "docker-compose.yml"
ENV_EXAMPLE_FILE = HISTORIAN_DIR / ".env.example"

_LITERAL_SECRET_RE = re.compile(
    r"(?:password\s*:\s*['\"][^\$\{\n'\"]+['\"]"
    r"|password\s*:\s*(?!\$\{)[a-zA-Z0-9_\-]+"
    r"|postgres(?:ql)?://[^:]+:[^@]+@)",
    re.IGNORECASE,
)

_WILDCARD_PORT_RE = re.compile(
    r"(?:^0\.0\.0\.0:|^:::"
    r"|^\d+:\d+$"
    r"|^\$\{?[A-Za-z0-9_]+(?::-[^}]*)?\}?:\d+$)",
)


def test_compose_file_exists_and_parses() -> None:
    """docker-compose.yml must exist, parse as valid YAML, and define services."""
    assert COMPOSE_FILE.is_file(), f"Missing docker-compose.yml at {COMPOSE_FILE}"
    content = COMPOSE_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "docker-compose.yml must be a YAML mapping"

    services = data.get("services", {})
    assert "timescale" in services, "compose must define 'timescale' service"
    assert "grafana" in services, "compose must define 'grafana' service"


def test_env_example_exists_and_has_variables() -> None:
    """.env.example must exist and define required environment configuration."""
    assert ENV_EXAMPLE_FILE.is_file(), f"Missing .env.example at {ENV_EXAMPLE_FILE}"
    content = ENV_EXAMPLE_FILE.read_text(encoding="utf-8")
    assert "HISTORIAN_DB" in content
    assert "HISTORIAN_USER" in content
    assert "HISTORIAN_PASSWORD" in content
    assert "GRAFANA_ADMIN_PASSWORD" in content


def test_compose_defines_no_literal_secrets() -> None:
    """docker-compose.yml must interpolate secrets from env vars, not literals."""
    content = COMPOSE_FILE.read_text(encoding="utf-8")
    matches = _LITERAL_SECRET_RE.findall(content)
    assert not matches, f"Found literal secrets in docker-compose.yml: {matches}"


def test_compose_binds_no_wildcard_port() -> None:
    """Published ports must never bind to 0.0.0.0 or omit a loopback IP."""
    data = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services: dict[str, Any] = data.get("services", {})

    for service_name, config in services.items():
        ports = config.get("ports", [])
        for port in ports:
            port_str = str(port).strip()
            # Must not start with 0.0.0.0 or be bare host:container port
            assert not port_str.startswith("0.0.0.0:"), (
                f"Service {service_name} binds to 0.0.0.0: {port_str}"
            )
            assert not _WILDCARD_PORT_RE.match(port_str), (
                f"Service {service_name} exposes port without loopback: {port_str}"
            )
            # Allowed formats: 127.0.0.1:PORT:PORT or internal network
            assert port_str.startswith("127.0.0.1:"), (
                f"Published port in {service_name} must bind 127.0.0.1: {port_str}"
            )


def test_settings_rejects_half_configured_remote_host_without_port() -> None:
    """P1AMSettings must reject remote historian host without a port."""
    with pytest.raises(ValidationError) as exc_info:
        P1AMSettings(historian_remote_host="192.168.1.50")
    assert "historian_remote_port" in str(exc_info.value)


def test_settings_rejects_half_configured_remote_port_without_host() -> None:
    """P1AMSettings must reject remote historian port without a host."""
    with pytest.raises(ValidationError) as exc_info:
        P1AMSettings(historian_remote_port=5432)
    assert "historian_remote_host" in str(exc_info.value)


def test_settings_accepts_valid_remote_configuration() -> None:
    """P1AMSettings accepts fully configured remote host, port, and TLS flag."""
    settings = P1AMSettings(
        historian_remote_host="timescale.internal",
        historian_remote_port=5432,
        historian_remote_tls=True,
    )
    assert settings.historian_remote_host == "timescale.internal"
    assert settings.historian_remote_port == 5432
    assert settings.historian_remote_tls is True


def test_unreachable_remote_does_not_prevent_control_startup() -> None:
    """Unreachable remote historian at startup fails safely and does not block."""
    from p1am_control_system.backend.settings import P1AMSettings
    from p1am_control_system.backend.timescale_sink import (
        FanOutSink,
        create_historian_sink,
    )

    class DummyPrimary:
        def __init__(self) -> None:
            self.closed = False

        def submit(self, record: Any) -> bool:
            return True

        def drain_once(self) -> int:
            return 0

        def stats(self) -> dict[str, Any]:
            return {"primary_ok": True}

        def close(self) -> None:
            self.closed = True

    settings = P1AMSettings(
        historian_remote_host="127.0.0.1",
        historian_remote_port=59999,  # Dead / unreachable port
        historian_remote_tls=False,
    )
    primary = DummyPrimary()

    # Factory / sink creation with remote destination must not raise if dead
    sink = create_historian_sink(
        "dual",
        primary,
        host=settings.historian_remote_host,
        port=settings.historian_remote_port,
        tls=settings.historian_remote_tls,
    )
    assert isinstance(sink, FanOutSink)
    assert sink.stats()["secondary_failures"] == 0
