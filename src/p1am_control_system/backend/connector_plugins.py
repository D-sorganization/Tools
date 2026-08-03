"""Isolated connector plugin contracts with fail-closed commands and redaction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SECRET_FRAGMENTS = ("password", "secret", "token", "api_key", "credential")


def _synthetic(value: str) -> str:
    normalized = value.strip()
    if not normalized.startswith("SYNTHETIC."):
        raise ValueError("connector and tag identifiers must begin with SYNTHETIC.")
    return normalized


class ConnectorDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    connector_id: str
    version: str = Field(min_length=1, max_length=100)
    tags: tuple[str, ...] = Field(min_length=1)
    writable_tags: tuple[str, ...] = ()

    _connector_is_synthetic = field_validator("connector_id")(_synthetic)

    @field_validator("tags", "writable_tags")
    @classmethod
    def _tags_are_synthetic(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_synthetic(value) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("connector tags must be unique")
        return normalized

    @model_validator(mode="after")
    def _read_write_tags_do_not_overlap(self) -> ConnectorDescriptor:
        if set(self.tags) & set(self.writable_tags):
            raise ValueError("read and writable tags must not overlap")
        return self


class ConnectorPlugin(Protocol):
    descriptor: ConnectorDescriptor

    def read(self) -> dict[str, float]: ...

    def write(self, tag: str, value: float) -> None: ...

    def diagnostics(self) -> dict[str, object]: ...


class ConnectorSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float | None
    quality: Literal["good", "bad"]
    diagnostic: str
    connector_id: str


class CommandDisposition(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ConnectorCommandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tag: str
    connector_id: str | None
    disposition: CommandDisposition
    fail_closed: bool
    diagnostic: str


class ConnectorDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)

    connector_id: str
    version: str
    details: dict[str, object]


def _redact(details: Mapping[str, object]) -> dict[str, object]:
    return {
        key: "[REDACTED]"
        if any(fragment in key.casefold() for fragment in _SECRET_FRAGMENTS)
        else value
        for key, value in details.items()
    }


class ConnectorManager:
    def __init__(self, connectors: Sequence[ConnectorPlugin]) -> None:
        self._connectors = tuple(connectors)
        connector_ids = [item.descriptor.connector_id for item in self._connectors]
        if len(connector_ids) != len(set(connector_ids)):
            raise ValueError("connector identifiers must be unique")
        all_tags = [
            tag
            for item in self._connectors
            for tag in (*item.descriptor.tags, *item.descriptor.writable_tags)
        ]
        if len(all_tags) != len(set(all_tags)):
            raise ValueError("tags may belong to only one connector")
        self._writers = {
            tag: connector
            for connector in self._connectors
            for tag in connector.descriptor.writable_tags
        }

    def poll(self) -> dict[str, ConnectorSample]:
        samples: dict[str, ConnectorSample] = {}
        for connector in self._connectors:
            descriptor = connector.descriptor
            try:
                values = connector.read()
                if set(values) != set(descriptor.tags):
                    raise ValueError("connector returned an unexpected tag set")
                for tag, value in values.items():
                    if not math.isfinite(value):
                        raise ValueError("connector returned a non-finite value")
                    samples[tag] = ConnectorSample(
                        value=value,
                        quality="good",
                        diagnostic="",
                        connector_id=descriptor.connector_id,
                    )
            except (
                Exception
            ) as exc:  # Connector boundary intentionally isolates plugins.
                diagnostic = (
                    f"{descriptor.connector_id} read failed ({type(exc).__name__})"
                )
                for tag in descriptor.tags:
                    samples[tag] = ConnectorSample(
                        value=None,
                        quality="bad",
                        diagnostic=diagnostic,
                        connector_id=descriptor.connector_id,
                    )
        return samples

    def command(self, tag: str, value: float) -> ConnectorCommandResult:
        connector = self._writers.get(tag)
        if connector is None:
            return ConnectorCommandResult(
                tag=tag,
                connector_id=None,
                disposition=CommandDisposition.REJECTED,
                fail_closed=True,
                diagnostic="No connector owns this writable tag",
            )
        try:
            if not math.isfinite(value):
                raise ValueError("command value must be finite")
            connector.write(tag, value)
        except Exception as exc:  # Connector boundary intentionally isolates plugins.
            return ConnectorCommandResult(
                tag=tag,
                connector_id=connector.descriptor.connector_id,
                disposition=CommandDisposition.REJECTED,
                fail_closed=True,
                diagnostic=(
                    f"{connector.descriptor.connector_id} command failed "
                    f"({type(exc).__name__})"
                ),
            )
        return ConnectorCommandResult(
            tag=tag,
            connector_id=connector.descriptor.connector_id,
            disposition=CommandDisposition.ACCEPTED,
            fail_closed=False,
            diagnostic="",
        )

    def diagnostics(self) -> list[ConnectorDiagnostic]:
        results: list[ConnectorDiagnostic] = []
        for connector in self._connectors:
            try:
                details = _redact(connector.diagnostics())
            except Exception as exc:
                details = {"error": f"diagnostics failed ({type(exc).__name__})"}
            results.append(
                ConnectorDiagnostic(
                    connector_id=connector.descriptor.connector_id,
                    version=connector.descriptor.version,
                    details=details,
                )
            )
        return results
