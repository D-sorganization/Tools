"""Typed client transport for the local durable ensemble authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from rate_of_closure.application.morris.client import (
    MorrisAuthorityClient,
    MorrisAuthorityHttpError,
)

from .contracts import (
    DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
    DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
    DURABLE_ENSEMBLE_SCOPE,
    DurableEnsembleAuthorityRequest,
    DurableEnsembleJobEnvelope,
    parse_durable_ensemble_job,
    parse_durable_ensemble_request,
)

_CAPABILITY_PATH = "/api/rate-of-closure/v1/durable-ensembles/capabilities"
_JOBS_PATH = "/api/rate-of-closure/v1/durable-ensembles/jobs"
_ParsedT = TypeVar("_ParsedT")


@dataclass(frozen=True, slots=True)
class DurableEnsembleCapability:
    """Exact availability and schema declaration for the local authority."""

    available: bool
    api_prefix: str


def _capability(value: object) -> DurableEnsembleCapability:
    fields = frozenset(
        {
            "schema_id",
            "schema_version",
            "available",
            "api_prefix",
            "scope",
            "request_schema_id",
            "job_schema_id",
        }
    )
    item = exact_mapping(value, fields, "durable ensemble capability")
    expected = {
        "schema_id": "rate-of-closure/durable-ensemble-authority-capability",
        "schema_version": 1,
        "scope": DURABLE_ENSEMBLE_SCOPE,
        "request_schema_id": DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
        "job_schema_id": DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
    }
    if any(item[key] != expected[key] for key in expected):
        raise ValueError("durable ensemble capability is incompatible")
    if type(item["available"]) is not bool:
        raise TypeError("durable ensemble capability available must be boolean")
    if item["api_prefix"] != "/api/rate-of-closure/v1":
        raise ValueError("durable ensemble capability api_prefix is unsupported")
    return DurableEnsembleCapability(item["available"], item["api_prefix"])


@dataclass(frozen=True, slots=True)
class DurableEnsembleAuthorityClient:
    """Progress, cancel, resume, and inspect client using one shared transport."""

    base_url: str
    headers: Mapping[str, str] = field(repr=False)
    timeout_s: float = 5.0
    _transport: MorrisAuthorityClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        transport = MorrisAuthorityClient(self.base_url, self.headers, self.timeout_s)
        object.__setattr__(self, "base_url", transport.base_url)
        object.__setattr__(self, "headers", transport.headers)
        object.__setattr__(self, "_transport", transport)

    def capability(self) -> DurableEnsembleCapability:
        """Fetch the exact authority availability declaration."""
        return self._parse(
            _capability,
            self._transport.request_document("GET", _CAPABILITY_PATH, None, 200),
        )

    def create(
        self, request: DurableEnsembleAuthorityRequest | object
    ) -> DurableEnsembleJobEnvelope:
        """Create or exactly resume one server-owned archive."""
        parsed = (
            request
            if isinstance(request, DurableEnsembleAuthorityRequest)
            else parse_durable_ensemble_request(request)
        )
        document = self._transport.request_document(
            "POST", _JOBS_PATH, parsed.to_json_dict(), 202
        )
        return self._parse(parse_durable_ensemble_job, document)

    def status(self, job_id: str) -> DurableEnsembleJobEnvelope:
        """Inspect current incremental evidence for one job."""
        document = self._transport.request_document(
            "GET", self._job_path(job_id), None, 200
        )
        return self._parse(parse_durable_ensemble_job, document)

    def cancel(self, job_id: str) -> DurableEnsembleJobEnvelope:
        """Request idempotent cancellation without deleting its prefix."""
        document = self._transport.request_document(
            "DELETE", self._job_path(job_id), None, (200, 202)
        )
        return self._parse(parse_durable_ensemble_job, document)

    @staticmethod
    def _job_path(job_id: str) -> str:
        return f"{_JOBS_PATH}/{stable_id(job_id, 'job_id')}"

    @staticmethod
    def _parse(parser: Callable[[object], _ParsedT], document: object) -> _ParsedT:
        try:
            return parser(document)
        except (TypeError, ValueError) as exc:
            raise MorrisAuthorityHttpError(
                0, "durable ensemble response failed validation"
            ) from exc


DurableEnsembleAuthorityHttpError = MorrisAuthorityHttpError

__all__ = [
    "DurableEnsembleAuthorityClient",
    "DurableEnsembleAuthorityHttpError",
    "DurableEnsembleCapability",
]
