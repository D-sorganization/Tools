"""Advisory-only result record with provenance for MPC output (SCADA F16).

Issue #5533. MPC output is a recommendation for an operator to read. It is never
an authoritative setpoint, so this record is advisory by construction.

Safety property -- ADVISORY ONLY:
    - ``authoritative_write_available`` is typed ``Literal[False]`` with a fixed
      default and the model is frozen and strict: a record can neither be built
      with, nor mutated to, a True value.
    - The module is pure (stdlib + pydantic). It receives plain data from the
      caller and has no import path to PLC clients, the hardware seam or
      FastAPI; a test inspects the imports of this module and of ``mpc.py``.
    - Nothing here sits in the E-stop or de-energizing command path.

Design by Contract:
    - Provenance (model id/version, hashes, solver status, timestamp) is
      REQUIRED. A blank or missing item raises instead of yielding a record
      with unknown origin (fail closed).
    - ``created_at`` must be timezone-aware.
    - Hashed payloads must be JSON-representable with finite numbers, so the
      hashes are stable for identical inputs and never depend on wall time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["AdvisoryResult", "build_advisory_result", "stable_hash"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def stable_hash(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 hex digest of ``payload`` in canonical JSON form.

    Raises:
        ValueError: when ``payload`` holds a non-finite number or a value that
            is not JSON-representable (``allow_nan=False`` is the guard).
    """
    try:
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not canonically hashable: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AdvisoryResult(BaseModel):
    """Provenance envelope marking MPC output as advisory, never authoritative."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    finding_type: Literal["advisory"] = "advisory"
    authoritative_write_available: Literal[False] = False
    model_id: str
    model_version: str
    model_hash: str
    inputs_hash: str
    input_window: dict[str, Any]
    constraint_set: dict[str, Any]
    solver_status: str
    created_at: datetime

    @field_validator("model_id", "model_version", "solver_status")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provenance field must not be blank")
        return value

    @field_validator("model_hash", "inputs_hash")
    @classmethod
    def _sha256_hex(cls, value: str) -> str:
        if not _SHA256_HEX.match(value):
            raise ValueError("hash must be 64 lowercase hex characters")
        return value

    @field_validator("created_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


def build_advisory_result(
    *,
    model_id: str,
    model_version: str,
    model_parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
    input_window: Mapping[str, Any],
    constraint_set: Mapping[str, Any],
    solver_status: str,
    created_at: datetime,
) -> AdvisoryResult:
    """Build an :class:`AdvisoryResult`, hashing model parameters and inputs.

    ``model_hash`` covers the model identity and parameters; ``inputs_hash``
    covers the inputs. Neither includes ``created_at``, so identical inputs
    give identical hashes. ``created_at`` defaults to now (UTC).
    """
    model_hash = stable_hash(
        {
            "model_id": model_id,
            "model_version": model_version,
            "parameters": dict(model_parameters),
        }
    )
    return AdvisoryResult(
        model_id=model_id,
        model_version=model_version,
        model_hash=model_hash,
        inputs_hash=stable_hash(dict(inputs)),
        input_window=dict(input_window),
        constraint_set=dict(constraint_set),
        solver_status=solver_status,
        created_at=created_at,
    )
