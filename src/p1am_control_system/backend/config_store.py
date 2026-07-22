"""Durable operator-configuration store (SQLite).

Persists operator-set control *settings* so the machine comes back with the last
session's configuration after a restart instead of resetting to defaults —
alarm/interlock setpoints, PID parameters, heater/power-supply limits, the last
commanded setpoints, the historian capture rate and the performance mode.

Safety scope: this stores **settings only**. It never persists a *running* state
— a heater always comes back stopped (IDLE); the operator presses Start to
resume to the recalled setpoint. Restoring limits/setpoints on boot cannot
energize an output on its own.

Storage is a single key -> JSON-blob table in the same ``dcs_scada.db`` the
historian uses (already WAL-tuned). Values are small and read once at startup /
written on each operator change, so this adds negligible load.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TypeVar, cast

from models import utc_now
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel

logger = logging.getLogger("dcs_backend.config_store")

__all__ = [
    "PersistedConfig",
    "save_config",
    "load_config",
    "save_model",
    "load_model",
]

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class PersistedConfig(SQLModel, table=True):  # type: ignore[call-arg]
    """One persisted operator setting: a JSON blob keyed by a stable name."""

    key: str = Field(primary_key=True)
    value_json: str
    updated_at: datetime = Field(default_factory=utc_now)


def save_config(session: Session, key: str, payload: dict[str, object]) -> None:
    """Upsert a JSON-serializable ``payload`` under ``key``.

    Args:
        session: An active SQLModel session bound to the config DB.
        key: Non-empty stable identifier (e.g. ``"routing"``, ``"temperature"``).
        payload: A JSON-serializable dict.

    Raises:
        TypeError: if ``key`` is not a str or ``payload`` is not a dict.
        ValueError: if ``key`` is empty or ``payload`` is not JSON-serializable.
    """
    if not isinstance(key, str):
        raise TypeError(f"key must be a str, got {type(key).__name__}")
    if not key:
        raise ValueError("key must be non-empty")
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")
    try:
        text = json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"payload for {key!r} is not JSON-serializable: {exc}"
        ) from exc

    row = session.get(PersistedConfig, key)
    if row is None:
        session.add(PersistedConfig(key=key, value_json=text))
    else:
        row.value_json = text
        row.updated_at = utc_now()
        session.add(row)
    session.commit()


def load_config(session: Session, key: str) -> dict[str, object] | None:
    """Return the stored dict for ``key``, or ``None`` if absent/corrupt.

    Never raises on a corrupt/legacy blob — it logs and returns ``None`` so a bad
    row can only fall back to defaults, never block startup of the controller.

    Raises:
        TypeError: if ``key`` is not a str.
    """
    if not isinstance(key, str):
        raise TypeError(f"key must be a str, got {type(key).__name__}")
    row = session.get(PersistedConfig, key)
    if row is None:
        return None
    try:
        data = json.loads(row.value_json)
    except (json.JSONDecodeError, ValueError) as exc:  # pragma: no cover - corrupt row
        logger.warning("Discarding corrupt persisted config %r: %s", key, exc)
        return None
    if not isinstance(data, dict):  # pragma: no cover - unexpected shape
        logger.warning("Persisted config %r is not an object; ignoring", key)
        return None
    return data


def save_model(session: Session, key: str, model: BaseModel) -> None:
    """Persist a pydantic ``model`` under ``key`` (JSON-mode dump).

    Raises:
        TypeError: if ``model`` is not a pydantic ``BaseModel``.
    """
    if not isinstance(model, BaseModel):
        raise TypeError(
            f"model must be a pydantic BaseModel, got {type(model).__name__}"
        )
    save_config(session, key, model.model_dump(mode="json"))


def load_model(session: Session, key: str, model_cls: type[_ModelT]) -> _ModelT | None:
    """Load and validate a persisted setting back into ``model_cls``.

    Returns ``None`` when nothing is stored OR when the stored blob no longer
    matches the current model schema (a config model changed shape since it was
    written) — the caller then keeps its default. Schema drift must never crash
    the boot of a safety-critical controller.
    """
    data = load_config(session, key)
    if data is None:
        return None
    try:
        return cast("_ModelT", model_cls(**data))
    except Exception as exc:  # noqa: BLE001 - any validation error -> default
        logger.warning(
            "Persisted config %r no longer matches %s; using default (%s)",
            key,
            model_cls.__name__,
            exc,
        )
        return None
