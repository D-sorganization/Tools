"""Identity context for plots and exports (issue #4740, mirrors UpstreamDrift #8828).

Carries engine, model, run_id, and version context so it can be rendered as a
figure footer and embedded in export metadata.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "PlotIdentity",
    "apply_identity_footer",
    "resolve_and_apply_identity_footer",
]


@dataclass(frozen=True)
class PlotIdentity:
    """Optional engine/model/run/version identity attached to a plot or export.

    Attributes:
        engine: Physics engine name (e.g. ``"mujoco"``, ``"drake"``), when known.
        model: Model name loaded into the engine, when known.
        run_id: Identifier for the recording/run that produced the data, when known.
        version: Model/software version string, when known.
        timestamp: Optional run or export timestamp.
    """

    engine: str | None = None
    model: str | None = None
    run_id: str | None = None
    version: str | None = None
    timestamp: str | datetime.datetime | None = None

    @classmethod
    def from_recorder(cls, recorder: Any, run_id: str | None = None) -> PlotIdentity:
        """Derive identity from a ``RecorderInterface``-like object.

        Reads attributes that are present on the recorder's ``engine``
        (``engine_type`` / ``model_name``). Missing attributes are left ``None``.

        Args:
            recorder: Object exposing an optional ``.engine`` attribute.
            run_id: Explicit run identifier, if known by the caller.

        Returns:
            A ``PlotIdentity`` populated with whatever was discoverable.
        """
        engine_obj = getattr(recorder, "engine", None)
        engine_name: str | None = None
        model_name: str | None = None
        version_str: str | None = None

        if engine_obj is not None:
            raw_engine_type = getattr(engine_obj, "engine_type", None)
            if raw_engine_type is not None:
                engine_name = str(getattr(raw_engine_type, "value", raw_engine_type))

            raw_model_name = getattr(engine_obj, "model_name", None) or getattr(
                engine_obj, "model", None
            )
            if isinstance(raw_model_name, str) and raw_model_name:
                model_name = raw_model_name

            raw_version = getattr(engine_obj, "version", None)
            if isinstance(raw_version, str) and raw_version:
                version_str = raw_version

        return cls(
            engine=engine_name,
            model=model_name,
            run_id=run_id,
            version=version_str,
        )

    def is_empty(self) -> bool:
        """Return True when no identity field is populated."""
        return (
            self.engine is None
            and self.model is None
            and self.run_id is None
            and self.version is None
            and self.timestamp is None
        )

    def label(self) -> str | None:
        """Render a short human-readable label, or None if nothing is known."""
        parts: list[str] = []
        if self.engine:
            parts.append(f"Engine: {self.engine}")
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.run_id:
            parts.append(f"Run: {self.run_id}")
        if self.version:
            parts.append(f"Version: {self.version}")
        return " | ".join(parts) if parts else None

    def as_metadata_dict(self) -> dict[str, str]:
        """Return identity fields as a flat string dict for export metadata."""
        meta: dict[str, str] = {}
        if self.engine is not None:
            meta["engine"] = str(self.engine)
        if self.model is not None:
            meta["model"] = str(self.model)
        if self.run_id is not None:
            meta["run_id"] = str(self.run_id)
        if self.version is not None:
            meta["version"] = str(self.version)
        if self.timestamp is not None:
            meta["timestamp"] = (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime.datetime)
                else str(self.timestamp)
            )
        return meta


def apply_identity_footer(fig: Figure, identity: PlotIdentity | None) -> None:
    """Render ``identity`` as a small footer on ``fig``, if any is known.

    No-op when ``identity`` is ``None`` or carries no populated fields.
    """
    if identity is None:
        return
    label = identity.label()
    if not label:
        return
    fig.text(
        0.99,
        0.01,
        label,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
        alpha=0.85,
    )


def resolve_and_apply_identity_footer(
    fig: Figure, recorder: Any, identity: PlotIdentity | None
) -> PlotIdentity:
    """Resolve ``identity`` (explicit, or derived from ``recorder``) and render it.

    Returns:
        The resolved ``PlotIdentity``.
    """
    resolved = (
        identity if identity is not None else PlotIdentity.from_recorder(recorder)
    )
    apply_identity_footer(fig, resolved)
    return resolved
