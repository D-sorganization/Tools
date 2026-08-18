"""Public Morris workspace persistence and aggregate-export facade."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .workspace_csv import morris_report_csv
from .workspace_serialization import (
    dumps_morris_workspace,
    loads_json_document,
    morris_workspace_dict,
)
from .workspace_types import (
    MorrisCompletedEvidence,
    MorrisWorkspace,
    MorrisWorkspaceFactorDraft,
    MorrisWorkspaceSetup,
)
from .workspace_validation import (
    MORRIS_WORKSPACE_EXPORT_SCOPE,
    MORRIS_WORKSPACE_SCHEMA_ID,
    MORRIS_WORKSPACE_SCHEMA_VERSION,
    parse_morris_workspace,
)


def loads_morris_workspace(text: str) -> MorrisWorkspace:
    """Decode and fully validate one bounded strict JSON workspace."""
    return parse_morris_workspace(loads_json_document(text))


def _atomic_write(path: str | os.PathLike[str], text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_morris_workspace(
    workspace: MorrisWorkspace, path: str | os.PathLike[str]
) -> None:
    """Atomically write one validated workspace JSON document."""
    _atomic_write(path, dumps_morris_workspace(workspace))


def write_morris_csv(workspace: MorrisWorkspace, path: str | os.PathLike[str]) -> None:
    """Atomically write deterministic archived aggregate evidence as CSV."""
    _atomic_write(path, morris_report_csv(workspace))


__all__ = [
    "MORRIS_WORKSPACE_EXPORT_SCOPE",
    "MORRIS_WORKSPACE_SCHEMA_ID",
    "MORRIS_WORKSPACE_SCHEMA_VERSION",
    "MorrisCompletedEvidence",
    "MorrisWorkspace",
    "MorrisWorkspaceFactorDraft",
    "MorrisWorkspaceSetup",
    "dumps_morris_workspace",
    "loads_morris_workspace",
    "morris_report_csv",
    "morris_workspace_dict",
    "parse_morris_workspace",
    "write_morris_csv",
    "write_morris_workspace",
]
