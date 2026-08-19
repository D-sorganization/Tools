"""Small PyQt boundary for selected-club artifact save dialogs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QLabel, QMessageBox, QWidget

from rate_of_closure.club import (
    ClubAssemblyBinding,
    ClubSpec,
    default_clubhead_engineering_filename,
    default_clubhead_stl_filename,
    parse_club_assembly_binding,
    write_clubhead_engineering_sidecar_atomic,
    write_clubhead_stl_atomic,
)

logger = logging.getLogger(__name__)

ArtifactWriter = Callable[[ClubSpec, str | Path], Path]
FilenameFactory = Callable[[ClubSpec], str]


@dataclass(frozen=True)
class _ArtifactExport:
    """Immutable dialog and persistence policy for one artifact kind."""

    label: str
    filename: FilenameFactory
    file_filter: str
    writer: ArtifactWriter


_STL = _ArtifactExport(
    label="STL",
    filename=default_clubhead_stl_filename,
    file_filter="STL meshes (*.stl);;All files (*)",
    writer=write_clubhead_stl_atomic,
)
_ENGINEERING = _ArtifactExport(
    label="Engineering sidecar",
    filename=default_clubhead_engineering_filename,
    file_filter="Engineering JSON (*.engineering.json *.json);;All files (*)",
    writer=write_clubhead_engineering_sidecar_atomic,
)


def import_club_assembly_binding(
    parent: QWidget, spec: ClubSpec, status: QLabel
) -> ClubAssemblyBinding | None:
    """Prompt for and validate one binding against the exact selected spec."""
    path, _selected = QFileDialog.getOpenFileName(
        parent,
        "Import Club Assembly Binding",
        "",
        "Club assembly binding (*.club-assembly.json *.json);;All files (*)",
    )
    if not path:
        return None
    try:
        binding = parse_club_assembly_binding(spec, Path(path).read_bytes())
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("club assembly binding import failed: %s", exc)
        status.setText("Assembly binding import failed.")
        QMessageBox.warning(parent, "Assembly Binding Import Failed", str(exc))
        return None
    authority = binding.authority
    status.setText(
        f"Assembly binding loaded: {binding.assembly.assembly_id} — "
        f"{authority.kind.value}; {authority.document_id} rev {authority.revision}"
    )
    return binding


def _export_artifact(
    parent: QWidget,
    spec: ClubSpec,
    status: QLabel,
    artifact: _ArtifactExport,
) -> bool:
    """Run one save dialog and atomically persist the selected artifact."""
    path, _selected = QFileDialog.getSaveFileName(
        parent,
        f"Export Selected Clubhead {artifact.label}",
        artifact.filename(spec),
        artifact.file_filter,
    )
    if not path:
        return False
    try:
        artifact.writer(spec, path)
    except (OSError, ValueError) as exc:
        logger.warning("clubhead %s export failed: %s", artifact.label, exc)
        status.setText(f"{artifact.label} export failed.")
        QMessageBox.warning(parent, f"{artifact.label} Export Failed", str(exc))
        return False
    status.setText(f"{artifact.label} exported: {spec.name} — {path}")
    return True


def export_clubhead_stl(parent: QWidget, spec: ClubSpec, status: QLabel) -> bool:
    """Prompt for and atomically write the selected companion STL."""
    return _export_artifact(
        parent, spec, status, replace(_STL, writer=write_clubhead_stl_atomic)
    )


def export_clubhead_engineering_sidecar(
    parent: QWidget,
    spec: ClubSpec,
    status: QLabel,
    binding: ClubAssemblyBinding | None = None,
) -> bool:
    """Prompt for and atomically write the selected engineering JSON."""
    writer: ArtifactWriter = partial(
        write_clubhead_engineering_sidecar_atomic, binding=binding
    )
    return _export_artifact(
        parent,
        spec,
        status,
        replace(_ENGINEERING, writer=writer),
    )


__all__ = [
    "export_clubhead_engineering_sidecar",
    "export_clubhead_stl",
    "import_club_assembly_binding",
]
