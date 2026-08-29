"""Figure and data export helpers with metadata injection (Issue #4740)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .identity import PlotIdentity

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "ExportConfig",
    "export_all_figures",
    "export_figure",
    "export_plot_data",
]

_SOFTWARE_NAME = "Tools"


@dataclass
class ExportConfig:
    """Settings for figure and data export.

    Attributes:
        output_dir: Root directory for exported files.
        image_format: Default raster format (``"png"``, ``"jpg"``).
        vector_format: Default vector format (``"pdf"``, ``"svg"``).
        dpi: Resolution for raster exports.
        transparent: Use transparent background.
        bbox_inches: Matplotlib bounding-box mode.
        include_metadata: Embed timestamp and source info in exports.
    """

    output_dir: str | Path = "exports"
    image_format: str = "png"
    vector_format: str = "pdf"
    dpi: int = 300
    transparent: bool = False
    bbox_inches: str = "tight"
    include_metadata: bool = True


def _build_savefig_metadata(
    fmt: str, identity: PlotIdentity | None, timestamp: datetime
) -> dict[str, Any]:
    """Build a ``fig.savefig(metadata=...)`` dict appropriate for *fmt*."""
    fmt = fmt.lower()
    identity = identity or PlotIdentity()
    title = identity.label()

    if fmt == "pdf":
        meta: dict[str, Any] = {"Creator": _SOFTWARE_NAME, "CreationDate": timestamp}
        if title:
            meta["Title"] = title
            meta["Subject"] = title
            meta["Keywords"] = title
        return meta

    if fmt == "svg":
        meta = {"Creator": _SOFTWARE_NAME, "Date": timestamp.isoformat()}
        if title:
            meta["Title"] = title
            meta["Keywords"] = title
        return meta

    # PNG and raster formats accept arbitrary string key-value pairs
    meta = {"Software": _SOFTWARE_NAME, "Creation Time": timestamp.isoformat()}
    if title:
        meta["Title"] = title
    meta.update(identity.as_metadata_dict())
    return meta


def export_figure(
    fig: Figure,
    name: str,
    config: ExportConfig | None = None,
    formats: list[str] | None = None,
    identity: PlotIdentity | None = None,
) -> list[Path]:
    """Save a matplotlib ``Figure`` to one or more formats with metadata.

    Args:
        fig: The figure to export.
        name: Base filename (without extension).
        config: Export configuration (uses defaults if ``None``).
        formats: List of formats to export. Defaults to the formats in *config*.
        identity: Optional engine/model/run/version identity.

    Returns:
        List of paths to the saved files.
    """
    if fig is None:
        raise ValueError("fig must be provided")
    config = config or ExportConfig()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = [config.image_format, config.vector_format]

    timestamp = datetime.now(tz=UTC)

    saved: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        savefig_kwargs: dict[str, Any] = {
            "format": fmt,
            "dpi": config.dpi,
            "transparent": config.transparent,
            "bbox_inches": config.bbox_inches,
        }
        if config.include_metadata:
            savefig_kwargs["metadata"] = _build_savefig_metadata(
                fmt, identity, timestamp
            )
        fig.savefig(str(path), **savefig_kwargs)
        saved.append(path)

    return saved


def export_plot_data(
    data: dict[str, Any],
    name: str,
    config: ExportConfig | None = None,
    fmt: str = "json",
    identity: PlotIdentity | None = None,
) -> Path:
    """Export raw plot data series to CSV or JSON with optional metadata header.

    Args:
        data: Mapping of series names to numpy arrays or lists.
        name: Base filename (without extension).
        config: Export configuration.
        fmt: ``"json"`` or ``"csv"``.
        identity: Optional engine/model/run/version identity.

    Returns:
        Path to the exported file.
    """
    if data is None:
        raise ValueError("data must be provided")
    config = config or ExportConfig()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.{fmt}"

    if fmt == "json":
        payload: dict[str, Any] = {}
        if config.include_metadata:
            meta: dict[str, str] = {
                "exported_at": datetime.now(tz=UTC).isoformat(),
                "source": _SOFTWARE_NAME,
            }
            if identity is not None:
                meta.update(identity.as_metadata_dict())
            payload["_meta"] = meta
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                payload[key] = val.tolist()
            else:
                payload[key] = val
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    elif fmt == "csv":
        columns: dict[str, list[Any]] = {}
        for key, val in data.items():
            arr = np.asarray(val)
            if arr.ndim == 1:
                columns[key] = arr.tolist()
            elif arr.ndim == 2:
                for col in range(arr.shape[1]):
                    columns[f"{key}_{col}"] = arr[:, col].tolist()

        max_rows = max((len(v) for v in columns.values()), default=0)
        with open(path, "w", newline="", encoding="utf-8") as f:
            if config.include_metadata:
                f.write(f"# Source: {_SOFTWARE_NAME}\n")
                f.write(f"# Exported At: {datetime.now(tz=UTC).isoformat()}\n")
                if identity is not None:
                    for k, v in identity.as_metadata_dict().items():
                        f.write(f"# {k}: {v}\n")
            writer = csv.writer(f)
            writer.writerow(list(columns.keys()))
            for row in range(max_rows):
                writer.writerow(
                    [columns[k][row] if row < len(columns[k]) else "" for k in columns]
                )
    else:
        raise ValueError(f"Unsupported export format: {fmt!r}")

    return path


def export_all_figures(
    figures: dict[str, Figure],
    config: ExportConfig | None = None,
    formats: list[str] | None = None,
    identity: PlotIdentity | None = None,
) -> dict[str, list[Path]]:
    """Export multiple named figures at once.

    Args:
        figures: ``{name: Figure}`` mapping.
        config: Shared export configuration.
        formats: Formats for each figure.
        identity: Optional engine/model/run identity applied to every figure.

    Returns:
        ``{name: [paths]}`` mapping.
    """
    if figures is None:
        raise ValueError("figures must be provided")
    results: dict[str, list[Path]] = {}
    for name, fig in figures.items():
        results[name] = export_figure(
            fig, name, config=config, formats=formats, identity=identity
        )
    return results
