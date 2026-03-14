"""
Command line interface for programmatic access to the Data Processor.

The CLI intentionally focuses on two core workflows that users routinely
automate:

1. Quickly inspect a batch of files to discover available signals.
2. Run a lightweight processing pipeline (load → optional filtering →
   optional signal selection → export) defined either via CLI flags or a
   declarative JSON config file.

Example JSON pipeline:
{
  "files": ["./data/example.csv"],
  "combine": true,
  "selected_signals": ["time", "pressure", "temperature"],
  "filter": {
    "filter_type": "Moving Average",
    "ma_window": 5
  },
  "output": {
    "path": "./output/processed.csv",
    "format": "csv"
  }
}

Run with:
    python -m data_processor.cli run --config pipeline.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from .core.data_loader import DataLoader
from .core.signal_processor import SignalProcessor
from .logging_config import get_logger
from .models import FilterConfig, PipelineConfig

if TYPE_CHECKING:
    import pandas as pd

console = Console()
app = typer.Typer(help="Data Processor CLI for automated workflows.")
logger = get_logger(__name__)


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load a JSON pipeline configuration."""
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            return cast(dict[str, Any], json.load(fp))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in config '{config_path}': {exc}"
        raise typer.BadParameter(
            msg,
        ) from exc
    except OSError as exc:
        msg = f"Unable to read config '{config_path}': {exc}"
        raise typer.BadParameter(
            msg,
        ) from exc


def _build_pipeline_config(pipeline_data: dict[str, object]) -> PipelineConfig:
    """Validate and construct a strongly-typed pipeline configuration."""

    try:
        return PipelineConfig.from_mapping(pipeline_data)
    except ValueError as exc:
        msg = f"Invalid pipeline configuration: {exc}"
        raise typer.BadParameter(
            msg,
        ) from exc


def _select_signals(
    df: pd.DataFrame,
    selected_signals: list[str] | None,
    source_label: str,
) -> pd.DataFrame:
    """Return frame restricted to selected signals, with warnings for missing ones."""
    if not selected_signals:
        return df

    valid_signals = [col for col in selected_signals if col in df.columns]
    missing = sorted(set(selected_signals) - set(valid_signals))

    if missing:
        console.print(
            f"[yellow]Warning: missing signals skipped in {source_label} -> "
            f"{', '.join(missing)}[/yellow]",
        )

    if not valid_signals:
        msg = "None of the selected signals are present in the current dataset."
        raise typer.BadParameter(
            msg,
        )

    return df[valid_signals]


def _apply_filter_if_requested(
    df: pd.DataFrame,
    filter_config: FilterConfig | None,
    signal_processor: SignalProcessor,
) -> pd.DataFrame:
    """Apply configured filter if specified."""
    assert df is not None, "df must be provided"
    if filter_config is None:
        return df

    return signal_processor.apply_filter(df, filter_config)


def _process_dataframe(
    df: pd.DataFrame,
    pipeline: PipelineConfig,
    signal_processor: SignalProcessor,
    source_label: str,
) -> pd.DataFrame:
    """Run the configured operations for a single dataframe."""
    assert df is not None, "df must be provided"
    result = df.copy()
    result = _select_signals(
        result,
        pipeline.selected_signals,
        source_label=source_label,
    )
    return _apply_filter_if_requested(
        result,
        pipeline.filter,
        signal_processor,
    )


def _format_output_filename(source_path: str, output_format: str) -> str:
    """Generate an output filename for per-file exports."""
    assert source_path is not None, "source_path must be provided"
    stem = Path(source_path).stem
    extension_map = {
        "csv": ".csv",
        "excel": ".xlsx",
        "xlsx": ".xlsx",
        "parquet": ".parquet",
        "json": ".json",
        "tsv": ".tsv",
    }
    suffix = extension_map.get(output_format.lower(), f".{output_format.lower()}")
    return f"{stem}{suffix}"


@app.command()  # type: ignore[misc]
def detect(
    files: list[Path] = typer.Argument(  # noqa: B008
        ...,
        help="One or more CSV/Parquet data files.",
    ),
    high_perf: bool = typer.Option(
        True,
        "--high-perf/--no-high-perf",
        help="Use the high performance loader.",
    ),
) -> None:
    """Detect and print unique signal names from the supplied files."""
    loader = DataLoader(use_high_performance=high_perf)
    file_paths = [str(path) for path in files]

    console.rule("Signal Detection")
    signals = loader.detect_signals(file_paths)
    if not signals:
        console.print("[yellow]No signals detected.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Detected Signals")
    table.add_column("Signal", justify="left")
    for signal in sorted(signals):
        table.add_row(signal)

    console.print(table)


@app.command()  # type: ignore[misc]
def run(
    config: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="Path to pipeline JSON config. CLI options override values inside.",
    ),
    files: list[Path] | None = typer.Option(  # noqa: B008
        None,
        "--file",
        "-f",
        help="Input files (ignored when provided via config). May be repeated.",
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Destination file path when not specified in config.",
    ),
    output_format: str = typer.Option(
        "csv",
        "--format",
        "-t",
        help="Output format fallback (csv/excel/parquet/json).",
    ),
    combine: bool | None = typer.Option(
        None,
        "--combine/--no-combine",
        help="Override combine flag from config.",
    ),
    high_perf: bool = typer.Option(
        True,
        "--high-perf/--no-high-perf",
        help="Use the high performance loader.",
    ),
) -> None:
    """Execute a lightweight processing pipeline."""
    assert output_format is not None, "output_format must be provided"
    pipeline_data: dict[str, object] = {}
    if config:
        pipeline_data.update(_load_config(config))

    if files:
        pipeline_data["files"] = [str(path) for path in files]

    if combine is not None:
        pipeline_data["combine"] = combine

    if output:
        pipeline_data.setdefault("output", {})
        output_dict = cast("dict[str, object]", pipeline_data["output"])
        output_dict["path"] = str(output)
        output_dict["format"] = output_format

    pipeline = _build_pipeline_config(pipeline_data)
    loader = DataLoader(use_high_performance=high_perf)
    processor = SignalProcessor()

    logger.info("Executing pipeline", extra={"pipeline_config": pipeline.summary()})

    console.rule("Loading data")
    data = loader.load_multiple_files(pipeline.files, combine=pipeline.combine)

    if pipeline.combine:
        _run_combined(data, pipeline, processor, loader)
    else:
        _run_uncombined(data, pipeline, processor, loader)


def _run_combined(
    data: object,
    pipeline: object,
    processor: object,
    loader: DataLoader,
) -> None:
    """Process a combined dataset and optionally save the result."""
    assert data is not None, "data must be provided"
    dataframe = _process_dataframe(
        cast("pd.DataFrame", data),
        pipeline,
        processor,
        source_label="combined dataset",
    )
    if pipeline.output:
        output_path = pipeline.output.path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        loader.save_dataframe(
            dataframe,
            str(output_path),
            format_type=pipeline.output.format,
        )
        console.print(f"[green]Saved processed data to {output_path}[/green]")
    else:
        console.print("[cyan]Pipeline completed (no output specified).[/cyan]")


def _run_uncombined(
    data: object,
    pipeline: object,
    processor: object,
    loader: DataLoader,
) -> None:
    """Process each file independently and optionally save results."""
    assert data is not None, "data must be provided"
    processed_frames: dict[str, pd.DataFrame] = {}
    for source_path, frame in cast("dict[str, pd.DataFrame]", data).items():
        processed_frames[source_path] = _process_dataframe(
            frame,
            pipeline,
            processor,
            source_label=Path(source_path).name,
        )

    if pipeline.output:
        output_path = pipeline.output.ensure_directory_for_uncombined()
        target_format = pipeline.output.format
        output_path.mkdir(parents=True, exist_ok=True)

        for source_path, processed_df in processed_frames.items():
            destination = output_path / _format_output_filename(
                source_path,
                target_format,
            )
            loader.save_dataframe(
                processed_df,
                str(destination),
                format_type=target_format,
            )

        console.print(
            f"[green]Saved processed data for {len(processed_frames)} files "
            f"to {output_path}[/green]",
        )
    else:
        console.print(
            "[cyan]Processed files (no output directory provided): "
            f"{', '.join(Path(p).name for p in processed_frames)}[/cyan]",
        )


def main() -> None:
    """Entry point for `python -m data_processor.cli`."""
    from .logging_config import init_default_logging

    init_default_logging()
    app()


if __name__ == "__main__":
    main()
