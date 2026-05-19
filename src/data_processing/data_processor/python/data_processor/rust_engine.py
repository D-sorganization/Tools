"""Python facade for the native Rust bulk-data engine.

The facade keeps UI and pipeline code behind a stable contract. It intentionally
does not expose Cargo, command-line arguments, or process details to callers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataProcessorRustError(RuntimeError):
    """Raised when the Rust data engine rejects or cannot complete a request."""


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata returned by the native bulk-data inspect operation."""

    format: str
    row_count: int
    columns: list[str]
    byte_size: int


@dataclass(frozen=True)
class PreviewTable:
    """Preview rows returned by the native bulk-data preview operation."""

    columns: list[str]
    rows: list[dict[str, str]]
    rows_returned: int


@dataclass(frozen=True)
class ConversionReport:
    """Report returned by the native bulk-data conversion operation."""

    input: str
    output: str
    output_format: str
    rows_read: int
    rows_written: int
    columns: list[str]
    bytes_written: int


class RustBulkDataEngine:
    """Small command facade for Rust-backed bulk CSV operations."""

    def __init__(
        self,
        *,
        executable: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = repo_root or _find_repo_root()
        self.executable = executable or _find_executable(self.repo_root)
        self._cargo = shutil.which("cargo")

    @classmethod
    def from_repo_root(cls, repo_root: Path | None = None) -> "RustBulkDataEngine":
        """Construct the engine using the current repository layout."""
        return cls(repo_root=repo_root)

    def inspect(self, path: Path | str) -> DatasetMetadata:
        """Inspect a supported dataset without loading it into pandas."""
        payload = self._run(["inspect", str(_require_path(path))])
        return DatasetMetadata(
            format=str(payload["format"]),
            row_count=int(payload["row_count"]),
            columns=[str(column) for column in payload["columns"]],
            byte_size=int(payload["byte_size"]),
        )

    def preview(
        self,
        path: Path | str,
        *,
        rows: int = 100,
        columns: list[str] | None = None,
    ) -> PreviewTable:
        """Return a small preview from a supported dataset."""
        if rows <= 0:
            raise ValueError("rows must be greater than zero")
        args = ["preview", str(_require_path(path)), "--rows", str(rows)]
        if columns:
            args.extend(["--columns", ",".join(columns)])
        payload = self._run(args)
        return PreviewTable(
            columns=[str(column) for column in payload["columns"]],
            rows=[
                {str(key): str(value) for key, value in row.items()}
                for row in payload["rows"]
            ],
            rows_returned=int(payload["rows_returned"]),
        )

    def convert(
        self,
        input_path: Path | str,
        output_path: Path | str,
        *,
        output_format: str = "csv",
        columns: list[str] | None = None,
    ) -> ConversionReport:
        """Convert a supported dataset using the native streaming engine."""
        args = [
            "convert",
            str(_require_path(input_path)),
            str(Path(output_path)),
            "--format",
            output_format,
        ]
        if columns:
            args.extend(["--columns", ",".join(columns)])
        payload = self._run(args)
        return ConversionReport(
            input=str(payload["input"]),
            output=str(payload["output"]),
            output_format=str(payload["output_format"]),
            rows_read=int(payload["rows_read"]),
            rows_written=int(payload["rows_written"]),
            columns=[str(column) for column in payload["columns"]],
            bytes_written=int(payload["bytes_written"]),
        )

    def _run(self, args: list[str]) -> dict[str, Any]:
        command = self._command(args)
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DataProcessorRustError(str(error)) from error

        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise DataProcessorRustError(message)

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise DataProcessorRustError(
                f"Rust engine returned invalid JSON: {completed.stdout[:200]!r}"
            ) from error
        if not isinstance(payload, dict):
            raise DataProcessorRustError("Rust engine returned a non-object response")
        return payload

    def _command(self, args: list[str]) -> list[str]:
        if self.executable is not None:
            return [str(self.executable), *args]
        if self._cargo is None:
            raise DataProcessorRustError(
                "Rust data engine executable was not found and cargo is unavailable"
            )
        return [
            self._cargo,
            "run",
            "-p",
            "data-processor-core",
            "--quiet",
            "--",
            *args,
        ]


def _require_path(path: Path | str) -> Path:
    if path is None:
        raise ValueError("path must be provided")
    resolved = Path(path)
    if not str(resolved):
        raise ValueError("path must be non-empty")
    return resolved


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (parent / "rust_core").is_dir():
            return parent
    raise DataProcessorRustError("Could not locate Tools repository root")


def _find_executable(repo_root: Path) -> Path | None:
    override = os.environ.get("DATA_PROCESSOR_RUST_ENGINE")
    if override:
        path = Path(override)
        if not path.is_file():
            raise DataProcessorRustError(
                f"DATA_PROCESSOR_RUST_ENGINE does not exist: {path}"
            )
        return path

    exe_name = "data-processor-core.exe" if os.name == "nt" else "data-processor-core"
    candidates = [
        repo_root / "target" / "release" / exe_name,
        repo_root / "target" / "debug" / exe_name,
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)

