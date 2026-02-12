"""Script Generation System for Automated Processing Pipelines.

Provides functionality to:
- Record processing operations as reproducible scripts
- Generate Python scripts for batch processing
- Create CLI commands for automation
- Export processing configurations for CI/CD pipelines

Supports consistent data processing workflows that can be
called programmatically for automation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of data processing operations."""

    LOAD = "load"
    FILTER = "filter"
    TRANSFORM = "transform"
    CALCULATE = "calculate"
    RESAMPLE = "resample"
    INTEGRATE = "integrate"
    DIFFERENTIATE = "differentiate"
    TRIM = "trim"
    MERGE = "merge"
    SELECT = "select"
    RENAME = "rename"
    EXPORT = "export"
    CUSTOM = "custom"


@dataclass
class ProcessingStep:
    """A single processing operation."""

    operation: OperationType
    parameters: dict[str, Any]
    description: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation.value,
            "parameters": self.parameters,
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingStep:
        """Create from dictionary."""
        return cls(
            operation=OperationType(data["operation"]),
            parameters=data["parameters"],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
        )


@dataclass
class ProcessingPipeline:
    """A complete processing pipeline with multiple steps."""

    name: str
    description: str = ""
    steps: list[ProcessingStep] = field(default_factory=list)
    input_config: dict[str, Any] = field(default_factory=dict)
    output_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        operation: OperationType,
        parameters: dict[str, Any],
        description: str = "",
    ) -> ProcessingStep:
        """Add a processing step to the pipeline."""
        step = ProcessingStep(
            operation=operation,
            parameters=parameters,
            description=description,
        )
        self.steps.append(step)
        return step

    def remove_step(self, index: int) -> ProcessingStep | None:
        """Remove a step by index."""
        if 0 <= index < len(self.steps):
            return self.steps.pop(index)
        return None

    def move_step(self, from_index: int, to_index: int) -> bool:
        """Move a step from one position to another."""
        if 0 <= from_index < len(self.steps) and 0 <= to_index < len(self.steps):
            step = self.steps.pop(from_index)
            self.steps.insert(to_index, step)
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "input_config": self.input_config,
            "output_config": self.output_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingPipeline:
        """Create from dictionary."""
        steps = [ProcessingStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            input_config=data.get("input_config", {}),
            output_config=data.get("output_config", {}),
            metadata=data.get("metadata", {}),
        )


class PipelineRecorder:
    """Records data processing operations into a pipeline."""

    def __init__(self, pipeline_name: str = "Untitled Pipeline") -> None:
        """Initialize the recorder."""
        self._pipeline = ProcessingPipeline(name=pipeline_name)
        self._recording = True

    @property
    def pipeline(self) -> ProcessingPipeline:
        """Get the current pipeline."""
        return self._pipeline

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._recording

    def start_recording(self) -> None:
        """Start recording operations."""
        self._recording = True

    def stop_recording(self) -> None:
        """Stop recording operations."""
        self._recording = False

    def clear(self) -> None:
        """Clear all recorded steps."""
        self._pipeline.steps.clear()

    def record_load(
        self,
        file_path: str,
        file_format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Record a file load operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.LOAD,
            parameters={
                "file_path": file_path,
                "file_format": file_format,
                "options": options or {},
            },
            description=f"Load data from {file_path}",
        )

    def record_filter(
        self,
        filter_type: str,
        parameters: dict[str, Any],
        signals: list[str] | None = None,
    ) -> None:
        """Record a filter operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.FILTER,
            parameters={
                "filter_type": filter_type,
                "filter_params": parameters,
                "signals": signals,
            },
            description=f"Apply {filter_type} filter",
        )

    def record_transform(
        self,
        transform_type: str,
        parameters: dict[str, Any],
    ) -> None:
        """Record a transformation operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.TRANSFORM,
            parameters={
                "transform_type": transform_type,
                **parameters,
            },
            description=f"Apply {transform_type} transformation",
        )

    def record_calculate(
        self,
        column_name: str,
        formula: str,
    ) -> None:
        """Record a calculated column operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.CALCULATE,
            parameters={
                "column_name": column_name,
                "formula": formula,
            },
            description=f"Calculate {column_name} = {formula}",
        )

    def record_resample(
        self,
        time_column: str,
        rule: str,
        method: str = "mean",
    ) -> None:
        """Record a resampling operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.RESAMPLE,
            parameters={
                "time_column": time_column,
                "rule": rule,
                "method": method,
            },
            description=f"Resample to {rule} using {method}",
        )

    def record_integrate(
        self,
        time_column: str,
        signals: list[str],
        method: str = "trapezoidal",
    ) -> None:
        """Record an integration operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.INTEGRATE,
            parameters={
                "time_column": time_column,
                "signals": signals,
                "method": method,
            },
            description=f"Integrate signals using {method}",
        )

    def record_differentiate(
        self,
        time_column: str,
        signals: list[str],
        method: str = "spline",
        orders: list[int] | None = None,
    ) -> None:
        """Record a differentiation operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.DIFFERENTIATE,
            parameters={
                "time_column": time_column,
                "signals": signals,
                "method": method,
                "orders": orders or [1],
            },
            description=f"Differentiate signals using {method}",
        )

    def record_trim(
        self,
        time_column: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """Record a time range trim operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.TRIM,
            parameters={
                "time_column": time_column,
                "start_time": start_time,
                "end_time": end_time,
            },
            description=(
                f"Trim time range: {start_time or 'start'} "
                f"to {end_time or 'end'}"
            ),
        )

    def record_select(
        self,
        columns: list[str],
    ) -> None:
        """Record a column selection operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.SELECT,
            parameters={"columns": columns},
            description=f"Select {len(columns)} columns",
        )

    def record_export(
        self,
        file_path: str,
        file_format: str = "csv",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Record an export operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.EXPORT,
            parameters={
                "file_path": file_path,
                "file_format": file_format,
                "options": options or {},
            },
            description=f"Export to {file_path}",
        )

    def record_custom(
        self,
        operation_name: str,
        parameters: dict[str, Any],
        description: str = "",
    ) -> None:
        """Record a custom operation."""
        if not self._recording:
            return

        self._pipeline.add_step(
            operation=OperationType.CUSTOM,
            parameters={
                "operation_name": operation_name,
                **parameters,
            },
            description=description or f"Custom operation: {operation_name}",
        )


class ScriptGenerator:
    """Generates executable scripts from processing pipelines."""

    def __init__(self) -> None:
        """Initialize the script generator."""
        self._templates: dict[str, str] = {}

    def generate_python_script(
        self,
        pipeline: ProcessingPipeline,
        output_path: Path | str | None = None,
        include_imports: bool = True,
        include_logging: bool = True,
        use_argparse: bool = True,
    ) -> str:
        """Generate a Python script from a pipeline.

        Args:
            pipeline: Processing pipeline
            output_path: Optional path to write script
            include_imports: Include import statements
            include_logging: Include logging setup
            use_argparse: Include argparse for CLI arguments

        Returns:
            Generated script as string
        """
        lines = []

        # Header
        lines.extend(
            [
                '"""',
                f"Data Processing Script: {pipeline.name}",
                f"Generated: {datetime.now().isoformat()}",
                "",
                f"Description: {pipeline.description}",
                '"""',
                "",
            ]
        )

        # Imports
        if include_imports:
            lines.extend(
                self._generate_imports(pipeline, include_logging, use_argparse)
            )

        # Logging setup
        if include_logging:
            lines.extend(
                [
                    "",
                    "# Logging setup",
                    "logging.basicConfig(",
                    "    level=logging.INFO,",
                    "    format='%(asctime)s - %(levelname)s - %(message)s'",
                    ")",
                    "logger = logging.getLogger(__name__)",
                    "",
                ]
            )

        # Main processing function
        lines.extend(
            [
                "",
                "def process_data(",
                "    input_path: str,",
                "    output_path: str,",
                "    **kwargs",
                ") -> pd.DataFrame:",
                '    """Process data according to the defined pipeline."""',
                "",
            ]
        )

        # Generate step code
        for i, step in enumerate(pipeline.steps):
            if not step.enabled:
                lines.append(f"    # Step {i+1} (disabled): {step.description}")
                continue

            lines.append(f"    # Step {i+1}: {step.description}")
            step_code = self._generate_step_code(step, indent=4)
            lines.extend(step_code)
            lines.append("")

        lines.extend(
            [
                "    return df",
                "",
            ]
        )

        # Argparse
        if use_argparse:
            lines.extend(self._generate_argparse(pipeline))

        # Main block
        lines.extend(
            [
                "",
                "if __name__ == '__main__':",
            ]
        )

        if use_argparse:
            lines.extend(
                [
                    "    args = parse_args()",
                    "    result = process_data(",
                    "        input_path=args.input,",
                    "        output_path=args.output,",
                    "    )",
                    "    logger.info(f'Processing complete. Output shape: {result.shape}')",  # noqa: E501
                ]
            )
        else:
            lines.extend(
                [
                    "    # Configure paths",
                    "    INPUT_PATH = 'input.csv'",
                    "    OUTPUT_PATH = 'output.csv'",
                    "",
                    "    result = process_data(INPUT_PATH, OUTPUT_PATH)",
                    "    print(f'Processing complete. Output shape: {result.shape}')",
                ]
            )

        script = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(script)
            logger.info(f"Generated script: {output_path}")

        return script

    def generate_cli_command(
        self,
        pipeline: ProcessingPipeline,
        input_path: str,
        output_path: str,
    ) -> str:
        """Generate a CLI command to run the pipeline.

        Args:
            pipeline: Processing pipeline
            input_path: Input file path
            output_path: Output file path

        Returns:
            CLI command string
        """
        # Generate JSON config
        config = pipeline.to_dict()
        config_json = json.dumps(config)

        return (
            f"python -m data_processor.cli run "
            f"--input '{input_path}' "
            f"--output '{output_path}' "
            f"--config '{config_json}'"
        )

    def generate_batch_script(
        self,
        pipeline: ProcessingPipeline,
        input_patterns: list[str],
        output_dir: str,
        parallel: bool = True,
    ) -> str:
        """Generate a batch processing script.

        Args:
            pipeline: Processing pipeline
            input_patterns: Glob patterns for input files
            output_dir: Output directory
            parallel: Use parallel processing

        Returns:
            Batch processing script
        """
        lines = [
            '"""',
            f"Batch Processing Script: {pipeline.name}",
            f"Generated: {datetime.now().isoformat()}",
            '"""',
            "",
            "import glob",
            "import os",
            "from pathlib import Path",
            "from concurrent.futures import ProcessPoolExecutor, as_completed",
            "import pandas as pd",
            "",
            "# Import processing functions",
            "from data_processor.core import signal_processing",
            "from data_processor import vectorized_filter_engine",
            "",
        ]

        # Add the process_single_file function
        lines.extend(
            [
                "def process_single_file(input_path: str, output_dir: str) -> str:",
                '    """Process a single file."""',
                "    try:",
                "        df = pd.read_csv(input_path)",
                "",
            ]
        )

        # Add processing steps
        for step in pipeline.steps:
            if not step.enabled:
                continue
            step_code = self._generate_step_code(step, indent=8)
            lines.extend(step_code)

        lines.extend(
            [
                "",
                "        # Save output",
                "        output_name = Path(input_path).stem + '_processed.csv'",
                "        output_path = os.path.join(output_dir, output_name)",
                "        df.to_csv(output_path, index=False)",
                "        return output_path",
                "    except Exception as e:",
                "        print(f'Error processing {input_path}: {e}')",
                "        return None",
                "",
            ]
        )

        # Main function
        lines.extend(
            [
                "def main():",
                f"    input_patterns = {input_patterns}",
                f"    output_dir = '{output_dir}'",
                "",
                "    # Ensure output directory exists",
                "    os.makedirs(output_dir, exist_ok=True)",
                "",
                "    # Collect input files",
                "    input_files = []",
                "    for pattern in input_patterns:",
                "        input_files.extend(glob.glob(pattern))",
                "",
                "    print(f'Found {len(input_files)} files to process')",
                "",
            ]
        )

        if parallel:
            lines.extend(
                [
                    "    # Process files in parallel",
                    "    with ProcessPoolExecutor() as executor:",
                    "        futures = {",
                    "            executor.submit(process_single_file, f, output_dir): f",  # noqa: E501
                    "            for f in input_files",
                    "        }",
                    "",
                    "        for future in as_completed(futures):",
                    "            input_file = futures[future]",
                    "            result = future.result()",
                    "            if result:",
                    "                print(f'Processed: {input_file} -> {result}')",
                ]
            )
        else:
            lines.extend(
                [
                    "    # Process files sequentially",
                    "    for input_file in input_files:",
                    "        result = process_single_file(input_file, output_dir)",
                    "        if result:",
                    "            print(f'Processed: {input_file} -> {result}')",
                ]
            )

        lines.extend(
            [
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )

        return "\n".join(lines)

    def export_pipeline_config(
        self,
        pipeline: ProcessingPipeline,
        output_path: Path | str,
    ) -> Path:
        """Export pipeline configuration to JSON.

        Args:
            pipeline: Processing pipeline
            output_path: Output file path

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        config = pipeline.to_dict()
        config["generated_at"] = datetime.now().isoformat()

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        return output_path

    def import_pipeline_config(self, config_path: Path | str) -> ProcessingPipeline:
        """Import pipeline configuration from JSON.

        Args:
            config_path: Path to configuration file

        Returns:
            ProcessingPipeline object
        """
        with open(config_path) as f:
            config = json.load(f)

        return ProcessingPipeline.from_dict(config)

    def _generate_imports(
        self,
        pipeline: ProcessingPipeline,
        include_logging: bool,
        use_argparse: bool,
    ) -> list[str]:
        """Generate import statements based on pipeline operations."""
        imports = [
            "from __future__ import annotations",
            "",
            "import pandas as pd",
            "import numpy as np",
        ]

        if include_logging:
            imports.append("import logging")

        if use_argparse:
            imports.append("import argparse")

        # Add imports based on operations used
        operations = {s.operation for s in pipeline.steps}

        if OperationType.FILTER in operations:
            imports.append(
                "from data_processor.vectorized_filter_engine"
                " import VectorizedFilterEngine"
            )

        if (
            OperationType.INTEGRATE in operations
            or OperationType.DIFFERENTIATE in operations
        ):
            imports.append(
                "from data_processor.core.signal_processing"
                " import integrate_signals, differentiate_signals"
            )

        if OperationType.RESAMPLE in operations:
            imports.append(
                "from data_processor.core.signal_processing import resample_data"
            )

        if OperationType.CALCULATE in operations:
            imports.append(
                "from data_processor.core.signal_processing"
                " import apply_custom_variable"
            )

        if OperationType.TRIM in operations:
            imports.append(
                "from data_processor.core.signal_processing import trim_time_range"
            )

        return imports

    def _generate_step_code(self, step: ProcessingStep, indent: int = 0) -> list[str]:
        """Generate Python code for a processing step."""
        prefix = " " * indent
        lines = []
        params = step.parameters

        if step.operation == OperationType.LOAD:
            file_path = params.get("file_path", "input_path")
            file_format = params.get("file_format", "csv")

            if file_format == "csv":
                lines.append(f"{prefix}df = pd.read_csv({file_path!r})")
            elif file_format in ("xlsx", "excel"):
                lines.append(f"{prefix}df = pd.read_excel({file_path!r})")
            elif file_format == "parquet":
                lines.append(f"{prefix}df = pd.read_parquet({file_path!r})")
            else:
                lines.append(f"{prefix}df = pd.read_csv({file_path!r})")

        elif step.operation == OperationType.FILTER:
            filter_type = params.get("filter_type")
            filter_params = params.get("filter_params", {})
            signals = params.get("signals")

            lines.append(f"{prefix}filter_engine = VectorizedFilterEngine()")
            if signals:
                lines.append(
                    f"{prefix}df = filter_engine.apply_filter_batch("
                    f"df, {filter_type!r}, {filter_params}, signal_names={signals})"
                )
            else:
                lines.append(
                    f"{prefix}df = filter_engine.apply_filter_batch("
                    f"df, {filter_type!r}, {filter_params})"
                )

        elif step.operation == OperationType.CALCULATE:
            col_name = params.get("column_name")
            formula = params.get("formula")
            lines.append(
                f"{prefix}df = apply_custom_variable(df, {col_name!r}, {formula!r})"
            )

        elif step.operation == OperationType.RESAMPLE:
            time_col = params.get("time_column")
            rule = params.get("rule")
            method = params.get("method", "mean")
            lines.append(
                f"{prefix}df = resample_data("
                f"df, {time_col!r}, {rule!r}, method={method!r})"
            )

        elif step.operation == OperationType.INTEGRATE:
            time_col = params.get("time_column")
            signals = params.get("signals")
            method = params.get("method", "trapezoidal")
            lines.append(
                f"{prefix}df = integrate_signals("
                f"df, {time_col!r}, {signals}, method={method!r})"
            )

        elif step.operation == OperationType.DIFFERENTIATE:
            time_col = params.get("time_column")
            signals = params.get("signals")
            method = params.get("method", "spline")
            orders = params.get("orders", [1])
            lines.append(
                f"{prefix}df = differentiate_signals(df, {time_col!r}, {signals}, "
                f"method={method!r}, orders={orders})"
            )

        elif step.operation == OperationType.TRIM:
            time_col = params.get("time_column")
            start = params.get("start_time")
            end = params.get("end_time")
            lines.append(
                f"{prefix}df = trim_time_range(df, {time_col!r}, "
                f"start_time={start!r}, end_time={end!r})"
            )

        elif step.operation == OperationType.SELECT:
            columns = params.get("columns", [])
            lines.append(f"{prefix}df = df[{columns}]")

        elif step.operation == OperationType.RENAME:
            mapping = params.get("mapping", {})
            lines.append(f"{prefix}df = df.rename(columns={mapping})")

        elif step.operation == OperationType.EXPORT:
            file_path = params.get("file_path", "output_path")
            file_format = params.get("file_format", "csv")

            if file_format == "csv":
                lines.append(f"{prefix}df.to_csv({file_path!r}, index=False)")
            elif file_format in ("xlsx", "excel"):
                lines.append(f"{prefix}df.to_excel({file_path!r}, index=False)")
            elif file_format == "parquet":
                lines.append(f"{prefix}df.to_parquet({file_path!r})")

        elif step.operation == OperationType.CUSTOM:
            op_name = params.get("operation_name", "custom_operation")
            lines.append(f"{prefix}# Custom operation: {op_name}")
            lines.append(f"{prefix}# Parameters: {params}")
            lines.append(f"{prefix}# TODO: Implement custom operation")

        return lines

    def _generate_argparse(self, pipeline: ProcessingPipeline) -> list[str]:
        """Generate argparse setup code."""
        return [
            "",
            "def parse_args():",
            '    """Parse command line arguments."""',
            f"    parser = argparse.ArgumentParser(description={pipeline.description!r})",  # noqa: E501
            "    parser.add_argument('--input', '-i', required=True, help='Input file path')",  # noqa: E501
            "    parser.add_argument('--output', '-o', required=True, help='Output file path')",  # noqa: E501
            "    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')",  # noqa: E501
            "    return parser.parse_args()",
        ]


class PipelineExecutor:
    """Executes processing pipelines on data."""

    def __init__(self) -> None:
        """Initialize the executor."""
        self._filter_engine = None

    def execute(
        self,
        pipeline: ProcessingPipeline,
        input_data: str | Path | pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Execute a pipeline on input data.

        Args:
            pipeline: Processing pipeline to execute
            input_data: Input file path or DataFrame
            output_path: Optional output file path

        Returns:
            Processed DataFrame
        """
        import pandas as pd

        # Load data if path provided
        if isinstance(input_data, (str, Path)):
            df = pd.read_csv(input_data)
        else:
            df = input_data.copy()

        # Execute each step
        for i, step in enumerate(pipeline.steps):
            if not step.enabled:
                logger.debug(f"Skipping disabled step {i+1}: {step.description}")
                continue

            logger.info(f"Executing step {i+1}: {step.description}")
            df = self._execute_step(df, step)

        # Export if output path provided
        if output_path:
            output_path = Path(output_path)
            suffix = output_path.suffix.lower()

            if suffix == ".csv":
                df.to_csv(output_path, index=False)
            elif suffix in (".xlsx", ".xls"):
                df.to_excel(output_path, index=False)
            elif suffix == ".parquet":
                df.to_parquet(output_path)
            else:
                df.to_csv(output_path, index=False)

            logger.info(f"Exported results to {output_path}")

        return df

    def _execute_step(self, df: pd.DataFrame, step: ProcessingStep) -> pd.DataFrame:
        """Execute a single processing step."""
        params = step.parameters

        if step.operation == OperationType.FILTER:
            from data_processor.vectorized_filter_engine import VectorizedFilterEngine

            if self._filter_engine is None:
                self._filter_engine = VectorizedFilterEngine()

            return self._filter_engine.apply_filter_batch(
                df,
                params.get("filter_type"),
                params.get("filter_params", {}),
                signal_names=params.get("signals"),
            )

        elif step.operation == OperationType.CALCULATE:
            from data_processor.core.signal_processing import apply_custom_variable

            return apply_custom_variable(
                df,
                params.get("column_name"),
                params.get("formula"),
            )

        elif step.operation == OperationType.RESAMPLE:
            from data_processor.core.signal_processing import resample_data

            return resample_data(
                df,
                params.get("time_column"),
                params.get("rule"),
                method=params.get("method", "mean"),
            )

        elif step.operation == OperationType.INTEGRATE:
            from data_processor.core.signal_processing import integrate_signals

            return integrate_signals(
                df,
                params.get("time_column"),
                params.get("signals"),
                method=params.get("method", "trapezoidal"),
            )

        elif step.operation == OperationType.DIFFERENTIATE:
            from data_processor.core.signal_processing import differentiate_signals

            return differentiate_signals(
                df,
                params.get("time_column"),
                params.get("signals"),
                method=params.get("method", "spline"),
                orders=params.get("orders", [1]),
            )

        elif step.operation == OperationType.TRIM:
            from data_processor.core.signal_processing import trim_time_range

            return trim_time_range(
                df,
                params.get("time_column"),
                start_time=params.get("start_time"),
                end_time=params.get("end_time"),
            )

        elif step.operation == OperationType.SELECT:
            return df[params.get("columns", [])]

        elif step.operation == OperationType.RENAME:
            return df.rename(columns=params.get("mapping", {}))

        else:
            logger.warning(f"Unknown operation: {step.operation}")
            return df


__all__ = [
    "OperationType",
    "ProcessingStep",
    "ProcessingPipeline",
    "PipelineRecorder",
    "ScriptGenerator",
    "PipelineExecutor",
]
