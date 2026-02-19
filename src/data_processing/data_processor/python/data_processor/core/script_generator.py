# mypy: ignore-errors
"""Script Generation System for Automated Processing Pipelines.

Provides functionality to:
- Record processing operations as reproducible scripts
- Generate Python scripts for batch processing
- Create CLI commands for automation
- Export processing configurations for CI/CD pipelines

Supports consistent data processing workflows that can be
called programmatically for automation.

This module serves as a facade, composing the following submodules:
- script_generator_types: Data models (types, steps, pipelines)
- pipeline_recorder: PipelineRecorder for recording operations
- pipeline_executor: PipelineExecutor for running pipelines
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

# Re-export all public types for backward compatibility
from data_processor.core.pipeline_executor import PipelineExecutor  # noqa: F401
from data_processor.core.pipeline_recorder import PipelineRecorder  # noqa: F401
from data_processor.core.script_generator_types import (  # noqa: F401
    OperationType,
    ProcessingPipeline,
    ProcessingStep,
)

logger = logging.getLogger(__name__)


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
        lines: list[str] = []

        lines.extend(self._generate_script_header(pipeline))

        if include_imports:
            lines.extend(
                self._generate_imports(pipeline, include_logging, use_argparse)
            )

        if include_logging:
            lines.extend(self._generate_logging_setup())

        lines.extend(self._generate_process_function(pipeline))

        if use_argparse:
            lines.extend(self._generate_argparse(pipeline))

        lines.extend(self._generate_main_block(use_argparse))

        script = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(script)
            logger.info(f"Generated script: {output_path}")

        return script

    @staticmethod
    def _generate_script_header(pipeline: ProcessingPipeline) -> list[str]:
        """Generate the module docstring header."""
        return [
            '"""',
            f"Data Processing Script: {pipeline.name}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            f"Description: {pipeline.description}",
            '"""',
            "",
        ]

    @staticmethod
    def _generate_logging_setup() -> list[str]:
        """Generate logging configuration lines."""
        return [
            "",
            "# Logging setup",
            "logging.basicConfig(",
            "    level=logging.INFO,",
            "    format='%(asctime)s - %(levelname)s - %(message)s'",
            ")",
            "logger = logging.getLogger(__name__)",
            "",
        ]

    def _generate_process_function(self, pipeline: ProcessingPipeline) -> list[str]:
        """Generate the main process_data function."""
        lines = [
            "",
            "def process_data(",
            "    input_path: str,",
            "    output_path: str,",
            "    **kwargs",
            ") -> pd.DataFrame:",
            '    """Process data according to the defined pipeline."""',
            "",
        ]

        for i, step in enumerate(pipeline.steps):
            if not step.enabled:
                lines.append(f"    # Step {i+1} (disabled): {step.description}")
                continue

            lines.append(f"    # Step {i+1}: {step.description}")
            lines.extend(self._generate_step_code(step, indent=4))
            lines.append("")

        lines.extend(["    return df", ""])
        return lines

    @staticmethod
    def _generate_main_block(use_argparse: bool) -> list[str]:
        """Generate the if __name__ == '__main__' block."""
        lines = ["", "if __name__ == '__main__':"]
        if use_argparse:
            lines.extend([
                "    args = parse_args()",
                "    result = process_data(",
                "        input_path=args.input,",
                "        output_path=args.output,",
                "    )",
                "    logger.info(f'Processing complete. Output shape: {result.shape}')",  # noqa: E501
            ])
        else:
            lines.extend([
                "    # Configure paths",
                "    INPUT_PATH = 'input.csv'",
                "    OUTPUT_PATH = 'output.csv'",
                "",
                "    result = process_data(INPUT_PATH, OUTPUT_PATH)",
                "    print(f'Processing complete. Output shape: {result.shape}')",
            ])
        return lines

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
        lines: list[str] = []
        lines.extend(self._generate_batch_header(pipeline))
        lines.extend(self._generate_batch_process_func(pipeline))
        lines.extend(
            self._generate_batch_main(input_patterns, output_dir, parallel)
        )
        return "\n".join(lines)

    @staticmethod
    def _generate_batch_header(pipeline: ProcessingPipeline) -> list[str]:
        """Generate header and imports for batch script."""
        return [
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

    def _generate_batch_process_func(
        self, pipeline: ProcessingPipeline
    ) -> list[str]:
        """Generate the process_single_file function for batch script."""
        lines = [
            "def process_single_file(input_path: str, output_dir: str) -> str:",
            '    """Process a single file."""',
            "    try:",
            "        df = pd.read_csv(input_path)",
            "",
        ]

        for step in pipeline.steps:
            if step.enabled:
                lines.extend(self._generate_step_code(step, indent=8))

        lines.extend([
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
        ])
        return lines

    @staticmethod
    def _generate_batch_main(
        input_patterns: list[str], output_dir: str, parallel: bool
    ) -> list[str]:
        """Generate the main function and entry point for batch script."""
        lines = [
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

        if parallel:
            lines.extend([
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
            ])
        else:
            lines.extend([
                "    # Process files sequentially",
                "    for input_file in input_files:",
                "        result = process_single_file(input_file, output_dir)",
                "        if result:",
                "            print(f'Processed: {input_file} -> {result}')",
            ])

        lines.extend(["", "if __name__ == '__main__':", "    main()"])
        return lines

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
        imports: list[str] = [
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
        lines: list[str] = []
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


def _get_all_exports() -> list[str]:
    """Return the list of all public exports."""
    return [
        "OperationType",
        "ProcessingStep",
        "ProcessingPipeline",
        "PipelineRecorder",
        "ScriptGenerator",
        "PipelineExecutor",
    ]


__all__ = _get_all_exports()
