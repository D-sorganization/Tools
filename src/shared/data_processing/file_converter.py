from __future__ import annotations

from pathlib import Path
from typing import Callable, Set

import pandas as pd

from .file_io import DataReader, DataWriter, FileFormatDetector
from .models import SplitConfig


class FileConverter:
    """Handles file conversion logic."""

    def __init__(
        self,
        log_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[float], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.cancel_flag = False

    def log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    def update_progress(self, value: float) -> None:
        if self.progress_callback:
            self.progress_callback(value)

    def update_status(self, message: str) -> None:
        if self.status_callback:
            self.status_callback(message)

    def convert_files(
        self,
        input_files: list[str | Path],
        output_directory: str | Path,
        output_format: str,
        combine_files: bool = True,
        selected_columns: Set[str] | None = None,
        use_all_columns: bool = True,
        split_config: SplitConfig | None = None,
    ) -> int:
        """
        Convert files to the specified format.

        Returns:
            int: Number of files processed
        """
        try:
            self.update_status("Converting files...")
            self.update_progress(0)

            total_files = len(input_files)
            processed_files = 0

            # Ensure output directory exists
            Path(output_directory).mkdir(parents=True, exist_ok=True)

            if combine_files:
                # Combine all files into one
                self.log(
                    f"Starting conversion: combining {total_files} files into "
                    f"{output_format.upper()}"
                )

                combined_data = []
                for i, file_path in enumerate(input_files):
                    if self.cancel_flag:
                        break

                    try:
                        format_type = FileFormatDetector.detect_format(file_path)
                        if not format_type:
                            self.log(
                                f"Warning: Could not detect format for "
                                f"{Path(file_path).name}"
                            )
                            continue

                        df = DataReader.read_file(file_path, format_type)

                        # Apply column selection
                        if not use_all_columns and selected_columns:
                            available_columns = [
                                col for col in selected_columns if col in df.columns
                            ]
                            if available_columns:
                                df = df[available_columns]
                            else:
                                self.log(
                                    f"Warning: No selected columns found in "
                                    f"{Path(file_path).name}"
                                )
                                continue

                        combined_data.append(df)
                        self.log(
                            f"Loaded {Path(file_path).name}: {len(df)} rows, "
                            f"{len(df.columns)} columns"
                        )

                        processed_files += 1
                        self.update_progress(processed_files / total_files)

                    except Exception as e:
                        self.log(f"Error reading {Path(file_path).name}: {str(e)}")

                if combined_data:
                    try:
                        combined_df = pd.concat(combined_data, ignore_index=True)
                        output_filename = self._generate_output_filename(
                            output_format, "combined_data"
                        )
                        output_path = Path(output_directory) / output_filename

                        DataWriter.write_file(combined_df, output_path, output_format)
                        self.log(f"Successfully created: {output_filename}")
                        self.log(
                            f"Combined data: {len(combined_df)} rows, "
                            f"{len(combined_df.columns)} columns"
                        )

                    except Exception as e:
                        self.log(f"Error writing combined file: {str(e)}")
                else:
                    self.log("No valid data to combine")

            else:
                # Process files individually
                self.log(
                    f"Starting conversion: processing {total_files} files individually"
                )

                for i, file_path in enumerate(input_files):
                    if self.cancel_flag:
                        break

                    try:
                        format_type = FileFormatDetector.detect_format(file_path)
                        if not format_type:
                            self.log(
                                f"Warning: Could not detect format for "
                                f"{Path(file_path).name}"
                            )
                            continue

                        df = DataReader.read_file(file_path, format_type)

                        # Apply column selection
                        if not use_all_columns and selected_columns:
                            available_columns = [
                                col for col in selected_columns if col in df.columns
                            ]
                            if available_columns:
                                df = df[available_columns]
                            else:
                                self.log(
                                    f"Warning: No selected columns found in "
                                    f"{Path(file_path).name}"
                                )
                                continue

                        # Generate output filename
                        base_name = Path(file_path).stem
                        output_filename = self._generate_output_filename(
                            output_format, base_name
                        )
                        output_path = Path(output_directory) / output_filename

                        DataWriter.write_file(df, output_path, output_format)
                        self.log(
                            f"Converted {Path(file_path).name} -> {output_filename}"
                        )

                        processed_files += 1
                        self.update_progress(processed_files / total_files)

                    except Exception as e:
                        self.log(f"Error converting {Path(file_path).name}: {str(e)}")

            self.update_status(
                f"Conversion complete. {processed_files} files processed."
            )
            self.update_progress(1.0)
            return processed_files

        except Exception as e:
            self.log(f"Conversion error: {str(e)}")
            self.update_status("Conversion failed")
            raise

    def _generate_output_filename(
        self, output_format: str, base_name: str | None = None
    ) -> str:
        """Generate output filename with proper extension."""
        if not base_name:
            base_name = "converted_data"

        extensions = {
            "parquet": ".parquet",
            "csv": ".csv",
            "tsv": ".tsv",
            "excel": ".xlsx",
            "json": ".json",
            "hdf5": ".h5",
            "pickle": ".pkl",
            "numpy": ".npy",
            "matlab": ".mat",
            "feather": ".feather",
            "arrow": ".arrow",
            "sqlite": ".db",
        }

        extension = extensions.get(output_format, ".txt")
        return f"{base_name}{extension}"
