#!/usr/bin/env python3
"""
Fix remaining line-too-long errors systematically
"""

import re
from pathlib import Path


def fix_data_processor_lines():
    """Fix specific long lines in Data_Processor_r0.py"""
    file_path = Path(
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
    )

    if not file_path.exists():
        return False

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Define specific replacements for long lines
    replacements = [
        # Long docstrings
        (
            r'"""Combine multiple processed files into a single dataset for time series data\."""',
            r'"""Combine multiple processed files into a single dataset for time series data."""',
        ),
        # Long print statements with time ranges
        (
            r'f"Time range: {combined_df\[time_col\]\.min\(\)} to {combined_df\[time_col\]\.max\(\)}",',
            r'(\n            f"Time range: {combined_df[time_col].min()} to "\n            f"{combined_df[time_col].max()}",\n        )',
        ),
        # Long scipy error messages
        (
            r'"scipy\.signal\.savgol_filter unavailable\. Install SciPy or skip smoothing\.",',
            r'(\n                                "scipy.signal.savgol_filter unavailable. "\n                                "Install SciPy or skip smoothing.",\n                            )',
        ),
        # Long docstring for plotting function
        (
            r'"""Get data for plotting from the specified file - simplified baseline approach\."""',
            r'"""Get data for plotting from the specified file - simplified baseline approach."""',
        ),
        # Long debug print statements
        (
            r'f"processed_files: {len\(getattr\(self, \'processed_files\', {}\)\) if hasattr\(self, \'processed_files\'\) else \'None\'}",',
            r'(\n            f"processed_files: {len(getattr(self, \'processed_files\', {})) "\n            f"if hasattr(self, \'processed_files\') else \'None\'}",\n        )',
        ),
        # Long comment about structure checking
        (
            r"# Check if it has the expected structure \(processing configs have \'saved_at\', plotting configs have \'plot_name\'\)",
            r"(\n        # Check if it has the expected structure (processing configs have \'saved_at\',\n        # plotting configs have \'plot_name\')\n        )",
        ),
        # Long comment about skipping files
        (
            r"# Skip files that can\'t be read as JSON or don\'t have the right structure",
            r"(\n                        # Skip files that can\'t be read as JSON or\n                        # don\'t have the right structure\n                        )",
        ),
    ]

    original_content = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed long lines in {file_path}")
        return True
    else:
        print("No automatic fixes applied")
        return False


if __name__ == "__main__":
    fix_data_processor_lines()
