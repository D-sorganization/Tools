#!/usr/bin/env python3
"""
Fix the remaining line length issues.
"""

from pathlib import Path


def fix_line_lengths():
    """Fix specific line length issues."""

    # Fix Data_Processor_Integrated.py
    file_path = Path(
        "data_processing/data_processor/python/data_processor/Data_Processor_Integrated.py"
    )
    if file_path.exists():
        content = file_path.read_text(encoding="utf-8")

        # Fix long docstring
        content = content.replace(
            "analyzing, and visualizing time series data from CSV files and DAT files with DBF tag files.",
            "analyzing, and visualizing time series data from CSV files\n# and DAT files with DBF tag files.",
        )

        file_path.write_text(content, encoding="utf-8")
        print(f"Fixed line lengths in {file_path}")

    # Fix Data_Processor_r0.py
    file_path = Path(
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
    )
    if file_path.exists():
        content = file_path.read_text(encoding="utf-8")

        # Fix long docstring
        content = content.replace(
            "processing, analyzing, and visualizing time series data from CSV files and DAT files with DBF tag files.",
            "processing, analyzing, and visualizing time series data from CSV files\n# and DAT files with DBF tag files.",
        )

        file_path.write_text(content, encoding="utf-8")
        print(f"Fixed line lengths in {file_path}")

    # Fix matlab_quality_check.py
    file_path = Path(
        "media_processing/video_processor/tools/matlab_utilities/scripts/matlab_quality_check.py"
    )
    if file_path.exists():
        content = file_path.read_text(encoding="utf-8")

        # Fix long comment
        content = content.replace(
            "This is the unified version combining the best features from all repository implementations.",
            "This is the unified version combining the best features from all\n# repository implementations.",
        )

        file_path.write_text(content, encoding="utf-8")
        print(f"Fixed line lengths in {file_path}")


if __name__ == "__main__":
    fix_line_lengths()
    print("✅ Line length fixes completed!")
