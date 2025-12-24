#!/usr/bin/env python3
"""
Fix specific long lines in Data_Processor_r0.py
"""

import re
from pathlib import Path


def fix_data_processor_long_lines():
    """Fix specific long lines in the data processor file"""
    file_path = Path("data_processing/data_processor/python/data_processor/Data_Processor_r0.py")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    # Fix specific long lines with targeted replacements
    replacements = [
        # Long debug print statements
        (
            r'f"DEBUG: Early return - saved_signal_list: {bool\(self\.saved_signal_list\)}, signal_vars: {bool\(self\.signal_vars\)}"',
            r'(\n                f"DEBUG: Early return - saved_signal_list: {bool(self.saved_signal_list)}, "\n                f"signal_vars: {bool(self.signal_vars)}"\n            )'
        ),

        # Long status label configurations
        (
            r'text=f"Signal list saved: {signal_list_name} \({len\(selected_signals\)} signals\)"',
            r'text=(\n                    f"Signal list saved: {signal_list_name} "\n                    f"({len(selected_signals)} signals)"\n                )'
        ),

        # Long messagebox calls
        (
            r'"scipy\.signal\.savgol_filter unavailable\. Install SciPy or skip smoothing\."',
            r'(\n                                "scipy.signal.savgol_filter unavailable. "\n                                "Install SciPy or skip smoothing."\n                            )'
        ),

        # Long configuration messages
        (
            r'"Are you sure you want to delete this configuration file\?\\n\\n{filename}\\n\\nThis action cannot be undone\."',
            r'(\n                f"Are you sure you want to delete this configuration file?\\n\\n"\n                f"{filename}\\n\\nThis action cannot be undone."\n            )'
        ),

        # Long plot title generation
        (
            r'or f"Signals from {selected_file} \(Time Range: {start_time_str} - {end_time_str}\)"',
            r'or (\n                    f"Signals from {selected_file} "\n                    f"(Time Range: {start_time_str} - {end_time_str})"\n                )'
        ),

        # Long help text
        (
            r'"Filter settings from the plot tab have been applied to the main processing configuration\."',
            r'(\n            "Filter settings from the plot tab have been applied "\n            "to the main processing configuration."\n        )'
        ),
    ]

    original_content = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed long lines in {file_path}")
        return True
    else:
        print("No changes made")
        return False


if __name__ == "__main__":
    fix_data_processor_long_lines()
