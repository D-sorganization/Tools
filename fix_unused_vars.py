#!/usr/bin/env python3
"""
Fix unused loop control variables
"""

import re
from pathlib import Path


def fix_unused_vars(file_path):
    """Fix unused loop control variables"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Fix patterns
    patterns = [
        (
            r"\bfor root, _dirs, files in os\.walk\(",
            r"for _root, _dirs, files in os.walk(",
        ),
        (
            r"\bfor signal, data in self\.signal_vars\.items\(\):",
            r"for _signal, data in self.signal_vars.items():",
        ),
        (
            r"\bfor base_name, file_list in files_by_base_name\.items\(\):",
            r"for _base_name, file_list in files_by_base_name.items():",
        ),
    ]

    new_content = content
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, new_content)

    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed unused variables in {file_path}")
        return True
    return False


# Fix the problematic files
files_to_fix = [
    "data_processing/data_processor/python/data_processor/Data_Processor_Integrated.py",
    "data_processing/data_processor/python/data_processor/Data_Processor_r0.py",
]

for file_path in files_to_fix:
    path = Path(file_path)
    if path.exists():
        fix_unused_vars(path)
