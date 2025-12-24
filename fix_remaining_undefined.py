#!/usr/bin/env python3
"""
Fix remaining undefined name errors
"""

import re
from pathlib import Path


def fix_undefined_names(file_path):
    """Fix undefined name errors in the specified file"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Fix patterns for root variable
    patterns = [
        # Fix os.path.join(root, ...) where root should be _root
        (r"os\.path\.join\(root,", r"os.path.join(_root,"),
        # Fix os.path.relpath(root, ...) where root should be _root
        (r"os\.path\.relpath\(root,", r"os.path.relpath(_root,"),
        # Fix signal variable in loops
        (
            r"if search_text in signal\.lower\(\):",
            r"if search_text in _signal.lower():",
        ),
        (r"if signal in present_signals:", r"if _signal in present_signals:"),
    ]

    new_content = content
    changes_made = False

    for pattern, replacement in patterns:
        old_content = new_content
        new_content = re.sub(pattern, replacement, new_content)
        if new_content != old_content:
            changes_made = True

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed undefined names in {file_path}")
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
        fix_undefined_names(path)
