#!/usr/bin/env python3
"""
Fix line-too-long errors systematically
"""

import re
from pathlib import Path


def fix_long_lines_in_file(file_path):
    """Fix line-too-long errors in the specified file"""
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    changes_made = False

    for line in lines:
        original_line = line

        # Skip if line is already short enough
        if len(line.rstrip()) <= 88:
            new_lines.append(line)
            continue

        # Common patterns to fix
        fixed_line = line

        # Pattern 1: Long f-strings with multiple parts
        if 'f"' in fixed_line and len(fixed_line.rstrip()) > 88:"
            # Split long f-strings at logical points
            if " - " in fixed_line and 'f"' in fixed_line:"
                # Split at " - " in f-strings
                fixed_line = re.sub(
                    r'f"([^"]*) - ([^"]*)"',
                    r'f"\1 - "\n                f"\2"',
                    fixed_line,
                )
            elif ": " in fixed_line and 'f"' in fixed_line:"
                # Split at ": " in f-strings
                fixed_line = re.sub(
                    r'f"([^"]*): ([^"]*)"',
                    r'f"\1: "\n                f"\2"',
                    fixed_line,
                )

        # Pattern 2: Long messagebox calls
        if "messagebox." in fixed_line and len(fixed_line.rstrip()) > 88:
            # Split messagebox text arguments
            fixed_line = re.sub(
                r'messagebox\.(
                    showinfo|showwarning|askyesno)\(\s*"([^"]*)",
                    \s*"([^"]*
                )"',
                r'messagebox.\1(\n                "\2",\n                "\3"',
                fixed_line,
            )

        # Pattern 3: Long comments
        if fixed_line.strip().startswith("#") and len(fixed_line.rstrip()) > 88:
            # Split long comments at word boundaries
            comment_match = re.match(r"(\s*#\s*)(.*)", fixed_line)
            if comment_match:
                indent, comment_text = comment_match.groups()
                if len(comment_text) > 80:
                    # Find a good split point
                    words = comment_text.split()
                    if len(words) > 1:
                        mid_point = len(words) // 2
                        first_part = " ".join(words[:mid_point])
                        second_part = " ".join(words[mid_point:])
                        fixed_line = f"{indent}{first_part}\n{indent}{second_part}\n"

        # Pattern 4: Long string literals
        if '"""' in fixed_line and len(fixed_line.rstrip()) > 88:
            # Split long docstrings
            fixed_line = re.sub(r'"""([^"]{60,})"""', r'"""\1\n        """', fixed_line)

        # Pattern 5: Long function calls with multiple arguments
        if "(" in fixed_line and ")" in fixed_line and len(fixed_line.rstrip()) > 88:
            # Split function calls at commas
            if fixed_line.count(",") >= 2:
                # Find function call pattern
                func_match = re.match(r"(\s*)([^(]+\([^,]+),(.+)\)", fixed_line)
                if func_match:
                    indent, func_start, remaining = func_match.groups()
                    fixed_line = f"{func_start},\n{indent}    {remaining.strip()}\n"

        if fixed_line != original_line:
            changes_made = True

        new_lines.append(fixed_line)

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"Fixed long lines in {file_path}")
        return True
    return False


# Focus on the main problematic file first
main_file = "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
if Path(main_file).exists():
    fix_long_lines_in_file(main_file)
