#!/usr/bin/env python3
"""
Fix line-too-long errors (E501) by breaking long lines appropriately
"""

import re
from pathlib import Path


def fix_long_lines(file_path):
    """Fix long lines in the specified file"""
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

        # Pattern 1: Long f-strings - break at logical points
        if 'f"' in line and len(line.rstrip()) > 88:
            # Break long f-strings at commas or logical separators
            if ", " in line and line.count('"') == 2:
                # Find a good break point
                parts = line.split(", ")
                if len(parts) > 1:
                    indent = len(line) - len(line.lstrip())
                    first_part = parts[0] + ","
                    remaining = ", ".join(parts[1:])
                    if len(first_part.rstrip()) <= 88:
                        new_line = first_part + "\n" + " " * (indent + 4) + remaining
                        new_lines.append(new_line)
                        changes_made = True
                        continue

        # Pattern 2: Long error messages - break at logical points
        if "raise RuntimeError(" in line and len(line.rstrip()) > 88:
            # Break long error messages
            match = re.match(r'(\s*raise RuntimeError\(\s*)"([^"]+)"', line)
            if match:
                indent_part = match.group(1)
                message = match.group(2)
                if len(message) > 50:  # If message is long, break it
                    new_line = (
                        indent_part
                        + "(\n"
                        + " " * (len(indent_part) + 4)
                        + f'"{message}"\n'
                        + " " * len(indent_part.rstrip())
                        + ")\n"
                    )
                    new_lines.append(new_line)
                    changes_made = True
                    continue

        # Pattern 3: Long comments - break at word boundaries
        if line.strip().startswith("#") and len(line.rstrip()) > 88:
            indent = len(line) - len(line.lstrip())
            comment_text = line.strip()[1:].strip()
            if len(comment_text) > 80:
                # Find a good break point around the middle
                words = comment_text.split()
                mid_point = len(words) // 2
                first_half = " ".join(words[:mid_point])
                second_half = " ".join(words[mid_point:])

                if len(first_half) < 80 and len(second_half) < 80:
                    new_line = (
                        " " * indent
                        + "# "
                        + first_half
                        + "\n"
                        + " " * indent
                        + "# "
                        + second_half
                        + "\n"
                    )
                    new_lines.append(new_line)
                    changes_made = True
                    continue

        # Pattern 4: Long string literals - break at logical points
        if '"""' in line and len(line.rstrip()) > 88:
            # Handle docstrings and multi-line strings
            if line.count('"""') == 2:  # Single line docstring
                indent = len(line) - len(line.lstrip())
                content = line.strip()[3:-3]
                if len(content) > 70:
                    new_line = (
                        " " * indent
                        + '"""\n'
                        + " " * indent
                        + content
                        + "\n"
                        + " " * indent
                        + '"""\n'
                    )
                    new_lines.append(new_line)
                    changes_made = True
                    continue

        # Pattern 5: Long function calls - break at parameters
        if "(" in line and ")" in line and ", " in line and len(line.rstrip()) > 88:
            # Try to break function calls at parameters
            if line.count("(") == 1 and line.count(")") == 1:
                func_match = re.match(r"(\s*)([^(]+\()([^)]+)(\).*)", line)
                if func_match:
                    indent_part = func_match.group(1)
                    func_start = func_match.group(2)
                    params = func_match.group(3)
                    func_end = func_match.group(4)

                    if ", " in params:
                        param_list = [p.strip() for p in params.split(", ")]
                        if len(param_list) > 1:
                            new_line = indent_part + func_start + "\n"
                            for i, param in enumerate(param_list):
                                if i == len(param_list) - 1:
                                    new_line += (
                                        " " * (len(indent_part) + 4) + param + "\n"
                                    )
                                else:
                                    new_line += (
                                        " " * (len(indent_part) + 4) + param + ",\n"
                                    )
                            new_line += indent_part + func_end + "\n"
                            new_lines.append(new_line)
                            changes_made = True
                            continue

        # If no pattern matched, keep the original line
        new_lines.append(line)

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"Fixed long lines in {file_path}")
        return True
    return False


# Get files with E501 errors
def get_files_with_long_lines():
    """Get list of files with line-too-long errors"""
    import subprocess

    result = subprocess.run(
        ["ruff", "check", ".", "--select", "E501", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        import json

        try:
            errors = json.loads(result.stdout)
            files = set()
            for error in errors:
                files.add(error["filename"])
            return list(files)
        except:
            # Fallback to known problematic files
            return [
                "data_processing/data_processor/python/data_processor/Data_Processor_r0.py",
                "media_processing/video_processor/python/tests/test_logger_utils_mock.py",
                "media_processing/video_processor/scripts/quality_check.py",
                "media_processing/video_processor/tools/code_quality_check.py",
                "media_processing/video_processor/tools/matlab_utilities/scripts/matlab_quality_check.py",
                "web_applications/unit_converter/tools/code_quality_check.py",
            ]
    return []


if __name__ == "__main__":
    files_to_fix = get_files_with_long_lines()

    for file_path in files_to_fix:
        path = Path(file_path)
        if path.exists():
            fix_long_lines(path)
