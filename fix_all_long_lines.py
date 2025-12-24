#!/usr/bin/env python3
"""
Comprehensive script to fix all line-too-long errors (E501)
"""

import json
import re
import subprocess
from pathlib import Path


def get_long_line_errors():
    """Get all E501 errors with file and line information"""
    try:
        result = subprocess.run(
            ["ruff", "check", ".", "--select", "E501", "--format", "json"],
            capture_output=True,
            text=True,
        )
        if result.stdout:
            errors = json.loads(result.stdout)
            return [
                (error["filename"], error["location"]["row"], error["message"])
                for error in errors
            ]
    except:
        pass
    return []


def fix_docstring_line(line):
    """Fix long docstring lines"""
    if '"""' in line and line.count('"""') == 2:
        # Single line docstring
        indent = len(line) - len(line.lstrip())
        content = line.strip()[3:-3].strip()
        if len(content) > 70:
            return (
                " " * indent
                + '"""\n'
                + " " * indent
                + content
                + "\n"
                + " " * indent
                + '"""\n'
            )
    elif line.strip().startswith('"""') and not line.strip().endswith('"""'):
        # Multi-line docstring start
        indent = len(line) - len(line.lstrip())
        content = line.strip()[3:].strip()
        if content and len(line.rstrip()) > 88:
            return " " * indent + '"""\n' + " " * indent + content + "\n"
    return None


def fix_comment_line(line):
    """Fix long comment lines"""
    if line.strip().startswith("#") and len(line.rstrip()) > 88:
        indent = len(line) - len(line.lstrip())
        comment_text = line.strip()[1:].strip()

        # Find good break points
        if len(comment_text) > 75:
            # Try to break at common separators
            for sep in [" - ", ": ", ", ", " and ", " or ", " but "]:
                if sep in comment_text:
                    parts = comment_text.split(sep, 1)
                    if len(parts[0]) < 75 and len(parts[1]) < 75:
                        return (
                            " " * indent
                            + "# "
                            + parts[0]
                            + sep.rstrip()
                            + "\n"
                            + " " * indent
                            + "# "
                            + parts[1]
                            + "\n"
                        )

            # Fallback: break at word boundaries
            words = comment_text.split()
            if len(words) > 8:
                mid = len(words) // 2
                first_half = " ".join(words[:mid])
                second_half = " ".join(words[mid:])
                if len(first_half) < 80 and len(second_half) < 80:
                    return (
                        " " * indent
                        + "# "
                        + first_half
                        + "\n"
                        + " " * indent
                        + "# "
                        + second_half
                        + "\n"
                    )
    return None


def fix_string_line(line):
    """Fix long string literals"""
    # Long f-strings
    if 'f"' in line and len(line.rstrip()) > 88:
        # Try to break at format specifiers or logical points
        if "{" in line and "}" in line:
            indent = len(line) - len(line.lstrip())
            # For simple cases, break the string
            match = re.match(r'(\s*.*f"[^"]*)"([^"]*)"(.*)', line)
            if match and len(match.group(2)) > 40:
                prefix = match.group(1)
                middle = match.group(2)
                suffix = match.group(3)
                # Break at a logical point
                for break_point in [", ", " - ", ": ", " to ", " from "]:
                    if break_point in middle:
                        parts = middle.split(break_point, 1)
                        if len(parts[0]) < 60 and len(parts[1]) < 60:
                            return (
                                prefix
                                + '"\n'
                                + " " * (indent + 4)
                                + f'"{break_point}{parts[1]}"{suffix}\n'
                            )

    # Regular long strings
    if (
        '"' in line
        and line.count('"') >= 2
        and not line.strip().startswith("#")
        and len(line.rstrip()) > 88
    ):
        # Try to break long string literals
        match = re.match(r'(\s*.*)"([^"]{50,})"(.*)', line)
        if match:
            prefix = match.group(1)
            string_content = match.group(2)
            suffix = match.group(3)
            indent = len(line) - len(line.lstrip())

            # Break at logical points
            for break_point in [". ", ", ", " - ", ": ", " and ", " or "]:
                if break_point in string_content:
                    parts = string_content.split(break_point, 1)
                    if len(parts[0]) < 70 and len(parts[1]) < 70:
                        return (
                            prefix
                            + '"\n'
                            + " " * (indent + 4)
                            + f'"{break_point}{parts[1]}"{suffix}\n'
                        )
    return None


def fix_function_call_line(line):
    """Fix long function calls"""
    if (
        "(" in line
        and ")" in line
        and ", " in line
        and len(line.rstrip()) > 88
        and line.count("(") == line.count(")")
    ):

        # Simple function call with parameters
        match = re.match(r"(\s*)([^(]+\()([^)]+)(\).*)", line)
        if match:
            indent_part = match.group(1)
            func_start = match.group(2)
            params = match.group(3)
            func_end = match.group(4)

            if ", " in params and len(params) > 50:
                param_list = [p.strip() for p in params.split(", ")]
                if len(param_list) >= 2:
                    # Break parameters across lines
                    result = indent_part + func_start + "\n"
                    for i, param in enumerate(param_list):
                        if i == len(param_list) - 1:
                            result += " " * (len(indent_part) + 4) + param + "\n"
                        else:
                            result += " " * (len(indent_part) + 4) + param + ",\n"
                    result += indent_part + func_end + "\n"
                    return result
    return None


def fix_long_line(line):
    """Try to fix a long line using various strategies"""
    # Try different fixing strategies
    fixes = [
        fix_docstring_line,
        fix_comment_line,
        fix_string_line,
        fix_function_call_line,
    ]

    for fix_func in fixes:
        result = fix_func(line)
        if result:
            return result

    return None


def fix_file_long_lines(file_path):
    """Fix all long lines in a file"""
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        changes_made = False

        for line in lines:
            if len(line.rstrip()) > 88:
                fixed_line = fix_long_line(line)
                if fixed_line:
                    new_lines.append(fixed_line)
                    changes_made = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if changes_made:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"Fixed long lines in {file_path}")
            return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return False


def main():
    """Main function to fix all long lines"""
    # Get all files with E501 errors
    errors = get_long_line_errors()
    files_to_fix = set(error[0] for error in errors)

    print(f"Found {len(errors)} long line errors in {len(files_to_fix)} files")

    for file_path in files_to_fix:
        if Path(file_path).exists():
            fix_file_long_lines(file_path)


if __name__ == "__main__":
    main()
