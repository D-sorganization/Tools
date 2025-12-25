#!/usr/bin/env python3
"""
Fix remaining long lines that black couldn't handle
"""

import re
from pathlib import Path


def fix_docstring_and_comments(file_path):
    """Fix long docstrings and comments"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    changes_made = False

    for line in lines:
        if len(line.rstrip()) <= 88:
            new_lines.append(line)
            continue

        # Fix long docstring content
        if (
            line.strip()
            and not line.strip().startswith('"""')
            and not line.strip().endswith('"""')
        ):
            if any(prev_line.strip().startswith('"""') for prev_line in new_lines[-3:]):
                # We're inside a docstring
                indent = len(line) - len(line.lstrip())
                content_text = line.strip()
                if len(content_text) > 75:
                    # Break at logical points
                    for break_point in [
                        " and ",
                        " with ",
                        " for ",
                        " to ",
                        " from ",
                        " in ",
                        " of ",
                        " at ",
                    ]:
                        if break_point in content_text:
                            parts = content_text.split(break_point, 1)
                            if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                new_lines.append(
                                    " " * indent + parts[0] + break_point.rstrip()
                                )
                                new_lines.append(" " * indent + parts[1])
                                changes_made = True
                                break
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Fix long comments
        elif line.strip().startswith("#") and len(line.rstrip()) > 88:
            indent = len(line) - len(line.lstrip())
            comment_text = line.strip()[1:].strip()

            # Break at logical points
            for break_point in [
                " - ",
                ": ",
                ", ",
                " and ",
                " or ",
                " but ",
                " with ",
                " for ",
            ]:
                if break_point in comment_text:
                    parts = comment_text.split(break_point, 1)
                    if 20 < len(parts[0]) < 75 and len(parts[1]) < 75:
                        new_lines.append(
                            " " * indent + "# " + parts[0] + break_point.rstrip()
                        )
                        new_lines.append(" " * indent + "# " + parts[1])
                        changes_made = True
                        break
            else:
                new_lines.append(line)

        # Fix long f-strings and regular strings
        elif 'f"' in line or '"' in line:
            # Try to break long strings
            if len(line.rstrip()) > 88:
                # Look for f-string patterns
                match = re.match(r'(\s*.*f"[^"]*)"([^"]*)"(.*)', line)
                if match:
                    prefix = match.group(1)
                    middle = match.group(2)
                    suffix = match.group(3)
                    if len(middle) > 40:
                        # Break the string
                        for break_point in [
                            ", ",
                            " - ",
                            ": ",
                            " to ",
                            " from ",
                            " with ",
                        ]:
                            if break_point in middle:
                                parts = middle.split(break_point, 1)
                                if len(parts[0]) < 60 and len(parts[1]) < 60:
                                    indent = len(line) - len(line.lstrip())
                                    new_lines.append(prefix + '"')
                                    new_lines.append(
                                        " " * (indent + 4)
                                        + f'"{break_point}{parts[1]}"{suffix}'
                                    )
                                    changes_made = True
                                    break
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        print(f"Fixed long lines in {file_path}")
        return True
    return False


def main():
    """Fix long lines in problematic files"""
    files_to_check = [
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py",
        "data_processing/data_processor/python/data_processor/Data_Processor_Integrated.py",
        "media_processing/video_processor/python/tests/test_logger_utils_mock.py",
        "media_processing/video_processor/scripts/quality_check.py",
        "media_processing/video_processor/tools/code_quality_check.py",
        "media_processing/video_processor/tools/matlab_utilities/scripts/matlab_quality_check.py",
        "web_applications/unit_converter/tools/code_quality_check.py",
    ]

    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            fix_docstring_and_comments(path)


if __name__ == "__main__":
    main()
