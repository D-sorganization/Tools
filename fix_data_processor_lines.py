#!/usr/bin/env python3
"""
Fix long lines in Data_Processor_r0.py file specifically
"""

import re
from pathlib import Path


def fix_long_lines_in_file(file_path: str) -> bool:
    """Fix long lines in the specified file"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    changes_made = False

    for _i, line in enumerate(lines):
        if len(line.rstrip()) <= 88:
            new_lines.append(line)
            continue

        # Fix long f-strings and print statements
        if 'f"' in line and len(line.rstrip()) > 88:"
            # Handle f-string patterns
            indent = len(line) - len(line.lstrip())

            # Pattern for print(f"...") statements
            if line.strip().startswith("print(") and line.strip().endswith('"),'):
                # Extract the f-string content
                match = re.match(r'(\s*print\(\s*f")(.*?)("\s*,?\s*\))', line)
                if match:
                    prefix = match.group(1)
                    content_text = match.group(2)
                    suffix = match.group(3)

                    # Break at logical points
                    for break_point in [
                        ": ",
                        " - ",
                        ", ",
                        " from ",
                        " to ",
                        " with ",
                        " for ",
                        " in ",
                        " of ",
                    ]:
                        if break_point in content_text:
                            parts = content_text.split(break_point, 1)
                            if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                new_lines.append(
                                    prefix + parts[0] + break_point.rstrip() + '"'
                                )
                                new_lines.append(
                                    " " * (indent + 4) + 'f"' + parts[1] + suffix"
                                )
                                changes_made = True
                                break
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            # Pattern for text= assignments with f-strings
            elif 'text=f"' in line:"
                match = re.match(r'(\s*.*text=f")(.*?)(".*)', line)
                if match:
                    prefix = match.group(1)
                    content_text = match.group(2)
                    suffix = match.group(3)

                    # Break at logical points
                    for break_point in [
                        ": ",
                        " - ",
                        ", ",
                        " from ",
                        " to ",
                        " with ",
                        " for ",
                        " in ",
                        " of ",
                    ]:
                        if break_point in content_text:
                            parts = content_text.split(break_point, 1)
                            if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                new_lines.append(
                                    prefix + parts[0] + break_point.rstrip() + '"'
                                )
                                new_lines.append(
                                    " " * (indent + 4) + 'f"' + parts[1] + suffix"
                                )
                                changes_made = True
                                break
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Fix long string literals
        elif '"' in line and len(line.rstrip()) > 88:
            indent = len(line) - len(line.lstrip())

            # Handle long string assignments or parameters
            if "=" in line or "(" in line:
                # Try to break at logical points
                for break_point in [
                    " - ",
                    ": ",
                    ", ",
                    " and ",
                    " or ",
                    " with ",
                    " for ",
                    " to ",
                    " from ",
                ]:
                    if break_point in line:
                        # Find the string content
                        quote_start = line.find('"')
                        quote_end = line.rfind('"')
                        if (
                            quote_start != -1
                            and quote_end != -1
                            and quote_start < quote_end
                        ):
                            before_quote = line[: quote_start + 1]
                            string_content = line[quote_start + 1 : quote_end]
                            after_quote = line[quote_end:]

                            if break_point in string_content:
                                parts = string_content.split(break_point, 1)
                                if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                    new_lines.append(
                                        before_quote
                                        + parts[0]
                                        + break_point.rstrip()
                                        + '"'
                                    )
                                    new_lines.append(
                                        " " * (indent + 4)
                                        + '"'
                                        + parts[1]
                                        + after_quote
                                    )
                                    changes_made = True
                                    break
                        break
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
        else:
            new_lines.append(line)

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        print(f"Fixed long lines in {file_path}")
        return True
    return False


def main() -> None:
    """Fix long lines in Data_Processor_r0.py"""
    file_path = (
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
    )
    path = Path(file_path)
    if path.exists():
        fix_long_lines_in_file(str(path))
    else:
        print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()
