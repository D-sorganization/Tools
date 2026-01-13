"""Script to convert print() statements to logging calls."""

import re
import sys
from pathlib import Path


def convert_print_to_logging(file_path: Path) -> tuple[int, str]:
    """Convert print() statements to logging calls.

    Args:
        file_path: Path to the Python file to convert

    Returns:
        Tuple of (number of conversions, modified content)
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()  # More efficient than split("\n")
    modified_lines = []
    conversions = 0

    # Add logging import if not present - check for various logging patterns
    has_logging_import = any(
        "import logging" in line
        or "from logging" in line
        or "logging_config" in line
        or "logger_utils" in line
        for line in lines[:50]
    )

    for line in lines:
        # Skip if line is a comment or in a docstring context
        stripped = line.strip()
        if stripped.startswith("#"):
            modified_lines.append(line)
            continue

        # Match print statements
        # Pattern: print(...)
        match = re.match(r"^(\s*)print\((.+)\)(\s*)$", line)
        if match:
            indent = match.group(1)
            content_inside = match.group(2)
            trailing = match.group(3)

            # Determine log level based on content
            if "error" in content_inside.lower() or "Error" in content_inside:
                level = "error"
            elif "warn" in content_inside.lower():
                level = "warning"
            elif "debug" in content_inside.lower() or "DEBUG" in content_inside:
                level = "debug"
            else:
                level = "info"

            # Convert f-strings to logging format if simple enough
            # For now, keep f-strings as they work with logging
            new_line = f"{indent}logger.{level}({content_inside}){trailing}"
            modified_lines.append(new_line)
            conversions += 1
        else:
            modified_lines.append(line)

    # Add logger import at the top after other imports if we made changes
    if conversions > 0 and not has_logging_import:
        # Find the first non-import line after the imports
        insert_pos = 0
        in_docstring = False
        for i, line in enumerate(modified_lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    continue
                if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    continue
                in_docstring = True
                continue
            if in_docstring:
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_pos = i + 1

        # Insert logger import - use standard logging module for portability
        logger_import = "import logging\n\n" "logger = logging.getLogger(__name__)\n"
        modified_lines.insert(insert_pos, logger_import)

    return conversions, "\n".join(modified_lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python convert_print_to_logging.py <file_path>\n")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        sys.stderr.write(f"File not found: {file_path}\n")
        sys.exit(1)

    conversions, new_content = convert_print_to_logging(file_path)

    if conversions > 0:
        file_path.write_text(new_content, encoding="utf-8")
        sys.stdout.write(
            f"Converted {conversions} print() statements to logging in {file_path}\n"
        )
    else:
        sys.stdout.write(f"No print() statements found in {file_path}\n")
