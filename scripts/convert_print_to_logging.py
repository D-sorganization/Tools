"""Script to convert print() statements to logging calls."""

import re
import sys
from pathlib import Path

PRINT_PATTERN = re.compile(r"^(\s*)print\((.+)\)(\s*)$")


def _has_logging_import(lines: list[str]) -> bool:
    """Check if file already has logging imports."""
    return any(
        "import logging" in line
        or "from logging" in line
        or "logging_config" in line
        or "logger_utils" in line
        for line in lines[:50]
    )


def _determine_log_level(content: str) -> str:
    """Determine appropriate log level based on message content."""
    content_lower = content.lower()
    if "error" in content_lower:
        return "error"
    if "warn" in content_lower:
        return "warning"
    if "debug" in content_lower:
        return "debug"
    return "info"


def _convert_print_line(line: str) -> tuple[str, bool]:
    """Convert a single print statement to logging call.

    Returns:
        Tuple of (converted line, was_converted).
    """
    match = PRINT_PATTERN.match(line)
    if not match:
        return line, False

    indent = match.group(1)
    content_inside = match.group(2)
    trailing = match.group(3)
    level = _determine_log_level(content_inside)

    return f"{indent}logger.{level}({content_inside}){trailing}", True


def _find_import_insert_position(lines: list[str]) -> int:
    """Find the position to insert logger import after existing imports."""
    insert_pos = 0
    in_docstring = False

    for i, line in enumerate(lines):
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

    return insert_pos


def convert_print_to_logging(file_path: Path) -> tuple[int, str]:
    """Convert print() statements to logging calls.

    Args:
        file_path: Path to the Python file to convert

    Returns:
        Tuple of (number of conversions, modified content)
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    modified_lines = []
    conversions = 0

    has_logging = _has_logging_import(lines)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            modified_lines.append(line)
            continue

        converted_line, was_converted = _convert_print_line(line)
        modified_lines.append(converted_line)
        if was_converted:
            conversions += 1

    if conversions > 0 and not has_logging:
        insert_pos = _find_import_insert_position(modified_lines)
        logger_import = "import logging\n\nlogger = logging.getLogger(__name__)\n"
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
