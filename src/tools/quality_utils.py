"""Shared utilities for code quality checks."""

import ast
import re
import sys
from pathlib import Path
from re import Pattern

from src.shared.python.contracts import require


# ANSI colors for terminal output
class Colors:
    if sys.stderr.isatty():
        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
    else:
        HEADER = ""
        BLUE = ""
        CYAN = ""
        GREEN = ""
        WARNING = ""
        FAIL = ""
        ENDC = ""
        BOLD = ""


# Configuration
BANNED_PATTERNS: list[tuple[Pattern, str]] = [
    (re.compile(r"\bTODO\b"), "TODO placeholder found"),
    (re.compile(r"\bFIXME\b"), "FIXME placeholder found"),
    (re.compile(r"^\s*\.\.\.\s*$"), "Ellipsis placeholder"),
    (re.compile(r"NotImplementedError"), "NotImplementedError placeholder"),
    # More specific angle bracket patterns to avoid Tkinter event bindings
    (
        re.compile(r"<[^<>]*placeholder[^<>]*>", re.IGNORECASE),
        "Angle bracket placeholder",
    ),
    (
        re.compile(r"<[^<>]*TODO[^<>]*>", re.IGNORECASE),
        "Angle bracket TODO placeholder",
    ),
    (
        re.compile(r"<[^<>]*FIXME[^<>]*>", re.IGNORECASE),
        "Angle bracket FIXME placeholder",
    ),
    (re.compile(r"your.*here", re.IGNORECASE), "Template placeholder"),
    (re.compile(r"insert.*here", re.IGNORECASE), "Template placeholder"),
]

PASS_PATTERNS: list[tuple[Pattern, str]] = [
    # Empty pass statements that are likely placeholders
    (re.compile(r"^\s*pass\s*$"), "Empty pass statement"),
    # Pass statements in empty blocks that might be placeholders
    (
        re.compile(r"^\s*if\s+.*:\s*$"),
        "Empty if block - consider adding logic or comment",
    ),
    (
        re.compile(r"^\s*else:\s*$"),
        "Empty else block - consider adding logic or comment",
    ),
    (
        re.compile(r"^\s*except\s+.*:\s*$"),
        "Empty except block - consider adding error handling",
    ),
]

MAGIC_NUMBERS: list[tuple[Pattern, str]] = [
    (re.compile(r"(?<![0-9])3\.141"), "Use math.pi instead of 3.141"),
    (re.compile(r"(?<![0-9])9\.8[0-9]?(?![0-9])"), "Define GRAVITY_M_S2 constant"),
    (re.compile(r"(?<![0-9])6\.67[0-9]?(?![0-9])"), "Define gravitational constant"),
]


def is_legitimate_pass_context(lines: list[str], line_num: int) -> bool:
    """Check if a pass statement is in a legitimate context."""
    if not (lines is not None):
        raise ValueError("lines must be provided")
    require(isinstance(lines, list), "lines must be a list")
    require(isinstance(line_num, int), "line_num must be an integer")
    if line_num <= 0 or line_num > len(lines):
        return False

    line = lines[line_num - 1].strip()
    if line != "pass":
        return False

    # Check if this is in a class definition (legitimate)
    for i in range(line_num - 1, max(0, line_num - 10), -1):
        prev_line = lines[i - 1].strip()
        if prev_line.startswith("class "):
            return True
        if prev_line.startswith("def "):
            return False
        if prev_line.endswith(":") and any(
            keyword in prev_line
            for keyword in ["try:", "except", "finally:", "with ", "if __name__"]
        ):
            return True

    # Check if this is in a try/except block (legitimate)
    for i in range(line_num - 1, max(0, line_num - 5), -1):
        prev_line = lines[i - 1].strip()
        if "try:" in prev_line or "except" in prev_line:
            return True

    # Check if this is in a context manager (legitimate)
    for i in range(line_num - 1, max(0, line_num - 3), -1):
        prev_line = lines[i - 1].strip()
        if prev_line.startswith("with "):
            return True

    return False


def is_legitimate_tkinter_binding(line: str) -> bool:
    """Check if a line contains legitimate Tkinter event bindings."""
    require(isinstance(line, str), "line must be a string")
    # Common Tkinter event patterns
    tkinter_events = [
        r"<KeyRelease>",
        r"<KeyPress>",
        r"<Key>",
        r"<Return>",
        r"<Enter>",
        r"<Leave>",
        r"<Button-1>",
        r"<ButtonRelease-1>",
        r"<B1-Motion>",
        r"<Configure>",
        r"<MouseWheel>",
        r"<Button-4>",
        r"<Button-5>",
        r"<FocusIn>",
        r"<FocusOut>",
        r"<<ListboxSelect>>",
        r"<<ComboboxSelected>>",
        r"<<TreeviewSelect>>",
    ]

    return any(re.search(event_pattern, line) for event_pattern in tkinter_events)


def check_banned_patterns(
    lines: list[str],
    filepath: Path,
) -> list[tuple[int, str, str]]:
    """Check for banned patterns in lines."""
    if not (lines is not None):
        raise ValueError("lines must be provided")
    require(isinstance(lines, list), "lines must be a list of strings")
    require(isinstance(filepath, Path), "filepath must be a Path")
    issues: list[tuple[int, str, str]] = []
    # Skip checking files that contain placeholder detection patterns themselves
    excluded_files = {
        "quality_check_script.py",
        "matlab_quality_check.py",
        "code_quality_check.py",
        "quality_utils.py",
    }
    if filepath.name in excluded_files:
        return issues

    for line_num, line in enumerate(lines, 1):
        # Check for basic banned patterns
        for pattern, message in BANNED_PATTERNS:
            # Skip angle bracket patterns if line contains legitimate Tkinter bindings
            pattern_str = pattern.pattern if hasattr(pattern, "pattern") else str(pattern)
            if "<" in pattern_str and is_legitimate_tkinter_binding(line):
                continue
            if pattern.search(line):
                issues.append((line_num, message, line.strip()))

        # Special handling for pass statements
        if re.match(r"^\s*pass\s*$", line) and not is_legitimate_pass_context(
            lines,
            line_num,
        ):
            issues.append(
                (
                    line_num,
                    "Empty pass statement - consider adding logic or comment",
                    line.strip(),
                ),
            )

    return issues


def strip_comments_from_line(line: str) -> str:
    """Strip comments from a line, handling string literals correctly."""
    require(isinstance(line, str), "line must be a string")
    in_single_quote = False
    in_double_quote = False
    in_triple_single = False
    in_triple_double = False
    escaped = False
    i = 0

    while i < len(line):
        char = line[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if char == "\\":
            escaped = True
            i += 1
            continue

        if i + 2 < len(line):
            triple = line[i : i + 3]
            if triple == '"""' and not in_single_quote and not in_triple_single:
                in_triple_double = not in_triple_double
                i += 3
                continue
            if triple == "'''" and not in_double_quote and not in_triple_double:
                in_triple_single = not in_triple_single
                i += 3
                continue

        if not in_triple_single and not in_triple_double:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == "#" and not in_single_quote and not in_double_quote:
                return line[:i].rstrip()

        i += 1

    return line


def check_magic_numbers(lines: list[str], filepath: Path) -> list[tuple[int, str, str]]:
    """Check for magic numbers in lines."""
    if not (lines is not None):
        raise ValueError("lines must be provided")
    require(isinstance(lines, list), "lines must be a list of strings")
    require(isinstance(filepath, Path), "filepath must be a Path")
    issues: list[tuple[int, str, str]] = []
    excluded_files = {
        "quality_check_script.py",
        "matlab_quality_check.py",
        "code_quality_check.py",
        "quality_utils.py",
    }
    if filepath.name in excluded_files:
        return issues

    for line_num, line in enumerate(lines, 1):
        line_content = strip_comments_from_line(line)
        for pattern, message in MAGIC_NUMBERS:
            if pattern.search(line_content):
                issues.append((line_num, message, line.strip()))
    return issues


def check_ast_issues(content: str, filepath: Path) -> list[tuple[int, str, str]]:
    """Check AST for quality issues."""
    if not (content is not None):
        raise ValueError("content must be provided")
    issues: list[tuple[int, str, str]] = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not ast.get_docstring(node):
                issues.append(
                    (node.lineno, f"Function '{node.name}' missing docstring", ""),
                )
    except SyntaxError as e:
        issues.append((0, f"Syntax error: {e}", ""))
    return issues


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a Python file for quality issues."""
    require(isinstance(filepath, Path), "filepath must be a Path")
    require(filepath.exists(), f"File not found: {filepath}")
    require(filepath.is_file(), f"Path is not a regular file: {filepath}")
    try:
        content = filepath.read_text(encoding="utf-8")
        lines = content.splitlines()

        issues = []
        issues.extend(check_banned_patterns(lines, filepath))
        issues.extend(check_magic_numbers(lines, filepath))
        issues.extend(check_ast_issues(content, filepath))
    except (OSError, UnicodeDecodeError) as e:
        return [(0, f"Error reading file: {e}", "")]
    else:
        return issues
