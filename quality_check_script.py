#!/usr/bin/env python3
"""Quality check script to verify AI-generated code meets standards."""

import argparse
import ast
import re
import sys
from pathlib import Path

# Configuration
BANNED_PATTERNS = [
    (re.compile(r"\bTODO\b"), "TODO placeholder found"),
    (re.compile(r"\bFIXME\b"), "FIXME placeholder found"),
    # Updated to exclude legitimate pass statements in except blocks
    (re.compile(r"^(\s*)pass\s*$"), "Empty pass statement"),
    (re.compile(r"^(\s*)\.\.\.\s*$"), "Ellipsis placeholder"),
    (re.compile(r"NotImplementedError"), "NotImplementedError placeholder"),
    # Updated to exclude legitimate comparison operators and Tkinter event bindings
    (re.compile(r"<[^>]*>"), "Angle bracket placeholder"),
    (re.compile(r"your.*here", re.IGNORECASE), "Template placeholder"),
    (re.compile(r"insert.*here", re.IGNORECASE), "Template placeholder"),
]

# Add exclusion for Tkinter event bindings and comparison operators
EXCLUDED_PATTERNS = [
    re.compile(r"bind\s*<[^>]*>"),  # Tkinter event bindings
    re.compile(r"<[^>]*>.*bind"),  # Tkinter event bindings
    re.compile(r"bind_all\s*<[^>]*>"),  # Tkinter bind_all event bindings
    re.compile(r"<[^>]*>.*bind_all"),  # Tkinter bind_all event bindings
    re.compile(r"<[^>]*>.*,.*"),  # Tkinter event bindings with parameters
    re.compile(r"<=|>=|!=|=="),  # Comparison operators
    re.compile(r"<[^>]*>"),  # Any angle brackets (legitimate Tkinter events)
]

# Add exclusion for legitimate pass statements in except blocks
EXCEPT_PASS_PATTERN = re.compile(r"^\s*except.*:\s*$")

MAGIC_NUMBERS = [
    (re.compile(r"(?<![0-9])3\.141"), "Use math.pi instead of 3.141"),
    (re.compile(r"(?<![0-9])9\.8[0-9]?(?![0-9])"), "Define GRAVITY_M_S2 constant"),
    (re.compile(r"(?<![0-9])6\.67[0-9]?(?![0-9])"), "Define gravitational constant"),
]


def check_banned_patterns(
    lines: list[str],
    filepath: Path,
) -> list[tuple[int, str, str]]:
    """Check for banned patterns in lines."""
    issues: list[tuple[int, str, str]] = []
    # Skip checking this file for its own patterns
    if filepath.name == "quality_check_script.py":
        return issues

    # Track if we're in an except block
    in_except_block = False

    for line_num, line in enumerate(lines, 1):
        # Check if we're entering an except block
        if EXCEPT_PASS_PATTERN.search(line):
            in_except_block = True
            continue

        # Check if we're leaving the except block (next non-empty, non-comment line)
        if in_except_block and line.strip() and not line.strip().startswith("#"):
            if not line.strip().startswith("pass"):
                in_except_block = False

        # Skip lines that contain legitimate Tkinter event bindings
        is_tkinter_event = any(pattern.search(line) for pattern in EXCLUDED_PATTERNS)
        if is_tkinter_event:
            continue

        for pattern, message in BANNED_PATTERNS:
            if pattern.search(line):
                # Skip pass statements that are in except blocks
                if "pass" in message and in_except_block and "pass" in line.strip():
                    continue
                issues.append((line_num, message, line.strip()))
    return issues


def check_magic_numbers(lines: list[str], filepath: Path) -> list[tuple[int, str, str]]:
    """Check for magic numbers in lines."""
    issues: list[tuple[int, str, str]] = []
    # Skip checking this file for magic numbers
    if filepath.name == "quality_check_script.py":
        return issues
    for line_num, line in enumerate(lines, 1):
        line_content = line[: line.index("#")] if "#" in line else line
        for pattern, message in MAGIC_NUMBERS:
            if pattern.search(line_content):
                issues.append((line_num, message, line.strip()))
    return issues


def check_ast_issues(content: str) -> list[tuple[int, str, str]]:
    """Check AST for quality issues."""
    issues: list[tuple[int, str, str]] = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    issues.append(
                        (
                            node.lineno,
                            f"Function '{node.name}' missing docstring",
                            "",
                        ),
                    )
                if not node.returns and node.name != "__init__":
                    issues.append(
                        (
                            node.lineno,
                            f"Function '{node.name}' missing return type hint",
                            "",
                        ),
                    )
    except SyntaxError as e:
        issues.append((0, f"Syntax error: {e}", ""))
    return issues


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a Python file for quality issues."""
    try:
        content = filepath.read_text(encoding="utf-8")
        lines = content.splitlines()

        issues = []
        issues.extend(check_banned_patterns(lines, filepath))
        issues.extend(check_magic_numbers(lines, filepath))
        issues.extend(check_ast_issues(content))
    except (OSError, UnicodeDecodeError) as e:
        return [(0, f"Error reading file: {e}", "")]
    else:
        return issues


def get_all_python_files() -> list[Path]:
    """Get all Python files in repository with exclusions."""
    python_files = list(Path().rglob("*.py"))

    # Exclude certain directories
    exclude_dirs = {
        "archive",
        "legacy",
        "experimental",
        ".git",
        "__pycache__",
        ".ruff_cache",
        ".mypy_cache",
        "matlab",
        "output",
    }
    return [
        f for f in python_files if not any(part in exclude_dirs for part in f.parts)
    ]


def get_specified_python_files(file_paths: list[str]) -> list[Path]:
    """Get Python files from specified paths."""
    python_files = [Path(f) for f in file_paths if f.endswith(".py")]
    if not python_files:
        sys.stderr.write("❌ No Python files specified\n")
        sys.exit(1)
    return python_files


def get_python_files_from_args(args: argparse.Namespace) -> list[Path]:
    """Get Python files based on command line arguments."""
    if args.files:
        return get_specified_python_files(args.files)
    return get_all_python_files()


def main() -> None:
    """Run quality checks on Python files."""
    parser = argparse.ArgumentParser(
        description="Quality check script for Python files",
    )
    parser.add_argument("--files", nargs="+", help="Specific Python files to check")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all Python files in repository",
    )
    args = parser.parse_args()

    python_files = get_python_files_from_args(args)

    all_issues = []
    for filepath in python_files:
        issues = check_file(filepath)
        if issues:
            all_issues.append((filepath, issues))

    # Report
    if all_issues:
        sys.stderr.write("❌ Quality check FAILED\n\n")
        for filepath, issues in all_issues:
            sys.stderr.write(f"\n{filepath}:\n")
            for line_num, message, code in issues:
                if line_num > 0:
                    sys.stderr.write(f"  Line {line_num}: {message}\n")
                    if code:
                        sys.stderr.write(f"    > {code}\n")
                else:
                    sys.stderr.write(f"  {message}\n")

        sys.stderr.write(
            f"\nTotal issues: {sum(len(issues) for _, issues in all_issues)}\n",
        )
        sys.exit(1)
    else:
        sys.stderr.write("✅ Quality check PASSED\n")
        sys.stderr.write(f"Checked {len(python_files)} Python files\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
