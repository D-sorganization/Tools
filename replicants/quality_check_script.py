#!/usr/bin/env python3
"""Quality check script to verify AI-generated code meets standards."""

import ast
import re
import sys
from pathlib import Path

# Configuration
BANNED_PATTERNS = [
    (re.compile(r"\bTODO\b"), "TODO placeholder found"),
    (re.compile(r"\bFIXME\b"), "FIXME placeholder found"),
    (re.compile(r"^\s*\.\.\.\s*$"), "Ellipsis placeholder"),
    (re.compile(r"NotImplementedError"), "NotImplementedError placeholder"),
    # More specific angle bracket patterns to avoid Tkinter event bindings
    (re.compile(r"<[^<>]*placeholder[^<>]*>", re.IGNORECASE), "Angle bracket placeholder"),
    (re.compile(r"<[^<>]*TODO[^<>]*>", re.IGNORECASE), "Angle bracket TODO placeholder"),
    (re.compile(r"<[^<>]*FIXME[^<>]*>", re.IGNORECASE), "Angle bracket FIXME placeholder"),
    (re.compile(r"your.*here", re.IGNORECASE), "Template placeholder"),
    (re.compile(r"insert.*here", re.IGNORECASE), "Template placeholder"),
]

# More intelligent pass statement detection
PASS_PATTERNS = [
    # Empty pass statements that are likely placeholders
    (re.compile(r"^\s*pass\s*$"), "Empty pass statement"),
    # Pass statements in empty blocks that might be placeholders
    (re.compile(r"^\s*if\s+.*:\s*$"), "Empty if block - consider adding logic or comment"),
    (re.compile(r"^\s*else:\s*$"), "Empty else block - consider adding logic or comment"),
    (re.compile(r"^\s*except\s+.*:\s*$"), "Empty except block - consider adding error handling"),
]

MAGIC_NUMBERS = [
    (re.compile(r"(?<![0-9])3\.141"), "Use math.pi instead of 3.141"),
    (re.compile(r"(?<![0-9])9\.8[0-9]?(?![0-9])"), "Define GRAVITY_M_S2 constant"),
    (re.compile(r"(?<![0-9])6\.67[0-9]?(?![0-9])"), "Define gravitational constant"),
]


def is_legitimate_pass_context(lines: list[str], line_num: int) -> bool:
    """Check if a pass statement is in a legitimate context."""
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
    # Common Tkinter event patterns
    tkinter_events = [
        r"<KeyRelease>", r"<KeyPress>", r"<Key>", r"<Return>", r"<Enter>", r"<Leave>",
        r"<Button-1>", r"<ButtonRelease-1>", r"<B1-Motion>", r"<Configure>",
        r"<MouseWheel>", r"<Button-4>", r"<Button-5>", r"<FocusIn>", r"<FocusOut>",
        r"<<ListboxSelect>>", r"<<ComboboxSelected>>", r"<<TreeviewSelect>>",
    ]

    return any(re.search(event_pattern, line) for event_pattern in tkinter_events)


def check_banned_patterns(
    lines: list[str],
    filepath: Path,
) -> list[tuple[int, str, str]]:
    """Check for banned patterns in lines."""
    issues: list[tuple[int, str, str]] = []
    # Skip checking this file for its own patterns
    if filepath.name == "quality_check_script.py":
        return issues

    for line_num, line in enumerate(lines, 1):
        # Skip legitimate Tkinter event bindings
        if is_legitimate_tkinter_binding(line):
            continue

        # Check for basic banned patterns
        for pattern, message in BANNED_PATTERNS:
            if pattern.search(line):
                issues.append((line_num, message, line.strip()))

        # Special handling for pass statements
        if (re.match(r"^\s*pass\s*$", line) and
            not is_legitimate_pass_context(lines, line_num)):
            issues.append(
                (line_num, "Empty pass statement - consider adding logic or comment", line.strip()),
            )

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
                        (node.lineno, f"Function '{node.name}' missing docstring", ""),
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


def main() -> None:
    """Run quality checks on Python files."""
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
        ".ipynb_checkpoints",  # Add checkpoint files to exclusion
        ".Trash",  # Add trash files to exclusion
    }
    python_files = [
        f for f in python_files if not any(part in exclude_dirs for part in f.parts)
    ]

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
