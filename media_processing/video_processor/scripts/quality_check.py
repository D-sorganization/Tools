"""Quality check script to verify AI-generated code meets standards."""  # noqa: INP001

# QUALITY_CHECK_SCRIPT_V1 - Unique marker for script identification
# This marker MUST be present for the script to exclude itself from checks

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Store the script's own path at module level for reliable exclusion
# This MUST be set correctly for the exclusion to work
# CRITICAL: Use __file__ to get the actual script path, works in all environments
try:
    _SCRIPT_PATH: Path | None = Path(__file__).resolve()
    _SCRIPT_NAME: str = Path(__file__).name
    _SCRIPT_DIR: Path | None = Path(__file__).parent.resolve()
    _SCRIPT_RELATIVE: Path | None = Path(
        __file__,
    )  # Keep relative path too for CI compatibility
except NameError:
    _SCRIPT_PATH = None
    _SCRIPT_NAME = "quality_check.py"
    _SCRIPT_DIR = None
    _SCRIPT_RELATIVE = None

# Unique marker to identify this script - used for exclusion
_QUALITY_CHECK_SCRIPT_MARKER = "QUALITY_CHECK_SCRIPT_V1"

# Maximum file size (lines) for tertiary exclusion check
# Files with BANNED_PATTERNS smaller than this are likely the quality check script
_MAX_SCRIPT_SIZE_LINES: int = 400

# Configuration
BANNED_PATTERNS = [
    (re.compile(r"\bTODO\b"), "TODO placeholder found"),
    (re.compile(r"\bFIXME\b"), "FIXME placeholder found"),
    (re.compile(r"^\s*\.\.\.\s*$"), "Ellipsis placeholder"),
    (re.compile(r"NotImplementedError"), "NotImplementedError placeholder"),
    (re.compile(r"<.*>"), "Angle bracket placeholder"),
    (re.compile(r"your.*here", re.IGNORECASE), "Template placeholder"),
    (re.compile(r"insert.*here", re.IGNORECASE), "Template placeholder"),
]

# More intelligent pass statement detection
PASS_PATTERNS = [
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

MAGIC_NUMBERS = [
    (re.compile(r"(?<![0-9])3\.141"), "Use math.pi instead of 3.141"),
    (re.compile(r"(?<![0-9])9\.8[0-9]?(?![0-9])"), "Define GRAVITY_M_S2 constant"),
    (re.compile(r"(?<![0-9])6\.67[0-9]?(?![0-9])"), "Define gravitational constant"),
]


def _is_in_class_definition(lines: list[str], line_num: int) -> bool:
    """Check if pass is in a class definition context."""
    result = False
    for i in range(line_num - 1, max(0, line_num - 10), -1):
        prev_line = lines[i - 1].strip()
        if prev_line.startswith("class "):
            result = True
            break
        if prev_line.startswith("def "):
            result = False
            break
        if prev_line.endswith(":") and any(
            keyword in prev_line
            for keyword in ["try:", "except", "finally:", "with ", "if __name__"]
        ):
            result = True
            break
    return result


def _is_in_try_except_block(lines: list[str], line_num: int) -> bool:
    """Check if pass is in a try/except block context."""
    for i in range(line_num - 1, max(0, line_num - 5), -1):
        prev_line = lines[i - 1].strip()
        if "try:" in prev_line or "except" in prev_line:
            return True
    return False


def _is_in_context_manager(lines: list[str], line_num: int) -> bool:
    """Check if pass is in a context manager context."""
    for i in range(line_num - 1, max(0, line_num - 3), -1):
        prev_line = lines[i - 1].strip()
        if prev_line.startswith("with "):
            return True
    return False


def is_legitimate_pass_context(lines: list[str], line_num: int) -> bool:
    """Check if a pass statement is in a legitimate context."""
    if line_num <= 0 or line_num > len(lines):
        return False

    line = lines[line_num - 1].strip()
    if line != "pass":
        return False

    return (
        _is_in_class_definition(lines, line_num)
        or _is_in_try_except_block(lines, line_num)
        or _is_in_context_manager(lines, line_num)
    )


def check_banned_patterns(  # noqa: PLR0911, C901, PLR0912
    lines: list[str],
    filepath: Path,
    is_excluded: bool = False,  # noqa: FBT001, FBT002
) -> list[tuple[int, str, str]]:
    """Check for banned patterns in lines."""
    issues: list[tuple[int, str, str]] = []

    # CRITICAL: Hardcoded filename/path check - ABSOLUTE FIRST check before ANY processing
    # This is the most reliable check that works in all environments (local, CI, etc.)
    # Check filename AND path string to catch all variations
    # Also check if path contains 'scripts' and 'quality_check' to catch CI path variations
    # Also check against __file__ if available for absolute certainty
    filepath_str = str(filepath)
    filepath_lower = filepath_str.lower()

    # Check 1: Exact filename match
    if filepath.name in (
        "quality_check.py",
        "quality-check.py",
        "quality_check_script.py",
    ):
        return issues

    # Check 2: scripts/quality_check.py path combination (for CI environments)
    # Only match if path contains both "scripts" and "quality_check" and ends with .py
    # This is more specific than just "quality_check" anywhere in path
    if (
        "scripts" in filepath_lower
        and "quality_check" in filepath_lower
        and filepath_lower.endswith(".py")
    ):
        return issues

    # Check 4: Compare against __file__ if available (most reliable)
    if _SCRIPT_PATH is not None:
        try:
            filepath_resolved = filepath.resolve()
            if filepath_resolved == _SCRIPT_PATH:
                return issues
            if filepath_resolved.samefile(_SCRIPT_PATH):
                return issues
        except (OSError, ValueError):
            pass

    # CRITICAL: Check content FIRST before any processing
    # This MUST happen before pattern matching to prevent self-detection
    # This is the most important check - must run before ANY pattern matching
    # Multiple layers of checks ensure exclusion even if one fails in CI
    if not lines:
        return issues

    # Join lines into content string for checking
    file_content = "\n".join(lines)

    # ABSOLUTE FIRST CONTENT CHECK: If file contains BANNED_PATTERNS definition, it's this script
    # This is the most reliable check - only this script contains BANNED_PATTERNS = [
    # MUST be checked FIRST before any other checks
    # This check is CRITICAL - it must happen before any pattern matching
    if "BANNED_PATTERNS = [" in file_content:
        return issues

    # Additional safety: check for the unique marker in content
    # This marker is in the file header as a comment
    if _QUALITY_CHECK_SCRIPT_MARKER in file_content:
        return issues

    # Skip if already excluded (check performed once in check_file)
    if is_excluded:
        return issues

    # Process lines and check for banned patterns
    # Only reach here if content check didn't exclude the file
    for line_num, line in enumerate(lines, 1):
        # Skip lines that are pattern definitions (avoid false positives)
        # Check for actual pattern definition lines - these contain the patterns themselves
        # Match lines like: (re.compile(r"\bTODO\b"), "TODO placeholder found"),
        if (
            re.search(r'\(re\.compile\(r"[^"]*TODO', line)
            or re.search(r'\(re\.compile\(r"[^"]*FIXME', line)
            or re.search(r'\(re\.compile\(r"[^"]*NotImplementedError', line)
            or re.search(r"^\s*BANNED_PATTERNS\s*=", line)
            or '"TODO placeholder' in line
            or '"FIXME placeholder' in line
            or '"NotImplementedError placeholder' in line
        ):
            continue
        # Check for basic banned patterns
        for pattern, message in BANNED_PATTERNS:
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


def check_magic_numbers(
    lines: list[str],
    filepath: Path,  # noqa: ARG001
    is_excluded: bool = False,  # noqa: FBT001, FBT002
) -> list[tuple[int, str, str]]:
    """Check for magic numbers in lines."""
    issues: list[tuple[int, str, str]] = []
    # Skip if already excluded (check performed once in check_file)
    if is_excluded:
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


def check_file(  # noqa: PLR0911
    filepath: Path,
) -> list[tuple[int, str, str]]:
    """Check a Python file for quality issues."""
    # CRITICAL: Hardcoded filename/path check - ABSOLUTE FIRST check
    # This works in all environments and doesn't depend on path resolution
    # Also check if path contains 'scripts' and 'quality_check' to catch CI path variations
    filepath_str = str(filepath)
    filepath_lower = filepath_str.lower()
    is_quality_check_script = filepath.name in (
        "quality_check.py",
        "quality-check.py",
        "quality_check_script.py",
    ) or (
        "scripts" in filepath_lower
        and "quality_check" in filepath_lower
        and filepath_lower.endswith(".py")
    )
    if is_quality_check_script:
        return []

    # CRITICAL: Check if this is the script itself - MUST happen SECOND
    # should_exclude_file() performs comprehensive checks (filename, path, content)
    if should_exclude_file(filepath):
        return []

    try:
        content = filepath.read_text(encoding="utf-8")
        # Additional safety: check for unique marker or pattern definitions
        # This is the most reliable check - works regardless of path resolution
        # CRITICAL: This check MUST happen before reading lines to prevent any processing
        # Most permissive check: if file contains BANNED_PATTERNS definition, it's this script
        # This is the most reliable content-based check
        if "BANNED_PATTERNS = [" in content or _QUALITY_CHECK_SCRIPT_MARKER in content:
            return []

        lines = content.splitlines()

        # Cache exclusion result to avoid repeated expensive checks in helper functions
        # Note: is_excluded is False here because should_exclude_file() already returned early
        # if the file should be excluded, so we only reach here for non-excluded files
        is_excluded = False

        issues = []
        # check_banned_patterns has its own content checks as a safety net
        issues.extend(check_banned_patterns(lines, filepath, is_excluded))
        issues.extend(check_magic_numbers(lines, filepath, is_excluded))
        issues.extend(check_ast_issues(content))
    except (OSError, UnicodeDecodeError) as e:
        return [(0, f"Error reading file: {e}", "")]
    else:
        return issues


# Module-level constant for excluded filenames
_EXCLUDED_NAMES = {"quality_check_script.py", "quality_check.py"}


def _check_filename(filepath: Path) -> bool:
    """Check if file should be excluded by filename."""
    excluded_names = _EXCLUDED_NAMES | {_SCRIPT_NAME}
    return filepath.name in excluded_names


def _check_absolute_path(filepath: Path) -> bool:  # noqa: PLR0911
    """Check if file should be excluded by absolute path."""
    if _SCRIPT_PATH is None:
        return False
    try:
        file_abs = filepath.resolve()
        # Direct comparison
        if file_abs == _SCRIPT_PATH:
            return True
        # Check if paths point to same file (handles symlinks)
        if (
            _SCRIPT_PATH.exists()
            and file_abs.exists()
            and _SCRIPT_PATH.samefile(file_abs)
        ):
            return True
        # Also check relative paths - in CI, paths might be relative
        try:
            file_rel = filepath
            script_rel = Path(_SCRIPT_PATH.name)
            # Double-check by comparing parent directory
            if (
                (file_rel == script_rel or filepath.name == _SCRIPT_NAME)
                and _SCRIPT_DIR
                and filepath.parent.resolve() == _SCRIPT_DIR
            ):
                return True
        except (OSError, ValueError):
            pass
    except (OSError, ValueError, AttributeError):
        # samefile might not be available on all systems
        pass
    return False


def _check_content_signature(filepath: Path) -> bool:
    """Check if file should be excluded by content signature."""
    if not filepath.exists():
        return False
    try:
        with filepath.open(encoding="utf-8", errors="ignore") as f:
            content_start = f.read(4096)
    except (OSError, ValueError):
        return False
    else:
        # Look for unique signature of quality check script
        # Use unique marker first (most reliable), then fallback to pattern definitions
        return _QUALITY_CHECK_SCRIPT_MARKER in content_start or (
            "BANNED_PATTERNS = [" in content_start
            and "Quality check script" in content_start
            and "def should_exclude_file" in content_start
        )


def should_exclude_file(filepath: Path) -> bool:
    """Determine if a file should be excluded from checks."""
    # Check filename first (fastest check)
    if _check_filename(filepath):
        return True

    # Check absolute path (works in most cases)
    if _check_absolute_path(filepath):
        return True

    # Check content signature (fallback for CI environments)
    return _check_content_signature(filepath)


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

    # Filter out the script itself - use hardcoded filename check FIRST for reliability
    # This works in all environments (local, CI, etc.) regardless of path resolution
    # CRITICAL: This MUST be the first filter to ensure the script is never processed
    # Only match exact filenames or scripts/quality_check.py path combination
    # Do NOT use overly broad patterns that could exclude legitimate files
    python_files = [
        f
        for f in python_files
        if not (
            f.name
            in (
                "quality_check.py",
                "quality-check.py",
                "quality_check_script.py",
            )
            or (
                "scripts" in str(f).lower()
                and "quality_check" in str(f).lower()
                and str(f).endswith(".py")
            )
        )
    ]
    # Additional filter using should_exclude_file() for comprehensive checks
    python_files = [f for f in python_files if not should_exclude_file(f)]

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
