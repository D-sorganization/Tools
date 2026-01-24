#!/usr/bin/env python3
"""
Comprehensive batch fixer for DRY violations.

This script systematically fixes common DRY violations across the codebase.
"""

import re
import sys
from pathlib import Path

# Track total fixes
TOTAL_FIXES = 0


def fix_json_load_dump(content: str) -> tuple[str, int]:
    """Replace json.load/dump with file_utils functions."""
    global TOTAL_FIXES
    fixes = 0

    # Pattern: with open(...) as f: data = json.load(f)
    pattern1 = r'with open\(([^)]+),\s*encoding=["\']utf-8["\']?\s*\)\s+as\s+(\w+):\s*\n\s*(\w+)\s*=\s*json\.load\(\2\)'

    def replace1(match):
        nonlocal fixes
        fixes += 1
        file_path = match.group(1)
        var_name = match.group(3)
        return f"{var_name} = safe_read_json({file_path}, default=None)"

    content = re.sub(pattern1, replace1, content, flags=re.MULTILINE)

    # Pattern: json.loads(...)
    pattern2 = r"json\.loads\(([^)]+)\)"

    def replace2(match):
        nonlocal fixes
        fixes += 1
        return f"safe_read_json({match.group(1)}, default=None)"

    # Only replace if not already using safe_read_json
    if "safe_read_json" not in content:
        content = re.sub(pattern2, replace2, content)

    # Add import if fixes made
    if fixes > 0 and "from utils.file_utils import safe_read_json" not in content:
        import_pattern = r"(^import json\n|^from pathlib import Path\n)"
        match = re.search(import_pattern, content, re.MULTILINE)
        if match:
            pos = match.end()
            content = (
                content[:pos] + "\n# Use shared file utility\n"
                "try:\n"
                "    from utils.file_utils import safe_read_json\n"
                "except ImportError:\n"
                "    # Fallback\n"
                "    def safe_read_json(path, default=None):\n"
                "        import json\n"
                '        with open(path, encoding="utf-8") as f:\n'
                "            return json.load(f)\n" + content[pos:]
            )

    TOTAL_FIXES += fixes
    return content, fixes


def fix_os_path_operations(content: str) -> tuple[str, int]:
    """Replace os.path operations with Path objects."""
    global TOTAL_FIXES
    fixes = 0

    # Pattern: Path(a) / b -> Path(a) / b
    pattern1 = r"os\.path\.join\s*\(([^)]+)\)"

    def replace1(match):
        nonlocal fixes
        fixes += 1
        args = match.group(1)
        parts = [p.strip().strip("\"'") for p in args.split(",")]
        if len(parts) == 1:
            return f"Path({parts[0]})"
        result = f"Path({parts[0]})"
        for part in parts[1:]:
            result += f" / {part}"
        return result

    content = re.sub(pattern1, replace1, content)

    # Pattern: Path(path).exists() -> Path(path).exists()
    pattern2 = r"os\.path\.exists\s*\(([^)]+)\)"

    def replace2(match):
        nonlocal fixes
        fixes += 1
        return f"Path({match.group(1)}).exists()"

    content = re.sub(pattern2, replace2, content)

    # Pattern: Path(path).parent -> Path(path).parent
    pattern3 = r"os\.path\.dirname\s*\(([^)]+)\)"

    def replace3(match):
        nonlocal fixes
        fixes += 1
        return f"Path({match.group(1)}).parent"

    content = re.sub(pattern3, replace3, content)

    TOTAL_FIXES += fixes
    return content, fixes


def fix_logging_setup(content: str) -> tuple[str, int]:
    """Replace logging.basicConfig with logging_utils."""
    global TOTAL_FIXES
    fixes = 0

    # Pattern: init_default_logging()
    pattern = r"logging\.basicConfig\s*\([^)]+\)"

    def replace(match):
        nonlocal fixes
        fixes += 1
        return "init_default_logging()"

    content = re.sub(pattern, replace, content)

    # Add import if fixes made
    if (
        fixes > 0
        and "from utils.logging_utils import init_default_logging" not in content
    ):
        import_pattern = r"(^import logging\n|^from pathlib import Path\n)"
        match = re.search(import_pattern, content, re.MULTILINE)
        if match:
            pos = match.end()
            content = (
                content[:pos] + "\n# Use shared logging utility\n"
                "try:\n"
                "    from utils.logging_utils import init_default_logging\n"
                "except ImportError:\n"
                "    # Fallback\n"
                "    def init_default_logging():\n"
                "        init_default_logging()\n" + content[pos:]
            )

    TOTAL_FIXES += fixes
    return content, fixes


def fix_subprocess_calls(content: str) -> tuple[str, int]:
    """Replace subprocess calls with subprocess_utils."""
    global TOTAL_FIXES
    fixes = 0

    # Pattern: subprocess.run([...], ...)
    pattern1 = r"subprocess\.run\s*\(([^)]+)\)"

    def replace1(match):
        nonlocal fixes
        fixes += 1
        args = match.group(1)
        # Extract command list (simplified)
        return f"run_command({args})"

    # Only replace if not already using run_command
    if "run_command" not in content:
        content = re.sub(pattern1, replace1, content)

    # Add import if fixes made
    if fixes > 0 and "from utils.subprocess_utils import run_command" not in content:
        import_pattern = r"(^import subprocess\n|^from pathlib import Path\n)"
        match = re.search(import_pattern, content, re.MULTILINE)
        if match:
            pos = match.end()
            content = (
                content[:pos] + "\n# Use shared subprocess utility\n"
                "try:\n"
                "    from utils.subprocess_utils import run_command\n"
                "except ImportError:\n"
                "    # Fallback\n"
                "    import subprocess\n"
                "    run_command = subprocess.run\n" + content[pos:]
            )

    TOTAL_FIXES += fixes
    return content, fixes


def process_file(file_path: Path) -> int:
    """Process a single file and fix all DRY violations."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original = content
        total_fixes = 0

        # Apply all fixes
        content, fixes = fix_json_load_dump(content)
        total_fixes += fixes

        content, fixes = fix_os_path_operations(content)
        total_fixes += fixes

        content, fixes = fix_logging_setup(content)
        total_fixes += fixes

        content, fixes = fix_subprocess_calls(content)
        total_fixes += fixes

        # Only write if changes made
        if content != original:
            file_path.write_text(content, encoding="utf-8")
            return total_fixes

        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return 0


def main():
    """Main entry point."""
    import subprocess

    # Find files with violations
    patterns = {
        "json": r"json\.(load|dump)",
        "os.path": r"os\.path\.(join|exists|dirname)",
        "logging": r"logging\.basicConfig",
        "subprocess": r"subprocess\.(run|Popen)",
    }

    all_files = set()
    for pattern_name, pattern in patterns.items():
        result = subprocess.run(
            ["grep", "-r", "--include=*.py", "-l", "-E", pattern, "."],
            capture_output=True,
            text=True,
            cwd="/home/dieterolson/Linux_Tools/Tools",
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        for f in files:
            if "__pycache__" not in f and ".git" not in f and "utils" not in f:
                all_files.add(f)

    print(f"Found {len(all_files)} files with violations")

    total_fixes = 0
    fixed_files = 0

    for file_str in sorted(all_files):
        file_path = Path(file_str)
        if file_path.exists():
            fixes = process_file(file_path)
            if fixes > 0:
                print(f"Fixed {fixes} violations in {file_path}")
                total_fixes += fixes
                fixed_files += 1

    print(f"\nTotal: {fixed_files} files, {total_fixes} violations fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
