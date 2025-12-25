#!/usr/bin/env python3
"""
Aggressive fix for remaining Ruff issues.
Target the most impactful issues that can be safely automated.
"""

import re
from pathlib import Path


def fix_import_organization_aggressive() -> bool:
    """Aggressively fix import organization by completely restructuring files."""
    files_to_fix = [
        "replicants/python/folder_packer_pro/folder_packer_pro.py",
        "replicants/python/folder_tool_pro/folder_fix_pro.py",
    ]

    changes_made = False

    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Completely rewrite the file with proper import structure
            new_content = completely_restructure_imports(content, file_path)

            if new_content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"✅ Aggressively restructured imports in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def completely_restructure_imports(content: str, file_path: str) -> str:
    """Completely restructure file with proper imports at the top."""
    lines = content.split("\n")

    # Extract the shebang and module docstring
    header_lines = []
    docstring_start = -1
    docstring_end = -1

    # Handle shebang
    if lines and lines[0].startswith("#!"):
        header_lines.append(lines[0])
        start_idx = 1
    else:
        start_idx = 0

    # Find module docstring
    in_docstring = False
    docstring_quotes = None

    for _ in range(start_idx, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not stripped and docstring_start == -1:
            header_lines.append(line)
            continue

        if not in_docstring and (
            stripped.startswith('"""') or stripped.startswith("'''")
        ):
            docstring_quotes = stripped[:3]
            docstring_start = i
            in_docstring = True
            header_lines.append(line)

            # Check if it's a single-line docstring
            if stripped.count(docstring_quotes) >= 2 and len(stripped) > 3:
                docstring_end = i
                break
        elif in_docstring:
            header_lines.append(line)
            if docstring_quotes in stripped:
                docstring_end = i
                break
        elif docstring_start == -1:
            # No docstring found, this is the first real line
            break

    # Collect all imports from the entire file
    all_imports = set()
    code_lines = []

    # Scan the entire file for imports
    for line in lines[docstring_end + 1 :]:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Extract the import
            if stripped.startswith("from ") and " import " in stripped:
                module_part = stripped.split(" import ")[0] + " import "
                imports_part = stripped.split(" import ")[1]
                # Clean up the import
                clean_import = module_part + imports_part.split("#")[0].strip()
                all_imports.add(clean_import)
            elif stripped.startswith("import "):
                clean_import = stripped.split("#")[0].strip()
                all_imports.add(clean_import)
        elif not (stripped == "" or stripped.startswith("#")):
            # This is actual code, collect everything from here
            code_lines = lines[lines.index(line) :]
            break

    # Add specific imports we know are needed
    if "folder_packer_pro" in file_path:
        required_imports = {
            "import base64",
            "import gzip",
            "import json",
            "import logging",
            "import os",
            "import re",
            "import sys",
            "import threading",
            "import tkinter as tk",
            "from collections import defaultdict",
            "from datetime import UTC, datetime",
            "from pathlib import Path",
            "from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk",
            "from typing import Any, Final",
            "from cryptography.fernet import Fernet",
            "from cryptography.hazmat.primitives import hashes",
            "from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC",
        }
    elif "folder_fix_pro" in file_path:
        required_imports = {
            "import ctypes",
            "import hashlib",
            "import json",
            "import logging",
            "import os",
            "import re",
            "import shutil",
            "import sys",
            "import threading",
            "import tkinter as tk",
            "import typing",
            "import webbrowser",
            "from collections import defaultdict",
            "from datetime import datetime, timezone",
            "from pathlib import Path",
            "from tkinter import filedialog, messagebox, ttk",
        }
    else:
        required_imports = all_imports

    # Combine found imports with required imports
    all_imports.update(required_imports)

    # Organize imports
    organized_imports = organize_imports_professionally(list(all_imports))

    # Reconstruct the file
    result_lines = []
    result_lines.extend(header_lines)
    result_lines.append("")
    result_lines.extend(organized_imports)
    result_lines.append("")

    # Add constants and code
    if code_lines:
        result_lines.extend(code_lines)

    return "\n".join(result_lines)


def organize_imports_professionally(imports: list[str]) -> list[str]:
    """Organize imports with professional standards."""
    stdlib_imports = []
    third_party_imports = []

    # Comprehensive standard library modules
    stdlib_modules = {
        "base64",
        "gzip",
        "json",
        "logging",
        "os",
        "re",
        "sys",
        "threading",
        "ctypes",
        "hashlib",
        "shutil",
        "typing",
        "collections",
        "datetime",
        "pathlib",
        "tkinter",
        "webbrowser",
        "subprocess",
        "functools",
        "itertools",
        "operator",
        "tempfile",
        "uuid",
        "time",
        "math",
        "socket",
        "urllib",
        "http",
        "email",
        "html",
        "xml",
        "csv",
        "configparser",
        "argparse",
        "getpass",
        "platform",
        "glob",
    }

    for import_line in imports:
        if not import_line.strip():
            continue

        # Extract module name
        if import_line.startswith("from "):
            module = import_line.split()[1].split(".")[0]
        elif import_line.startswith("import "):
            module = import_line.split()[1].split(".")[0]
        else:
            continue

        if module in stdlib_modules:
            stdlib_imports.append(import_line)
        else:
            third_party_imports.append(import_line)

    # Sort imports within each group
    stdlib_imports.sort()
    third_party_imports.sort()

    # Combine with proper spacing
    result = []
    if stdlib_imports:
        result.extend(stdlib_imports)
    if third_party_imports:
        if stdlib_imports:
            result.append("")
        result.extend(third_party_imports)

    return result


def fix_line_length_aggressively() -> bool:
    """Aggressively fix line length issues."""
    files_to_fix = [
        "data_processing/data_processor/python/data_processor/Data_Processor_Integrated.py",
        "media_processing/video_processor/scripts/quality_check.py",
        "media_processing/video_processor/tools/matlab_utilities/scripts/matlab_quality_check.py",
    ]

    changes_made = False

    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply aggressive line length fixes
            content = apply_aggressive_line_fixes(content)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Aggressively fixed line lengths in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def apply_aggressive_line_fixes(content: str) -> str:
    """Apply aggressive line length fixes."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if len(line.rstrip()) > 88:
            fixed_lines = fix_line_aggressively(line)
            if isinstance(fixed_lines, list):
                new_lines.extend(fixed_lines)
            else:
                new_lines.append(fixed_lines)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def fix_line_aggressively(line: str) -> str | list[str]:
    """Aggressively fix a single long line."""
    indent = len(line) - len(line.lstrip())
    stripped = line.strip()

    # Fix long comments by breaking at any reasonable point
    if stripped.startswith("#"):
        comment_text = stripped[1:].strip()
        if len(comment_text) > 80:
            # Find any reasonable break point
            break_points = [
                " - ",
                ": ",
                ", ",
                " and ",
                " or ",
                " but ",
                " with ",
                " for ",
                " in ",
                " to ",
                " of ",
                " at ",
                " on ",
                " by ",
                " from ",
                " into ",
                " through ",
                " during ",
                " after ",
                " before ",
                " while ",
                " when ",
                " where ",
                " which ",
                " that ",
            ]

            for break_point in break_points:
                if break_point in comment_text:
                    idx = comment_text.find(break_point)
                    if 20 < idx < 75:
                        part1 = comment_text[: idx + len(break_point.rstrip())]
                        part2 = comment_text[idx + len(break_point) :].strip()
                        return [
                            " " * indent + "# " + part1,
                            " " * indent + "# " + part2,
                        ]

            # If no good break point, just break at 75 characters
            if len(comment_text) > 75:
                return [
                    " " * indent + "# " + comment_text[:75],
                    " " * indent + "# " + comment_text[75:],
                ]

    # Fix long string literals
    if '"' in stripped and len(stripped) > 88:
        # Try to break string concatenation
        if " + " in stripped:
            parts = stripped.split(" + ")
            if len(parts) > 1:
                result = [" " * indent + parts[0] + " +"]
                for part in parts[1:-1]:
                    result.append(" " * (indent + 4) + part + " +")
                result.append(" " * (indent + 4) + parts[-1])
                return result

    # Fix long function calls
    if "(" in stripped and ")" in stripped and "," in stripped:
        # Find the opening parenthesis
        paren_pos = stripped.find("(")
        if paren_pos > 0 and paren_pos < 60:
            func_part = stripped[: paren_pos + 1]
            remaining = stripped[paren_pos + 1 :]

            # Find the closing parenthesis
            close_paren = remaining.rfind(")")
            if close_paren > 0:
                args_part = remaining[:close_paren]
                end_part = remaining[close_paren:]

                if "," in args_part:
                    # Split arguments
                    args = [arg.strip() for arg in args_part.split(",")]
                    if len(args) > 1:
                        result = [" " * indent + func_part]
                        for i, arg in enumerate(args):
                            if i == len(args) - 1:
                                result.append(" " * (indent + 4) + arg + end_part)
                            else:
                                result.append(" " * (indent + 4) + arg + ",")
                        return result

    return line


def fix_undefined_names_safe() -> bool:
    """Fix undefined names where it's safe to do so."""
    changes_made = False

    # This is complex and risky, so we'll only fix obvious cases
    # For now, we'll skip this to avoid breaking code

    return changes_made


def fix_exception_handling_comprehensive() -> bool:
    """Comprehensively improve exception handling."""
    files_to_fix = [
        "replicants/python/folder_packer_pro/folder_packer_pro.py",
        "replicants/python/folder_tool_pro/folder_fix_pro.py",
    ]

    changes_made = False

    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Improve exception handling patterns
            # Add specific exception types where safe
            content = re.sub(
                r"except Exception as e:\s*\n(\s+)messagebox\.showerror",
                r'except (OSError, ValueError, TypeError) as e:\n\1logger.exception("Operation failed")\n\1messagebox.showerror',
                content,
            )

            # Improve bare except clauses
            content = re.sub(r"except:\s*\n", "except Exception:\n", content)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Improved exception handling in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_inline_imports_comprehensive() -> bool:
    """Fix all inline imports by moving them to the top."""
    changes_made = False

    # This was already handled in the import restructuring
    # But let's make sure any remaining inline imports are caught

    files_to_check = [
        "replicants/python/folder_tool_pro/folder_fix_pro.py",
    ]

    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Remove any remaining inline imports since they should be at the top now
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                # Skip inline imports that are inside functions
                if "import webbrowser" in line and any(
                    "def " in prev_line
                    for prev_line in lines[
                        max(0, lines.index(line) - 10) : lines.index(line)
                    ]
                ):
                    continue
                new_lines.append(line)

            content = "\n".join(new_lines)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Removed inline imports in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def main() -> None:
    """Main function to apply aggressive fixes."""
    print("🚀 Applying aggressive fixes for remaining issues...")

    fixes_applied = 0

    if fix_import_organization_aggressive():
        fixes_applied += 1

    if fix_line_length_aggressively():
        fixes_applied += 1

    if fix_exception_handling_comprehensive():
        fixes_applied += 1

    if fix_inline_imports_comprehensive():
        fixes_applied += 1

    print(f"\n✅ Applied {fixes_applied} categories of aggressive fixes")

    # Run Black to format the fixed files
    print("\n🎨 Running Black to format fixed files...")
    import subprocess

    try:
        result = subprocess.run(
            [
                "black",
                "--exclude=data_processing/data_processor/python/data_processor/Data_Processor_r0.py",
                ".",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ Black formatting completed successfully")
        else:
            print(f"⚠️ Black formatting had issues: {result.stderr}")
    except Exception as e:
        print(f"❌ Could not run Black: {e}")

    # Apply any remaining auto-fixes
    print("\n🔧 Applying remaining auto-fixes...")
    try:
        result = subprocess.run(
            [
                "ruff",
                "check",
                "--exclude=data_processing/data_processor/python/data_processor/Data_Processor_r0.py",
                ".",
                "--fix",
            ],
            capture_output=True,
            text=True,
        )
        print("✅ Auto-fixes applied")
    except Exception as e:
        print(f"❌ Could not run Ruff auto-fix: {e}")

    # Final status check
    print("\n📊 Final status check...")
    try:
        result = subprocess.run(
            [
                "ruff",
                "check",
                "--exclude=data_processing/data_processor/python/data_processor/Data_Processor_r0.py",
                ".",
                "--statistics",
            ],
            capture_output=True,
            text=True,
        )
        print("Current status:")
        print(result.stdout)

        # Extract total count
        lines = result.stdout.strip().split("\n")
        total_line = [line for line in lines if "Found" in line and "errors" in line]
        if total_line:
            current_count = int(total_line[0].split()[1])
            improvement = 293 - current_count
            print(f"\n📈 Progress: Reduced from 293 to {current_count} issues")
            print(f"🎯 Improvement: {improvement} additional issues resolved")

            if current_count < 250:
                print("🎉 Excellent progress! Under 250 issues remaining!")
            elif current_count < 280:
                print("✅ Good progress! Significant improvement achieved!")

    except Exception as e:
        print(f"❌ Could not run final status check: {e}")


if __name__ == "__main__":
    main()
