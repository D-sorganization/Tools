#!/usr/bin/env python3
"""
Comprehensive fix for remaining Ruff issues.
Target the most impactful remaining problems systematically.
"""

import re
from pathlib import Path


def fix_import_organization_complete() -> bool:
    """Completely fix import organization by restructuring files."""
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

            # Completely restructure the file with proper import organization
            new_content = restructure_file_imports(content)

            if new_content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"✅ Completely restructured imports in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def restructure_file_imports(content: str) -> str:
    """Restructure file to have proper import organization."""
    lines = content.split("\n")

    # Find module docstring
    docstring_start = -1
    docstring_end = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"""') and docstring_start == -1:
            docstring_start = i
            if stripped.count('"""') >= 2 and len(stripped) > 3:
                # Single line docstring
                docstring_end = i
                break
        elif docstring_start != -1 and '"""' in stripped:
            docstring_end = i
            break

    # Extract sections
    header_section = lines[: docstring_end + 1] if docstring_end >= 0 else []

    # Collect all imports and non-import code
    imports = []
    code_lines = []
    found_code = False

    start_idx = docstring_end + 1 if docstring_end >= 0 else 0

    for i, line in enumerate(lines[start_idx:], start_idx):
        stripped = line.strip()

        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or (stripped == "" and not found_code)
            or (stripped.startswith("#") and not found_code)
        ):

            if not found_code:
                if stripped.startswith("import ") or stripped.startswith("from "):
                    imports.append(line)
                # Skip empty lines and comments before code starts
        else:
            found_code = True
            code_lines.extend(lines[i:])
            break

    # Organize imports properly
    organized_imports = organize_imports_by_type(imports)

    # Reconstruct file
    result = header_section + [""] + organized_imports + [""] + code_lines

    return "\n".join(result)


def organize_imports_by_type(import_lines: list[str]) -> list[str]:
    """Organize imports by type with proper grouping."""
    stdlib_imports = []
    third_party_imports = []

    # Standard library modules (comprehensive list)
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
    }

    for line in import_lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Extract module name
        if stripped.startswith("from "):
            module = stripped.split()[1].split(".")[0]
        elif stripped.startswith("import "):
            module = stripped.split()[1].split(".")[0]
        else:
            continue

        if module in stdlib_modules:
            stdlib_imports.append(line)
        else:
            third_party_imports.append(line)

    # Sort imports within each group
    stdlib_imports.sort(key=lambda x: x.strip())
    third_party_imports.sort(key=lambda x: x.strip())

    # Combine with proper spacing
    result = []
    if stdlib_imports:
        result.extend(stdlib_imports)
    if third_party_imports:
        if stdlib_imports:
            result.append("")
        result.extend(third_party_imports)

    return result


def fix_line_length_systematically() -> bool:
    """Systematically fix line length issues."""
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

            # Fix line length issues systematically
            content = fix_long_lines_systematically(content)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed line length issues systematically in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_long_lines_systematically(content: str) -> str:
    """Fix long lines using systematic approaches."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if len(line.rstrip()) > 88:
            fixed_line = apply_line_fixes(line)
            if isinstance(fixed_line, list):
                new_lines.extend(fixed_line)
            else:
                new_lines.append(fixed_line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def apply_line_fixes(line: str) -> str | list[str]:
    """Apply various line fixing strategies."""
    indent = len(line) - len(line.lstrip())
    stripped = line.strip()

    # Strategy 1: Fix long comments
    if stripped.startswith("#"):
        return fix_long_comment(line, indent)

    # Strategy 2: Fix long f-strings
    if 'f"' in stripped and len(stripped) > 88:
        return fix_long_fstring(line, indent)

    # Strategy 3: Fix long function calls
    if "(" in stripped and ")" in stripped and len(stripped) > 88:
        return fix_long_function_call(line, indent)

    # Strategy 4: Fix long string concatenations
    if " + " in stripped and '"' in stripped:
        return fix_long_string_concat(line, indent)

    # Strategy 5: Fix long dictionary entries
    if '": "' in stripped and stripped.endswith('",'):
        return fix_long_dict_entry(line, indent)

    return line


def fix_long_comment(line: str, indent: int) -> str | list[str]:
    """Fix long comment lines."""
    comment_text = line.strip()[1:].strip()

    if len(comment_text) > 80:
        # Find good break points
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
        ]

        for break_point in break_points:
            if break_point in comment_text:
                parts = comment_text.split(break_point, 1)
                if 20 < len(parts[0]) < 75 and len(parts[1]) < 75:
                    return [
                        " " * indent + "# " + parts[0] + break_point.rstrip(),
                        " " * indent + "# " + parts[1],
                    ]

    return line


def fix_long_fstring(line: str, indent: int) -> str | list[str]:
    """Fix long f-string lines."""
    # Try to break f-strings at logical points
    if 'f"' in line and "{" in line and "}" in line:
        # Simple case: break into multiple f-strings
        parts = line.split('f"')
        if len(parts) > 1:
            # This is complex, return original for now
            pass

    return line


def fix_long_function_call(line: str, indent: int) -> str | list[str]:
    """Fix long function call lines."""
    stripped = line.strip()

    # Find function call pattern
    paren_pos = stripped.find("(")
    if paren_pos > 0 and paren_pos < 60:
        func_part = stripped[: paren_pos + 1]
        args_part = stripped[paren_pos + 1 : -1]  # Remove closing paren

        if ", " in args_part and len(args_part) > 40:
            # Break at argument boundaries
            args = args_part.split(", ")
            if len(args) > 1:
                result = [" " * indent + func_part]
                for i, arg in enumerate(args):
                    if i == len(args) - 1:
                        result.append(" " * (indent + 4) + arg + ")")
                    else:
                        result.append(" " * (indent + 4) + arg + ",")
                return result

    return line


def fix_long_string_concat(line: str, indent: int) -> str | list[str]:
    """Fix long string concatenation lines."""
    # This is complex, return original for now
    return line


def fix_long_dict_entry(line: str, indent: int) -> str | list[str]:
    """Fix long dictionary entry lines."""
    stripped = line.strip()

    if '": "' in stripped and stripped.endswith('",'):
        colon_pos = stripped.find('": "')
        key_part = stripped[: colon_pos + 3]
        value_part = stripped[colon_pos + 3 : -2]

        if len(key_part) + len(value_part) > 80:
            return [
                " " * indent + key_part,
                " " * (indent + 4) + '"' + value_part + '",',
            ]

    return line


def fix_undefined_names() -> bool:
    """Fix undefined name issues where possible."""
    # This would require more complex analysis
    # For now, we'll focus on the safer fixes
    return False


def fix_exception_handling_improvements() -> bool:
    """Improve exception handling where safe."""
    files_to_fix = [
        "replicants/python/folder_packer_pro/folder_packer_pro.py",
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

            # Improve specific exception handling patterns
            # Add logging to exception handlers where it makes sense
            content = re.sub(
                r"(\s+)except Exception as e:\s*\n(\s+)messagebox\.showerror\(",
                r'\1except Exception as e:\n\2logger.exception("Error occurred")\n\2messagebox.showerror(',
                content,
            )

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Improved exception handling in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def main() -> None:
    """Main function to apply comprehensive fixes."""
    print("🔧 Applying comprehensive remaining fixes...")

    fixes_applied = 0

    if fix_import_organization_complete():
        fixes_applied += 1

    if fix_line_length_systematically():
        fixes_applied += 1

    if fix_exception_handling_improvements():
        fixes_applied += 1

    print(f"\n✅ Applied {fixes_applied} categories of comprehensive fixes")

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

    # Apply remaining auto-fixes
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
        if result.returncode == 0:
            print("✅ All auto-fixable issues resolved")
        else:
            print("⚠️ Some issues remain (expected - these are quality suggestions)")
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
        print("Remaining issues:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Could not run final status check: {e}")


if __name__ == "__main__":
    main()
