#!/usr/bin/env python3
"""
Final comprehensive fix for remaining Ruff issues.
Focus on the most impactful and safe automated fixes.
"""

import re
from pathlib import Path


def fix_import_organization_comprehensive() -> bool:
    """Comprehensively fix import organization issues."""
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

            # For these specific files, we need to move all imports to the top
            # after the module docstring
            lines = content.split("\n")

            # Find the end of the module docstring
            docstring_end = 0
            in_docstring = False
            docstring_quotes = None

            for i, line in enumerate(lines):
                stripped = line.strip()
                if not in_docstring and (
                    stripped.startswith('"""') or stripped.startswith("'''")
                ):
                    docstring_quotes = stripped[:3]
                    in_docstring = True
                    if stripped.count(docstring_quotes) >= 2 and len(stripped) > 3:
                        # Single line docstring
                        docstring_end = i + 1
                        break
                elif in_docstring and docstring_quotes in stripped:
                    docstring_end = i + 1
                    break

            # Extract all imports and non-import code
            imports = []
            non_import_lines = []
            found_first_non_import = False

            for i, line in enumerate(lines[docstring_end:], docstring_end):
                stripped = line.strip()

                if (
                    stripped.startswith("import ")
                    or stripped.startswith("from ")
                    or (stripped == "" and not found_first_non_import)
                    or stripped.startswith("#")
                ):

                    if not found_first_non_import:
                        if stripped.startswith("import ") or stripped.startswith(
                            "from "
                        ):
                            imports.append(line)
                        elif stripped == "" and imports:
                            imports.append(line)
                        # Skip comments and empty lines before imports
                else:
                    found_first_non_import = True
                    non_import_lines.extend(lines[i:])
                    break

            if imports:
                # Organize imports properly
                organized_imports = organize_imports_properly(imports)

                # Reconstruct the file
                new_content = "\n".join(
                    lines[:docstring_end] + organized_imports + [""] + non_import_lines
                )

                if new_content != original_content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"✅ Comprehensively fixed imports in {file_path}")
                    changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def organize_imports_properly(import_lines: list[str]) -> list[str]:
    """Organize imports according to PEP 8 with proper grouping."""
    stdlib_imports = []
    third_party_imports = []

    # Standard library modules
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
    }

    for line in import_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Determine import type
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


def fix_line_length_comprehensive() -> bool:
    """Fix line length issues comprehensively."""
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
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                if len(line.rstrip()) > 88:
                    fixed_lines = fix_long_line_comprehensive(line)
                    if isinstance(fixed_lines, list):
                        new_lines.extend(fixed_lines)
                    else:
                        new_lines.append(fixed_lines)
                else:
                    new_lines.append(line)

            new_content = "\n".join(new_lines)

            if new_content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"✅ Fixed line length issues in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_long_line_comprehensive(line: str) -> str | list[str]:
    """Fix a single long line with comprehensive strategies."""
    indent = len(line) - len(line.lstrip())
    stripped = line.strip()

    # Handle comments
    if stripped.startswith("#"):
        comment_text = stripped[1:].strip()
        if len(comment_text) > 80:
            # Try to break at logical points
            for break_point in [
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
            ]:
                if break_point in comment_text:
                    parts = comment_text.split(break_point, 1)
                    if 20 < len(parts[0]) < 75 and len(parts[1]) < 75:
                        return [
                            " " * indent + "# " + parts[0] + break_point.rstrip(),
                            " " * indent + "# " + parts[1],
                        ]

    # Handle docstrings
    if '"""' in stripped and not stripped.startswith("#"):
        # Try to break long docstrings
        if "def " in line or "class " in line:
            # This is a function/class definition with docstring
            if len(stripped) > 88:
                # Break after the colon
                colon_pos = line.find(":")
                if colon_pos > 0 and colon_pos < 80:
                    return [
                        line[: colon_pos + 1],
                        " " * (indent + 4) + '"""' + stripped.split('"""')[1] + '"""',
                    ]

    # Handle string literals in dictionaries
    if '": "' in stripped and stripped.endswith('",'):
        key_end = stripped.find('": "')
        if key_end > 0:
            key_part = stripped[: key_end + 3]
            value_part = stripped[key_end + 3 : -2]

            if len(key_part) + len(value_part) > 80:
                return [
                    " " * indent + key_part,
                    " " * indent + '    "' + value_part + '",',
                ]

    # Handle long function calls or method chains
    if "(" in stripped and ")" in stripped and len(stripped) > 88:
        # Try to break at function parameters
        paren_pos = stripped.find("(")
        if paren_pos > 0 and paren_pos < 60:
            before_paren = stripped[: paren_pos + 1]
            after_paren = stripped[paren_pos + 1 :]

            if ", " in after_paren:
                # Break at parameter boundaries
                params = after_paren.split(", ")
                if len(params) > 1:
                    result = [" " * indent + before_paren]
                    for _, param in enumerate(params[:-1]):
                        result.append(" " * (indent + 4) + param + ",")
                    result.append(" " * (indent + 4) + params[-1])
                    return result

    return line


def fix_remaining_pathlib_usage() -> bool:
    """Fix remaining Path.open() usage."""
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

            # Replace remaining open() calls with Path.open()
            replacements = [
                (
                    r'with open\(file_path, "wb"\) as f:',
                    'with file_path.open("wb") as f:',
                ),
                (
                    r'with open\(file_path, "w"\) as f:',
                    'with file_path.open("w") as f:',
                ),
                (
                    r'with open\(file_path, "w", encoding="utf-8"\) as f:',
                    'with file_path.open("w", encoding="utf-8") as f:',
                ),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed remaining Path.open() usage in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_inline_imports() -> bool:
    """Fix inline imports by moving them to the top."""
    files_to_fix = [
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

            # Add webbrowser import at the top if it's used inline
            if (
                "import webbrowser" in content
                and "import webbrowser\n" not in content[:500]
            ):
                # Find the import section
                lines = content.split("\n")
                import_section_end = 0

                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith(
                        "from "
                    ):
                        import_section_end = i + 1
                    elif line.strip() and not line.strip().startswith("#"):
                        break

                # Insert webbrowser import
                lines.insert(import_section_end, "import webbrowser")

                # Remove inline import
                new_lines = []
                for line in lines:
                    if line.strip() == "import webbrowser" and "def " in "\n".join(
                        lines[max(0, lines.index(line) - 5) : lines.index(line)]
                    ):
                        continue
                    new_lines.append(line)

                content = "\n".join(new_lines)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed inline imports in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def main() -> None:
    """Main function to apply final Ruff fixes."""
    print("🔧 Applying final comprehensive Ruff fixes...")

    fixes_applied = 0

    if fix_import_organization_comprehensive():
        fixes_applied += 1

    if fix_line_length_comprehensive():
        fixes_applied += 1

    if fix_remaining_pathlib_usage():
        fixes_applied += 1

    if fix_inline_imports():
        fixes_applied += 1

    print(f"\n✅ Applied {fixes_applied} categories of final fixes")

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

    # Final Ruff check
    print("\n📊 Running final Ruff check...")
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
        print("Final Ruff statistics:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"❌ Could not run final Ruff check: {e}")


if __name__ == "__main__":
    main()
