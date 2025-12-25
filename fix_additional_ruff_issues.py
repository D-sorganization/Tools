#!/usr/bin/env python3
"""
Continue fixing remaining Ruff issues for better code quality.
Focus on the most impactful fixes that can be automated safely.
"""

import re
from pathlib import Path


def fix_import_organization() -> bool:
    """Fix import organization issues (E402)."""
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

            # Extract docstring and imports
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
                    if stripped.count(docstring_quotes) >= 2:  # Single line docstring
                        docstring_end = i + 1
                        break
                elif in_docstring and docstring_quotes in stripped:
                    docstring_end = i + 1
                    break

            # Collect all imports
            imports = []
            non_imports = []

            for i, line in enumerate(lines[docstring_end:], docstring_end):
                stripped = line.strip()
                if (
                    stripped.startswith("import ")
                    or stripped.startswith("from ")
                    or stripped == ""
                    or stripped.startswith("#")
                ):
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        imports.append(line)
                    elif stripped == "" and imports:  # Keep blank lines between imports
                        imports.append(line)
                else:
                    # Found first non-import line
                    non_imports = lines[i:]
                    break

            if imports:
                # Reorganize imports
                organized_imports = organize_imports(imports)

                # Reconstruct file
                new_lines = (
                    lines[:docstring_end] + organized_imports + [""] + non_imports
                )
                new_content = "\n".join(new_lines)

                if new_content != original_content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"✅ Fixed import organization in {file_path}")
                    changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def organize_imports(import_lines: list[str]) -> list[str]:
    """Organize imports according to PEP 8."""
    stdlib_imports = []
    third_party_imports = []
    local_imports = []

    # Standard library modules (common ones)
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
        elif module in ["cryptography"]:  # Known third-party
            third_party_imports.append(line)
        else:
            local_imports.append(line)

    # Combine with proper spacing
    result = []
    if stdlib_imports:
        result.extend(stdlib_imports)
        result.append("")
    if third_party_imports:
        result.extend(third_party_imports)
        result.append("")
    if local_imports:
        result.extend(local_imports)

    return result


def fix_line_length_issues() -> bool:
    """Fix line length issues in comments and strings."""
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
                    # Try to fix long lines
                    fixed_line = fix_long_line(line)
                    if isinstance(fixed_line, list):
                        new_lines.extend(fixed_line)
                    else:
                        new_lines.append(fixed_line)
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


def fix_long_line(line: str) -> str | list[str]:
    """Fix a single long line."""
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
            ]:
                if break_point in comment_text:
                    parts = comment_text.split(break_point, 1)
                    if 20 < len(parts[0]) < 75 and len(parts[1]) < 75:
                        return [
                            " " * indent + "# " + parts[0] + break_point.rstrip(),
                            " " * indent + "# " + parts[1],
                        ]

    # Handle string literals in dictionary definitions
    if '": "' in stripped and stripped.endswith('",'):
        # This is likely a dictionary entry
        key_end = stripped.find('": "')
        if key_end > 0:
            key_part = stripped[: key_end + 3]
            value_part = stripped[key_end + 3 : -2]  # Remove ": and ",

            if len(key_part) + len(value_part) > 80:
                # Break after the colon
                return [
                    " " * indent + key_part,
                    " " * indent + '    "' + value_part + '",',
                ]

    return line


def fix_exception_handling() -> bool:
    """Fix bare except clauses and blind exceptions."""
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

            # Fix try-except-pass patterns
            content = re.sub(
                r"(\s+)except Exception:\s*\n\s*# Not compressed\s*\n\s*pass",
                r"\1except Exception:\n\1    # Not compressed - this is expected for uncompressed files\n\1    pass",
                content,
            )

            # Add logging to bare exceptions where appropriate
            content = re.sub(
r'(\s +
                    )except Exception as e:\s*\n(\s +
                    )self\._log_message\(f"Error ([^"] +
                    ): \{e\}", "error"\)',"
                r'\1except Exception as e:\n\2logger.exception(
                    "Error \3")\n\2self._log_message(f"Error \3: {e}",
                    "error"
                )',
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


def fix_pathlib_usage() -> bool:
    """Replace open() with Path.open() where safe."""
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

            # Replace simple open() calls with Path.open()
            # Only replace where we have a clear Path object
            replacements = [
                (
                    r'with open\(file_path, "rb"\) as f:',
                    'with file_path.open("rb") as f:',
                ),
                (
                    r'with open\(file_path, encoding="utf-8", errors="ignore"\) as f:',
                    'with file_path.open(encoding="utf-8", errors="ignore") as f:',
                ),
                (
                    r'with open\(output_path, "wb"\) as f:',
                    'with output_path.open("wb") as f:',
                ),
                (
                    r'with open\(
                        manifest_path,
                        "w",
                        encoding="utf-8"\
                    ) as manifest_file:',
                    'with manifest_path.open("w", encoding="utf-8") as manifest_file:',
                ),
                (
                    r'with open\(package_path, "rb"\) as f:',
                    'with package_path.open("rb") as f:',
                ),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Updated to use Path.open() in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_logging_fstring() -> bool:
    """Fix f-string usage in logging statements."""
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

            # Fix f-string in logging
            content = re.sub(
                r'logger\.info\(f"SUCCESS: \{message\}"\)',
                'logger.info("SUCCESS: %s", message)',
                content,
            )

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed logging f-string usage in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def main() -> None:
    """Main function to fix additional Ruff issues."""
    print("🔧 Fixing additional Ruff issues...")

    fixes_applied = 0

    if fix_import_organization():
        fixes_applied += 1

    if fix_line_length_issues():
        fixes_applied += 1

    if fix_exception_handling():
        fixes_applied += 1

    if fix_pathlib_usage():
        fixes_applied += 1

    if fix_logging_fstring():
        fixes_applied += 1

    print(f"\n✅ Applied {fixes_applied} categories of additional fixes")

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


if __name__ == "__main__":
    main()
