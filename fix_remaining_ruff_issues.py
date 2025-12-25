#!/usr/bin/env python3
"""
Fix remaining Ruff issues to improve code quality.
Focus on the most impactful and automatable fixes.
"""

import re
from pathlib import Path


def fix_line_length_issues() -> bool:
    """Fix line length violations in documentation and comments."""
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

            # Fix long documentation lines
            content = fix_long_doc_lines(content)

            # Fix long comments
            content = fix_long_comment_lines(content)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed line length issues in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_long_doc_lines(content: str) -> str:
    """Fix long lines in documentation strings."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if len(line.rstrip()) > 88:
            # Check if it's a documentation line (starts with text, contains descriptions)
            if (
                line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith('"""')
                and (
                    "**" in line
                    or "processing" in line.lower()
                    or "analysis" in line.lower()
                )
            ):

                # Try to break at logical points
                for break_point in [
                    " - ",
                    ": ",
                    ", ",
                    " with ",
                    " and ",
                    " for ",
                    " in ",
                ]:
                    if break_point in line and len(line) > 88:
                        parts = line.split(break_point, 1)
                        if len(parts[0]) < 85 and len(parts[1]) > 10:
                            indent = len(line) - len(line.lstrip())
                            new_lines.append(parts[0] + break_point.rstrip())
                            new_lines.append(" " * indent + parts[1])
                            break
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def fix_long_comment_lines(content: str) -> str:
    """Fix long comment lines."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if (
            line.strip().startswith("#")
            and len(line.rstrip()) > 88
            and not line.strip().startswith("# ===")
        ):  # Don't break separator comments

            indent = len(line) - len(line.lstrip())
            comment_text = line.strip()[1:].strip()

            # Try to break at logical points
            for break_point in [" - ", ": ", ", ", " and ", " or ", " but ", " with "]:
                if break_point in comment_text and len(comment_text) > 60:
                    parts = comment_text.split(break_point, 1)
                    if 20 < len(parts[0]) < 75 and len(parts[1]) < 75:
                        new_lines.append(
                            " " * indent + "# " + parts[0] + break_point.rstrip()
                        )
                        new_lines.append(" " * indent + "# " + parts[1])
                        break
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def fix_magic_numbers() -> bool:
    """Fix magic numbers by adding constants."""
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

            # Add constants at the top of the file (after imports)
            if "PREVIEW_FILE_LIMIT = 500" not in content:
                # Find the last import line
                lines = content.split("\n")
                import_end = 0
                for i, line in enumerate(lines):
                    if (
                        line.strip().startswith("import ")
                        or line.strip().startswith("from ")
                        or line.strip() == ""
                    ):
                        import_end = i
                    elif line.strip() and not line.strip().startswith("#"):
                        break

                # Insert constants after imports
                constants = [
                    "",
                    "# Constants",
                    "PREVIEW_FILE_LIMIT = 500  # Maximum files to show in preview",
                    "PREVIEW_LINE_LIMIT = 1000  # Maximum lines to show in file preview",
                    "BYTES_PER_KB = 1024.0  # Bytes in a kilobyte",
                    "",
                ]

                lines[import_end:import_end] = constants
                content = "\n".join(lines)

            # Replace magic numbers with constants
            replacements = [
                (r"if len\(files\) >= 500:", "if len(files) >= PREVIEW_FILE_LIMIT:"),
                (r"if i >= 1000:", "if i >= PREVIEW_LINE_LIMIT:"),
                (r"if size < 1024\.0:", "if size < BYTES_PER_KB:"),
                (r"size /= 1024\.0", "size /= BYTES_PER_KB"),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed magic numbers in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_import_issues() -> bool:
    """Fix import-related issues."""
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

            # Move imports to top of file
            if "import webbrowser" in content:
                # Add webbrowser import at the top
                lines = content.split("\n")

                # Find where to insert the import
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith(
                        "from "
                    ):
                        insert_pos = i + 1
                    elif line.strip() and not line.strip().startswith("#"):
                        break

                # Check if webbrowser import already exists at top
                has_webbrowser_import = any(
                    "import webbrowser" in line for line in lines[:insert_pos]
                )

                if not has_webbrowser_import:
                    lines.insert(insert_pos, "import webbrowser")

                # Remove inline imports
                new_lines = []
                for line in lines:
                    if line.strip() == "import webbrowser" and "def " in "\n".join(
                        lines[max(0, lines.index(line) - 5) : lines.index(line)]
                    ):
                        # Skip inline import
                        continue
                    new_lines.append(line)

                content = "\n".join(new_lines)

            # Fix typing imports
            if "from typing import Any, Final" in content:
                content = content.replace(
                    "from typing import Any, Final",
                    "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from typing import Any, Final",
                )

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed import issues in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_datetime_issues() -> bool:
    """Fix datetime timezone issues."""
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

            # Add timezone import if needed
            if (
                "datetime.now()" in content
                and "from datetime import timezone" not in content
            ):
                content = content.replace(
                    "from datetime import datetime",
                    "from datetime import datetime, timezone",
                )

            # Fix datetime.now() calls
            content = re.sub(
                r"datetime\.now\(\)", "datetime.now(timezone.utc)", content
            )

            # Fix datetime.fromtimestamp() calls
            content = re.sub(
                r"datetime\.fromtimestamp\(([^)]+)\)",
                r"datetime.fromtimestamp(\1, timezone.utc)",
                content,
            )

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed datetime issues in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def main() -> None:
    """Main function to fix remaining Ruff issues."""
    print("🔧 Fixing remaining Ruff issues...")

    fixes_applied = 0

    if fix_line_length_issues():
        fixes_applied += 1

    if fix_magic_numbers():
        fixes_applied += 1

    if fix_import_issues():
        fixes_applied += 1

    if fix_datetime_issues():
        fixes_applied += 1

    print(f"\n✅ Applied {fixes_applied} categories of fixes")

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
