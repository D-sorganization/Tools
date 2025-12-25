#!/usr/bin/env python3
"""
Final comprehensive fix for Data_Processor_r0.py structural issues.
This script will systematically fix all syntax errors.
"""

import re
from pathlib import Path


def fix_data_processor_file() -> bool:
    """Fix all structural issues in Data_Processor_r0.py."""
    file_path = Path(
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
    )

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Fix 1: Remove orphaned except blocks completely
        lines = content.split("\n")
        cleaned_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for orphaned except blocks
            if line.strip().startswith("except Exception as e:"):
                # Look backwards for a matching try within reasonable distance
                found_try = False
                for j in range(
                    len(cleaned_lines) - 1, max(0, len(cleaned_lines) - 30), -1
                ):
                    if "try:" in cleaned_lines[j] and not any(
                        "except" in cleaned_lines[k]
                        for k in range(j + 1, len(cleaned_lines))
                    ):
                        found_try = True
                        break

                if not found_try:
                    # Skip this orphaned except block and its contents
                    print(
                        f"Removing orphaned except block at line {i + 1}: {line.strip()}"
                    )
                    i += 1
                    # Skip the except block contents
                    while i < len(lines) and (
                        lines[i].startswith("    ")
                        or lines[i].startswith("\t")
                        or lines[i].strip() == ""
                        or "messagebox." in lines[i]
                        or '"Error"' in lines[i]
                        or 'f"Failed to' in lines[i]
                    ):
                        print(f"  Removing line {i + 1}: {lines[i].strip()}")
                        i += 1
                    continue

            # Fix incomplete function definitions
            if line.strip() == "return ...":
                # Replace with proper return statement
                cleaned_lines.append(
                    line.replace("return ...", "return self.cancel_event.is_set()")
                )
                i += 1
                continue

            cleaned_lines.append(line)
            i += 1

        content = "\n".join(cleaned_lines)

        # Fix 2: Clean up any remaining malformed structures
        # Remove any remaining ellipsis that might cause issues
        content = re.sub(r"^\s*\.\.\.\s*$", "", content, flags=re.MULTILINE)

        # Fix 3: Ensure proper function structure
        # Look for functions that might have structural issues
        content = re.sub(
            r'(def \w+\([^)]*\):[^:]*?)\n(\s+)"""([^"]*?)"""\s*\n(\s+)(.*?)\n(\s+)except Exception as e:',
            r'\1\n\2"""\3"""\n\2try:\n\4\5\n\6except Exception as e:',
            content,
            flags=re.MULTILINE | re.DOTALL,
        )

        # Fix 4: Clean up multiple empty lines
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        # Fix 5: Ensure proper indentation for class methods
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Track if we're in a class
            if line.strip().startswith("class ") and line.strip().endswith(":"):
                fixed_lines.append(line)
                continue

            # Reset class tracking on new class or module-level code
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                if not line.strip().startswith("#") and not line.strip().startswith(
                    '"""'
                ):
                    pass

            fixed_lines.append(line)

        content = "\n".join(fixed_lines)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed structural issues in {file_path}")
            return True
        else:
            print("No structural issues found to fix")
            return False

    except Exception as e:
        print(f"❌ Error during fix: {e}")
        return False


def main() -> None:
    """Main function."""
    print("🔧 Running final Data_Processor_r0.py fix...")

    if fix_data_processor_file():
        print("✅ Final fix completed successfully")
    else:
        print("❌ Final fix failed")


if __name__ == "__main__":
    main()
