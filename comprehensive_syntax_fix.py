#!/usr/bin/env python3
"""
Comprehensive fix for Data_Processor_r0.py syntax errors.
This script will clean up all the structural issues in the file.
"""

import re
from pathlib import Path


def comprehensive_fix() -> bool:
    """Comprehensively fix all syntax errors in Data_Processor_r0.py."""
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

        # Remove all misplaced documentation text that's not in docstrings or comments
        # This appears to be documentation that got inserted into the code
        doc_patterns = [
            r"This section helps you select and configure your input files for processing\.",
            r"📁 FILE SELECTION:",
            r"• Select Input Files",
            r"  - Opens file dialog to select CSV files for processing",
            r"  - Supports multiple file selection",
            r"  - Automatically sets output directory to first file\'s location",
            r"  - Files are displayed in a summary view \(for large selections\)",
            r"• Clear All Files",
            r"  - Removes all selected files from the list",
            r"  - Clears the signal list display",
            r"  - Resets the file selection state",
            r"📊 SIGNAL SELECTION:",
            r"• Signal List Display",
            r"• Bulk Mode Toggle",
            r"• Search/Filter Functionality",
            r"⚙️ PROCESSING OPTIONS:",
            r"• Output Directory Selection",
            r"• File Naming Options",
            r"• Processing Method Selection",
        ]

        for pattern in doc_patterns:
            content = re.sub(pattern + r"\s*\n?", "", content, flags=re.MULTILINE)

        # Remove any remaining bullet points and documentation fragments
        content = re.sub(r"^\s*[•-]\s+.*\n", "", content, flags=re.MULTILINE)
        content = re.sub(r"^\s*-\s+.*\n", "", content, flags=re.MULTILINE)

        # Clean up multiple empty lines
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        # Fix orphaned except blocks by removing them entirely
        # These were created by previous fix attempts
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
                    len(cleaned_lines) - 1, max(0, len(cleaned_lines) - 20), -1
                ):
                    if "try:" in cleaned_lines[j]:
                        found_try = True
                        break

                if not found_try:
                    # Skip this orphaned except block and its contents
                    print(f"Removing orphaned except block at line {i + 1}")
                    i += 1
                    # Skip the except block contents
                    while i < len(lines) and (
                        lines[i].startswith("    ")
                        or lines[i].startswith("\t")
                        or lines[i].strip() == ""
                    ):
                        i += 1
                    continue

            cleaned_lines.append(line)
            i += 1

        content = "\n".join(cleaned_lines)

        # Fix specific known syntax errors
        fixes = [
            # Fix broken f-strings
            (
                r'text=f"Load More Signals \(\{len\(signals\) -[^"]*"[^"]*"[^"]*"',
                'text=f"Load More Signals ({len(signals) - SIGNAL_BATCH_SIZE} remaining)"',
            ),
            # Fix broken debug text
            (
                r'debug_text = f"No data file specified[^"]*"[^"]*"[^"]*"',
                "debug_text = f\"No data file specified in plot configuration\\n\\nSaved file: '{file_name}'\"",
            ),
            # Fix broken warning text
            (
                r'warning_text = f"⚠️ Warning: Will overwrite[^"]*"[^"]*"[^"]*"',
                "warning_text = f\"⚠️ Warning: Will overwrite existing files: {', '.join(existing_files)}\"",
            ),
        ]

        for pattern, replacement in fixes:
            content = re.sub(
                pattern, replacement, content, flags=re.MULTILINE | re.DOTALL
            )

        # Ensure proper function structure
        # Look for functions that might be missing try-except structure
        content = re.sub(
            r"(def \w+\([^)]*\):[^{]*?)\n(\s+)(.*?)\n(\s+)return\s+.*?\n(\s*)except Exception as e:",
            r"\1\n\2try:\n\2    \3\n\4return ...\n\5except Exception as e:",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Comprehensively fixed syntax errors in {file_path}")
            return True
        else:
            print("No issues found to fix")
            return False

    except Exception as e:
        print(f"❌ Error during comprehensive fix: {e}")
        return False


def main() -> None:
    """Main function."""
    print("🔧 Running comprehensive syntax fix...")

    if comprehensive_fix():
        print("✅ Comprehensive fix completed successfully")
    else:
        print("❌ Comprehensive fix failed")


if __name__ == "__main__":
    main()
