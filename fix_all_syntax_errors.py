#!/usr/bin/env python3
"""
Fix all syntax errors in Data_Processor_r0.py by removing orphaned try-except blocks.
"""

import re
from pathlib import Path


def fix_all_syntax_errors() -> bool:
    """Fix all syntax errors in the Data_Processor_r0.py file."""
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

        # Remove all orphaned try-except blocks that were incorrectly added
        # Pattern 1: Remove standalone try-except blocks with placeholder
        content = re.sub(
            r'\s*try:\s*\n\s*pass\s*#\s*Placeholder for try block\s*\n\s*except Exception as e:\s*\n\s*print\(f?"[^"]*"\)\s*\n?',
            "",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 2: Remove orphaned try blocks with pass placeholder
        content = re.sub(
            r"\s*try:\s*\n\s*pass\s*#\s*Placeholder for try block\s*\n",
            "",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 3: Remove specific orphaned except blocks
        content = re.sub(
            r'\s*except Exception as e:\s*\n\s*print\(
                \s*f?"[^"]*",
                ?\s*\n\s*"[^"]*",
                ?\s*\
            )\s*\n\s*df\[f?"[^"]*"\]\s*=\s*np\.nan\s*\n',
            "",
            content,
            flags=re.MULTILINE,
        )

        # Fix broken f-strings and string concatenations
        # Fix the specific broken f-string patterns
        content = re.sub(
            r'f"(
                [^"]*)\{([^}]*)\}([^"]*)",
                \s*\n\s*"([^"]*
            )"', r'f"\1{\2}\3\4"', content"
        )

        # Fix broken string literals that span multiple lines incorrectly
        content = re.sub(r'"([^"]*)",\s*\n\s*"([^"]*)"', r'"\1\2"', content)

        # Fix specific syntax errors we know about
        fixes = [
            # Fix the load more button text
            (
                r'text=f"Load More Signals \(
                    \{len\(signals\) -",
                    \s*"SIGNAL_BATCH_SIZE\} remaining\
                )"',
                r'text=f"Load More Signals ({len(signals) - SIGNAL_BATCH_SIZE} remaining)"',
            ),
            # Fix the debug text f-string
            (
                r'debug_text = f"No data file specified in plot configuration\\n\\nSaved file:"\s*"\'\{file_name\}\'"',
                r'debug_text = f"No data file specified in plot configuration\\n\\nSaved file: \'{file_name}\'"',
            ),
            # Fix the warning text f-string
            (
                r'warning_text = f"⚠️ Warning: Will overwrite existing files: \{\',"\s*"\'\.join\(existing_files\)\}"',
                r'warning_text = f"⚠️ Warning: Will overwrite existing files: {\', \'.join(existing_files)}"',
            ),
            # Fix typos
            (r"existting", r"existing"),
        ]

        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # Clean up any remaining orphaned try/except blocks
        lines = content.split("\n")
        cleaned_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Skip orphaned try blocks with just pass
            if (
                line.strip() == "try:"
                and i + 1 < len(lines)
                and "pass  # Placeholder" in lines[i + 1]
            ):
                # Skip this try block and its pass line
                i += 2
                # Also skip the following except if it exists
                if i < len(lines) and lines[i].strip().startswith(
                    "except Exception as e:"
                ):
                    # Skip the except and any following lines that are part of it
                    while i < len(lines) and (
                        lines[i].strip().startswith("except")
                        or lines[i].startswith("    ")
                        or lines[i].strip() == ""
                    ):
                        i += 1
                continue

            # Skip standalone except blocks that don't have a matching try
            if line.strip().startswith("except Exception as e:"):
                # Look backwards for a matching try
                found_try = False
                for j in range(
                    len(cleaned_lines) - 1, max(0, len(cleaned_lines) - 10), -1
                ):
                    if cleaned_lines[j].strip() == "try:":
                        found_try = True
                        break

                if not found_try:
                    # Skip this except block
                    i += 1
                    while i < len(lines) and (
                        lines[i].startswith("    ") or lines[i].strip() == ""
                    ):
                        i += 1
                    continue

            cleaned_lines.append(line)
            i += 1

        content = "\n".join(cleaned_lines)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed all syntax errors in {file_path}")
            return True
        else:
            print("No syntax errors found to fix")
            return False

    except Exception as e:
        print(f"❌ Error fixing syntax errors: {e}")
        return False


def main() -> None:
    """Main function to fix all syntax errors."""
    print("🔧 Fixing all syntax errors...")

    if fix_all_syntax_errors():
        print("✅ All syntax errors fixed successfully")
    else:
        print("❌ Failed to fix syntax errors")


if __name__ == "__main__":
    main()
