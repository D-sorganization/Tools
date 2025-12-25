#!/usr/bin/env python3
"""
Fix undefined variable 'i' in for loops across the codebase.
"""

import logging
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_undefined_i_in_file(file_path: str) -> bool:
    """Fix undefined variable 'i' in a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()


        # Fix pattern: for _ in range(...): ... lines[i - 1]
        # Replace _ with i in for loops where i is used in the body
        lines = content.split("\n")
        modified = False

        for line_num, line in enumerate(lines):
            # Look for for loops using _
            if "for i in range(" in line and ":" in line:
                # Check the next few lines for usage of 'i'
                for check_line in range(line_num + 1, min(line_num + 10, len(lines))):
                    if check_line < len(lines) and "lines[i" in lines[check_line]:
                        # Replace _ with i in the for loop
                        lines[line_num] = line.replace(
                            "for _ in range(", "for i in range("
                        )
                        modified = True
                        logger.info(
                            f"Fixed undefined 'i' in {file_path} at line {line_num + 1}"
                        )
                        break

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return True

        return False
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False


def fix_undefined_value_in_file(file_path: str) -> bool:
    """Fix undefined variable 'value' in for loops."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()


        # Fix pattern: for key, _ in items(): ... str(value)
        lines = content.split("\n")
        modified = False

        for line_num, line in enumerate(lines):
            # Look for for loops using _ as second variable
            if "for key, value in" in line and ".items()" in line:
                # Check the next few lines for usage of 'value'
                for check_line in range(line_num + 1, min(line_num + 5, len(lines))):
                    if (
                        check_line < len(lines)
                        and "value" in lines[check_line]
                        and "str(value)" in lines[check_line]
                    ):
                        # Replace _ with value in the for loop
                        lines[line_num] = line.replace(
                            "for key, _ in", "for key, value in"
                        )
                        modified = True
                        logger.info(
                            f"Fixed undefined 'value' in {file_path} at line {line_num + 1}"
                        )
                        break

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return True

        return False
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False


def get_python_files() -> list:
    """Get all Python files in the repository."""
    python_files = []
    for root, dirs, files in os.walk("."):
        # Skip certain directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in ["__pycache__", "node_modules"]
        ]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Skip the excluded problematic file
                if "Data_Processor_r0.py" not in file_path:
                    python_files.append(file_path)

    return python_files


def main():
    """Main function to fix undefined variables."""
    logger.info("Starting undefined variable fixes...")

    python_files = get_python_files()
    logger.info(f"Found {len(python_files)} Python files to process")

    total_fixes = 0

    for file_path in python_files:
        fixes_applied = 0

        # Fix undefined variable 'i'
        if fix_undefined_i_in_file(file_path):
            fixes_applied += 1

        # Fix undefined variable 'value'
        if fix_undefined_value_in_file(file_path):
            fixes_applied += 1

        if fixes_applied > 0:
            total_fixes += fixes_applied
            logger.info(f"Applied {fixes_applied} fixes to {file_path}")

    logger.info(f"Completed! Applied {total_fixes} total fixes")


if __name__ == "__main__":
    main()
