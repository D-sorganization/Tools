#!/usr/bin/env python3
"""
Comprehensive fix for critical Ruff, MyPy, and Black issues.
Addresses undefined variables, syntax errors, and line length issues.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime
)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_undefined_variable_i(file_path: str) -> bool:
    """Fix undefined variable 'i' in for loops."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern to find for loops using _ but referencing i
        patterns = [
            (r'for _ in range\(([^)]+)\):\s*\n(\s+)([^\n]*lines\[i[^\]]*\])', 
             r'for i in range(\1):\n\2\3'),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Fixed undefined variable 'i' in {file_path}")
            return True

        return False
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return False

def fix_f_string_syntax(file_path: str) -> bool:
    """Fix f-string syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        for i, line in enumerate(lines):
            # Fix unterminated f-strings
            if 'f"' in line and line.count('"') % 2 != 0:
                # Find the f-string and ensure it's properly closed
                if line.strip().endswith('f"') or 'f"' in line and not line.strip().endswith('"'):"
                    lines[i] = line.rstrip() + '"\n'
                    modified = True
                    logger.info(f"Fixed f-string syntax in {file_path} line {i+1}")

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True

        return False
    except Exception as e:
        logger.error(f"Error fixing f-string syntax in {file_path}: {e}")
        return False

def fix_line_lengths(file_path: str) -> bool:
    """Fix lines that are too long."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        for i, line in enumerate(lines):
            if len(line.rstrip()) > 88:
                # Simple line breaking for common patterns
                stripped = line.rstrip()
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent

                # Break long function calls
                if '(' in stripped and ')' in stripped and ',' in stripped:
                    # Find function call pattern
                    match = re.match(r'(\s*\w+.*?\()(.*?)(\).*)', stripped)
                    if match:
                        prefix, args, suffix = match.groups()
                        if ',' in args:
                            arg_list = [arg.strip() for arg in args.split(',')]
                            if len(arg_list) > 1:
                                new_line = prefix + '\n'
                                for j, arg in enumerate(arg_list):
                                    if j == len(arg_list) - 1:
new_line +
                                            = f"{indent_str}    {arg}\n{indent_str}{suffix}\n"
                                    else:
                                        new_line += f"{indent_str}    {arg},\n"
                                lines[i] = new_line
                                modified = True
                                continue

                # Break long string concatenations
                if '+' in stripped and '"' in stripped:
                    parts = stripped.split('+')
                    if len(parts) > 1:
                        new_line = f"{parts[0].strip()} +\n"
                        for part in parts[1:]:
                            new_line += f"{indent_str}    {part.strip()}"
                            if part != parts[-1]:
                                new_line += " +\n"
                            else:
                                new_line += "\n"
                        lines[i] = new_line
                        modified = True
                        continue

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            logger.info(f"Fixed line lengths in {file_path}")
            return True

        return False
    except Exception as e:
        logger.error(f"Error fixing line lengths in {file_path}: {e}")
        return False

def fix_whitespace_issues(file_path: str) -> bool:
    """Fix whitespace issues like blank lines with whitespace."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        for i, line in enumerate(lines):
            # Remove whitespace from blank lines
            if line.strip() == '' and line != '\n':
                lines[i] = '\n'
                modified = True

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            logger.info(f"Fixed whitespace issues in {file_path}")
            return True

        return False
    except Exception as e:
        logger.error(f"Error fixing whitespace in {file_path}: {e}")
        return False

def get_python_files() -> List[str]:
    """Get all Python files in the repository."""
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # Skip the excluded problematic file
                if 'Data_Processor_r0.py' not in file_path:
                    python_files.append(file_path)

    return python_files

def main():
    """Main function to fix all critical issues."""
    logger.info("Starting comprehensive Ruff/MyPy/Black fixes...")

    python_files = get_python_files()
    logger.info(f"Found {len(python_files)} Python files to process")

    total_fixes = 0

    # Fix critical issues in each file
    for file_path in python_files:
        logger.info(f"Processing {file_path}")

        fixes_applied = 0

        # Fix undefined variable 'i'
        if fix_undefined_variable_i(file_path):
            fixes_applied += 1

        # Fix f-string syntax errors
        if fix_f_string_syntax(file_path):
            fixes_applied += 1

        # Fix line length issues
        if fix_line_lengths(file_path):
            fixes_applied += 1

        # Fix whitespace issues
        if fix_whitespace_issues(file_path):
            fixes_applied += 1

        if fixes_applied > 0:
            total_fixes += fixes_applied
            logger.info(f"Applied {fixes_applied} fixes to {file_path}")

    logger.info(f"Completed! Applied {total_fixes} total fixes across all files")

    # Run Black formatting
    logger.info("Running Black formatting...")
    os.system("black . --line-length=88 --target-version=py38")

    # Run Ruff fixes
    logger.info("Running Ruff auto-fixes...")
    os.system("ruff check --fix --unsafe-fixes .")

    logger.info("All fixes completed!")

if __name__ == "__main__":
    main()