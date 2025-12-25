#!/usr/bin/env python3
"""
Fix remaining quality issues: line lengths and loop variable binding.
"""

import os
import re
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_line_lengths(file_path: str) -> bool:
    """Fix lines that are too long by breaking them appropriately."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        for i, line in enumerate(lines):
            if len(line.rstrip()) > 88:
                original_line = line.rstrip()

                # Skip if it's a comment or docstring that's hard to break
                if original_line.strip().startswith('#') or '"""' in original_line:
                    continue

                # Fix long f-strings
                if 'f"' in original_line and len(original_line) > 88:
                    # Break f-strings at logical points
                    if ' files into ' in original_line:
                        lines[i] = original_line.replace(
                            ' files into ',
                            ' files into "\n                        "'
                        ) +
                            '\n'                        modified = True
                        continue
                    elif ' for ' in original_line and 'basename' in original_line:
                        lines[i] = original_line.replace(
                            ' for ',
                            ' for "\n                        "'
                        ) +
                            '\n'                        modified = True
                        continue

                # Fix long function calls with multiple parameters
                if '(' in original_line and ')' in original_line and ',' in original_line:
                    indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent

                    # Look for function calls that can be broken
                    if 'self._log_conversion_message(' in original_line:
                        # Break after the opening parenthesis
                        parts = original_line.split('self._log_conversion_message(', 1)
                        if len(parts) == 2:
                            lines[i] = f"{parts[0]}self._log_conversion_message(\n{indent_str}    {parts[1]}"
                            modified = True
                            continue

                    # Break long string concatenations
                    if ' + ' in original_line and '"' in original_line:
                        # Find a good break point
                        if len(original_line) > 100:
                            mid_point = len(original_line) // 2
                            # Find the nearest ' + ' to the midpoint
                            plus_positions = [m.start(
                                ) for m in re.finditer(r' \+ ',
                                original_line
                            )]
                            if plus_positions:
                                best_pos = min(
                                    plus_positions,
                                    key=lambda x: abs(x - mid_point)
                                )
                                before = original_line[:best_pos].rstrip()
                                after = original_line[best_pos + 3:].lstrip()
                                lines[i] = f"{before} +\n{indent_str}    {after}"
                                modified = True
                                continue

                # Fix long dictionary/list definitions
                if (original_line.strip().startswith('"') and '":' in original_line and
                    len(original_line) > 100):
                    # Break long dictionary entries
                    if '": "' in original_line:
                        key_end = original_line.find('": "') + 3
                        key_part = original_line[:key_end]
                        value_part = original_line[key_end:]
                        if len(key_part) < 50:  # Only break if key is reasonable length
                            lines[i] = f"{key_part}\n{indent_str}    {value_part}"
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

def fix_lambda_binding_issues(file_path: str) -> bool:
    """Fix B023 issues where lambda functions don't bind loop variables."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern: lambda p=processed_files, t=total_files:
        # self.folder_status_var.set(...)
        # Fix by ensuring variables are properly captured
        patterns = [
            # Fix lambda with processed_files and total_files
            (
                r'lambda p=processed_files,
                t=total_files:\s*self\.folder_status_var\.set\(\s*f"([^"]*
            )"',
             r'lambda p=processed_files, t=total_files: self.folder_status_var.set(f"\1"'),

            # Fix lambda with just status setting
            (r'lambda:\s*self\.folder_status_var\.set\(([^)]+)\)',
             r'lambda: self.folder_status_var.set(\1)'),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # More specific fix for the B023 issues
        # Replace problematic lambda patterns with proper variable binding
        content = re.sub(
            r'self\.after\(
                \s*0,
                \s*lambda p=processed_files,
                t=total_files:\s*self\.folder_status_var\.set\(\s*f"Processed \{p\}/\{t\} files"\s*\
            )',
            'self.after(
                0,
                lambda p=processed_files,
                t=total_files: self.folder_status_var.set(f"Processed {p}/{t} files"))
            )',
            content
        )

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Fixed lambda binding issues in {file_path}")
            return True

        return False
    except Exception as e:
        logger.error(f"Error fixing lambda binding in {file_path}: {e}")
        return False

def get_python_files() -> List[str]:
    """Get all Python files excluding problematic ones."""
    python_files = []
    exclude_patterns = [
        'fix_*.py',
        'comprehensive*.py',
        'Data_Processor_r0.py',
        '*archive*'
    ]

    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                # Check if file should be excluded
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern.replace('*', '') in file_path:
                        should_exclude = True
                        break

                if not should_exclude:
                    python_files.append(file_path)

    return python_files

def main():
    """Main function to fix remaining quality issues."""
    logger.info("Starting remaining quality issue fixes...")

    python_files = get_python_files()
    logger.info(f"Found {len(python_files)} Python files to process")

    total_fixes = 0

    for file_path in python_files:
        fixes_applied = 0

        # Fix line length issues
        if fix_line_lengths(file_path):
            fixes_applied += 1

        # Fix lambda binding issues
        if fix_lambda_binding_issues(file_path):
            fixes_applied += 1

        if fixes_applied > 0:
            total_fixes += fixes_applied
            logger.info(f"Applied {fixes_applied} fixes to {file_path}")

    logger.info(f"Completed! Applied {total_fixes} total fixes")

    # Run Black formatting to clean up any formatting issues
    logger.info("Running Black formatting...")
    os.system("black --line-length=88 --target-version=py38 --exclude='fix_.*\\.py|comprehensive.*\\.py|Data_Processor_r0\\.py|.*archive.*' .")

    logger.info("All remaining quality fixes completed!")

if __name__ == "__main__":
    main()
