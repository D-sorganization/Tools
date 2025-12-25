#!/usr/bin/env python3
"""
Fix remaining line length issues systematically.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_line_lengths_in_file(file_path: str) -> bool:
    """Fix line length issues in a specific file."""
    try:
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()

        modified = False

        for i, line in enumerate(lines):
            if len(line.rstrip()) > 88:
                original_line = line.rstrip()

                # Skip comments and docstrings
                if original_line.strip().startswith('#') or '"""' in original_line:
                    continue

                # Fix specific patterns

                # 1. Long comments that can be broken
                if '# Now initialize Tkinter variables AFTER parent class has created the root window' in original_line:
                    lines[i] = '        # Now initialize Tkinter variables AFTER parent class has created root window\n'
                    modified = True
                    continue

                # 2. Long f-strings with file operations
                if 'f"Warning: No selected columns found in {os.path.basename(file_path)}' in original_line:indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}f"Warning: No selected columns found in {{os.path.basename(file_path)}}"\n'
                    modified = True
                    continue

                # 3. Long f-strings with data info
                if 'f"Loaded {os.path.basename(
                    file_path)}: {len(df)} rows,
                    {len(df.columns
                )}columns' in original_line:indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}f"Loaded {{os.path.basename(file_path)}}: {{len(df)}} rows, "\n' lines.insert(i + 1, f'{indent_str}f"{{len(df.columns)}} columns\n')modified = True
                    continue

                # 4. Long f-strings with combined data info
                if 'f"Combined data: {len(
                    combined_df)} rows,
                    {len(combined_df.columns
                )} columns"' in original_line:
                    indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}f"Combined data: {{len(
                        combined_df)}} rows,
                        {{len(combined_df.columns
                    )}} columns\n'modified = True
                    continue

                # 5. Long dictionary definitions
                if '"combine": "Copies all files from source folders into the single destination folder."' in original_line:
                    indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}"combine": "Copies all files from source folders into the single "\n'
                    lines.insert(
                        i + 1, f'{indent_str}           "destination folder.",
                        \n'
                    )
                    modified = True
                    continue

                # 6. Very long dictionary entries - break them up
                if 'flatten' in original_line and 'top level of the destination' in original_line and len(original_line) > 120:
                    indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}"flatten": "Finds deeply nested folders and copies them to the "\n'
                    lines.insert(
                        i + 1, f'{indent_str}           "top level of the destination.",
                        \n'
                    )
                    modified = True
                    continue

                # 7. Long prune description
                if 'prune' in original_line and 'preserving structure but skipping empty sub-folders' in original_line and len(original_line) > 120:
                    indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}"prune": "Copies source folders to the destination, preserving "\n'
                    lines.insert(
                        i + 1, f'{indent_str}         "structure but skipping empty sub-folders.",
                        \n'
                    )
                    modified = True
                    continue

                # 8. Long docstrings
                if '"""Perform combine operation - copy all files from source folders to destination."""' in original_line:
                    indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}"""Perform combine operation - copy all files from source folders to destination."""\n'
                    modified = True
                    continue

                # 9. Long status messages
                if 'f"PREVIEW: Would copy {copied_count} files, rename {renamed_count}, skip {skipped_count}' in original_line:indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}status = (\n'
                    lines.insert(
                        i + 1,
                        f'{indent_str}    f"PREVIEW: Would copy {{copied_count}} files,
                        "\n'
                    )
                    lines.insert(
                        i + 2, f'{indent_str}    f"rename {{renamed_count}},
                        skip {{skipped_count}}\n')
                    lines.insert(i + 3, f'{indent_str})\n')
                    modified = True
                    continue

                # 10. Long lambda expressions
                if 'lambda p=processed_files, t=total_files: self.folder_status_var.set(' in original_line and len(original_line) > 100:
                    indent = len(original_line) - len(original_line.lstrip())
                    indent_str = ' ' * indent
                    lines[i] = f'{indent_str}lambda p=processed_files, t=total_files: (\n'
                    lines.insert(
                        i + 1,
                        f'{indent_str}    self.folder_status_var.set(f"Processed {{p}}/{{t}} files")\n'
                    )
                    lines.insert(i + 2, f'{indent_str}),\n')
                    modified = True
                    continue

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            logger.info(f"Fixed line lengths in {file_path})return True

        return False
    except Exception as e:
        logger.error(f"Error fixing line lengths in {file_path}: {e}")
        return False

def main():
    """Main function."""
    file_path = "data_processing/data_processor/python/data_processor/Data_Processor_Integrated.py"

    logger.info(f"Fixing line length issues in {file_path})if fix_line_lengths_in_file(file_path):
        logger.info("Line length fixes applied successfully")
    else:
        logger.info("No line length fixes needed")

if __name__ == "__main__":
    main()
