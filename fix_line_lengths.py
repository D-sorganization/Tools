#!/usr/bin/env python3
"""
Fix line length issues (E501 errors) in the codebase.
"""

import subprocess
from pathlib import Path


def fix_line_lengths_in_file(file_path: Path, max_length: int = 88) -> bool:
    """Fix line length issues in a specific file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        modified = False

        for i, line in enumerate(lines):
            if len(line) > max_length:
                # Fix long docstrings
                if '"""' in line and line.strip().startswith('"""'):
                    # Break long docstrings
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent

                    if line.count('"""') == 2:  # Single line docstring
                        content_start = line.find('"""') + 3
                        content_end = line.rfind('"""')
                        docstring_content = line[content_start:content_end]

                        if len(docstring_content) > 40:
                            lines[i] = f'{indent_str}"""'
                            lines.insert(i + 1, f'{indent_str}{docstring_content.strip()}')
                            lines.insert(i + 2, f'{indent_str}"""')
                            modified = True

                # Fix long comments
                elif line.strip().startswith('#'):
                    if len(line) > max_length:
                        indent = len(line) - len(line.lstrip())
                        indent_str = ' ' * indent
                        comment_text = line.strip()[1:].strip()

                        # Break into multiple lines
                        words = comment_text.split()
                        if len(words) > 3:
                            mid_point = len(words) // 2
                            first_part = ' '.join(words[:mid_point])
                            second_part = ' '.join(words[mid_point:])

                            lines[i] = f'{indent_str}# {first_part}'
                            lines.insert(i + 1, f'{indent_str}# {second_part}')
                            modified = True

                # Fix long string literals
                elif '"' in line and line.count('"') >= 2:
                    # Simple case: break long strings with continuation
                    if len(line) > max_length and '"""' not in line:
                        indent = len(line) - len(line.lstrip())
                        indent_str = ' ' * indent

                        # Find a good break point around the middle
                        break_point = max_length - 10
                        if break_point < len(line):
                            # Try to break at a space
                            while break_point > max_length // 2 and line[break_point] != ' ':
                                break_point -= 1

                            if break_point > max_length // 2:
                                first_part = line[:break_point].rstrip()
                                second_part = line[break_point:].lstrip()

                                lines[i] = f'{first_part} \\'
                                lines.insert(i + 1, f'{indent_str}    {second_part}')
                                modified = True

        if modified:
            file_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            return True
        return False

    except Exception as e:
        print(f"Error fixing line lengths in {file_path}: {e}")
        return False


def main():
    """Fix line length issues across the codebase."""
    print("📏 Fixing line length issues...")

    # Get files with E501 errors
    try:
        result = subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--select", "E501", "--output-format=concise"],
            capture_output=True,
            text=True,
            check=False
        )

        files_with_errors = set()
        for line in result.stdout.splitlines():
            if 'E501' in line and ':' in line:
                file_path = line.split(':')[0].strip()
                files_with_errors.add(Path(file_path))

        print(f"Found {len(files_with_errors)} files with line length issues")

        fixed_count = 0
        for file_path in files_with_errors:
            if file_path.exists() and file_path.suffix == '.py':
                if fix_line_lengths_in_file(file_path):
                    print(f"Fixed line lengths in {file_path}")
                    fixed_count += 1

        print(f"Fixed line lengths in {fixed_count} files")

        # Final check
        result = subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--statistics"],
            capture_output=True,
            text=True,
            check=False
        )
        print("\nFinal statistics:")
        print(result.stdout)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
