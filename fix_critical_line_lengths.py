#!/usr/bin/env python3
"""
Fix critical line length issues that are blocking CI/CD.
"""

import re
from pathlib import Path


def fix_long_comments(content: str) -> str:
    """Fix long comments by breaking them at logical points."""
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        if line.strip().startswith('#') and len(line.rstrip()) > 88:
            indent = len(line) - len(line.lstrip())
            comment_text = line.strip()[1:].strip()

            # Break at logical points
            break_points = [' - ', ': ', ', ', ' and ', ' or ', ' but ', ' with ', ' for ', ' to ', ' from ']
            broken = False

            for break_point in break_points:
                if break_point in comment_text and len(comment_text) > 60:
                    parts = comment_text.split(break_point, 1)
                    if 20 < len(parts[0]) < 75 and len(parts[1]) < 75:
                        new_lines.append(' ' * indent + '# ' + parts[0] + break_point.rstrip())
                        new_lines.append(' ' * indent + '# ' + parts[1])
                        broken = True
                        break

            if not broken:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)


def fix_long_strings(content: str) -> str:
    """Fix long string literals by breaking them."""
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        if len(line.rstrip()) > 88 and '"' in line:
            # Handle specific patterns
            if 'f"' in line and ('DEBUG:' in line or 'Warning:' in line or 'Error:' in line):
                # Break f-strings at logical points
                match = re.search(r'f"([^"]*)"', line)
                if match:
                    content_text = match.group(1)
                    if len(content_text) > 50:
                        # Try to break at common separators
                        for sep in [': ', ' - ', ', ', ' from ', ' to ', ' with ']:
                            if sep in content_text:
                                parts = content_text.split(sep, 1)
                                if 15 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                    indent = len(line) - len(line.lstrip())
                                    new_line1 = line.replace(f'f"{content_text}"', f'f"{parts[0]}{sep.rstrip()}"')
                                    new_line2 = ' ' * (indent + 4) + f'f"{parts[1]}"'
                                    if line.endswith(','):
                                        new_line2 += ','
                                    elif line.endswith(')'):
                                        new_line2 = new_line2.replace('f"', 'f"').rstrip() + ')'
                                    new_lines.extend([new_line1, new_line2])
                                    break
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)


def fix_file_issues(filepath: Path) -> bool:
    """Fix issues in a single file."""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_long_comments(content)
        content = fix_long_strings(content)

        # Ensure file ends with newline
        if not content.endswith('\n'):
            content += '\n'

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            print(f"No changes needed: {filepath}")
            return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Fix critical files with line length issues."""
    files_to_fix = [
        "media_processing/video_processor/scripts/quality_check.py",
        "media_processing/video_processor/tools/matlab_utilities/scripts/matlab_quality_check.py",
        "media_processing/video_processor/python/tests/test_logger_utils_mock.py",
        "media_processing/video_processor/tools/code_quality_check.py",
        "web_applications/unit_converter/tools/code_quality_check.py",
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        path = Path(file_path)
        if path.exists():
            if fix_file_issues(path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
