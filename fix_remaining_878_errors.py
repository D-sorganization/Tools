#!/usr/bin/env python3
"""
Systematic fix for the remaining 878 CI/CD errors.
Prioritizes syntax errors first, then other issues.
"""

import re
import subprocess
from pathlib import Path


def get_error_breakdown() -> dict[str, list[str]]:
    """Get a breakdown of current errors by type."""
    try:
        result = subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--output-format=concise"],
            capture_output=True,
            text=True,
            check=False
        )

        errors = {"syntax": [], "line_length": [], "other": []}

        for line in result.stdout.splitlines():
            if "SyntaxError" in line:
                errors["syntax"].append(line)
            elif "E501" in line:
                errors["line_length"].append(line)
            else:
                errors["other"].append(line)

        return errors
    except Exception as e:
        print(f"Error getting error breakdown: {e}")
        return {"syntax": [], "line_length": [], "other": []}


def fix_syntax_errors_in_file(file_path: Path) -> bool:
    """Fix common syntax errors in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Fix 1: Missing closing brackets in list comprehensions
        content = re.sub(
            r'(\s+)(\w+)\s*=\s*\[\s*([^]]+)\s*(\w+\s*=)',
            r'\1\2 = [\3]\n\1\4',
            content
        )

        # Fix 2: Broken f-strings
        content = re.sub(
            r'f"([^"]*)\{\s*([^}]+)\s*\}\s*([^"]*)"([^,\n]*)\n\s*([^"]*)"',
            r'f"\1{\2}\3\4\5"',
            content
        )

        # Fix 3: Missing quotes in string literals
        content = re.sub(
            r'(\s+)"([^"]*)\n\s*([^"]*)"',
            r'\1"\2 \3"',
            content
        )

        # Fix 4: Broken multi-line strings
        content = re.sub(
            r'"([^"]*)\(\s*\n\s*([^)]*)\s*\n\s*\)"',
            r'"\1(\2)"',
            content
        )

        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error fixing syntax in {file_path}: {e}")
        return False


def fix_line_length_in_file(file_path: Path, max_length: int = 88) -> bool:
    """Fix line length issues in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        original_lines = lines.copy()

        for i, line in enumerate(lines):
            if len(line) > max_length:
                # Fix long comments
                if line.strip().startswith('#'):
                    indent_match = re.match(r'^(\s*)', line)
                    indent = indent_match.group(1) if indent_match else ''

                    # Break long comments
                    words = line.split()
                    if len(words) > 3:
                        current_line = f"{indent}#"
                        new_lines = []

                        for word in words[1:]:  # Skip the '#'
                            if len(current_line + ' ' + word) <= max_length:
                                current_line += ' ' + word
                            else:
                                new_lines.append(current_line)
                                current_line = f"{indent}# {word}"

                        if current_line.strip() != f"{indent}#".strip():
                            new_lines.append(current_line)

                        if len(new_lines) > 1:
                            lines[i:i+1] = new_lines
                            continue

                # Fix long string literals
                if '"' in line and line.count('"') >= 2:
                    # Find string content
                    match = re.search(r'(\s*.*?")(.*?)(".*)', line)
                    if match and len(match.group(2)) > 40:
                        prefix, string_content, suffix = match.groups()

                        # Break string into parts
                        words = string_content.split()
                        if len(words) > 3:
                            mid_point = len(words) // 2
                            first_part = ' '.join(words[:mid_point])
                            second_part = ' '.join(words[mid_point:])

                            indent_match = re.match(r'^(\s*)', line)
                            indent = indent_match.group(1) if indent_match else ''

                            new_lines = [
                                f'{prefix}{first_part}" \\',
                                f'{indent}    "{second_part}{suffix}'
                            ]
                            lines[i:i+1] = new_lines
                            continue

        if lines != original_lines:
            file_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error fixing line lengths in {file_path}: {e}")
        return False


def remove_broken_fix_scripts():
    """Remove broken fix scripts that are causing syntax errors."""
    broken_scripts = [
        "fix_additional_ruff_issues.py",
        "fix_all_syntax_errors.py",
        "fix_aggressive_remaining_issues.py",
        "fix_critical_ruff_issues.py",
        "fix_data_processor_lines.py",
        "fix_emoji_encoding.py",
        "fix_lambda_binding.py",
        "fix_line_lengths.py",
        "fix_remaining_issues.py",
        "fix_remaining_issues_comprehensive.py",
        "fix_remaining_lines.py",
        "fix_remaining_long_lines.py",
        "fix_remaining_quality_issues.py",
        "fix_remaining_ruff_issues.py",
    ]

    removed_count = 0
    for script in broken_scripts:
        script_path = Path(script)
        if script_path.exists():
            try:
                script_path.unlink()
                print(f"Removed broken script: {script}")
                removed_count += 1
            except Exception as e:
                print(f"Error removing {script}: {e}")

    return removed_count


def main():
    """Main function to systematically fix remaining errors."""
    print("🔧 Systematic fix for remaining 878 CI/CD errors...")

    # Step 1: Remove broken fix scripts
    print("\n📝 Step 1: Removing broken fix scripts...")
    removed_scripts = remove_broken_fix_scripts()
    print(f"Removed {removed_scripts} broken scripts")

    # Step 2: Get current error breakdown
    print("\n📊 Step 2: Analyzing current errors...")
    errors = get_error_breakdown()
    print(f"Syntax errors: {len(errors['syntax'])}")
    print(f"Line length errors: {len(errors['line_length'])}")
    print(f"Other errors: {len(errors['other'])}")

    # Step 3: Fix syntax errors first (highest priority)
    print("\n🚨 Step 3: Fixing syntax errors...")
    syntax_fixes = 0

    # Get files with syntax errors
    syntax_files = set()
    for error in errors['syntax']:
        if ':' in error:
            file_path = error.split(':')[0]
            syntax_files.add(Path(file_path))

    for file_path in syntax_files:
        if file_path.exists() and file_path.suffix == '.py':
            if fix_syntax_errors_in_file(file_path):
                syntax_fixes += 1
                print(f"Fixed syntax errors in {file_path}")

    print(f"Applied syntax fixes to {syntax_fixes} files")

    # Step 4: Fix line length issues
    print("\n📏 Step 4: Fixing line length issues...")
    length_fixes = 0

    # Get files with line length errors
    length_files = set()
    for error in errors['line_length']:
        if ':' in error:
            file_path = error.split(':')[0]
            length_files.add(Path(file_path))

    for file_path in length_files:
        if file_path.exists() and file_path.suffix == '.py':
            if fix_line_length_in_file(file_path):
                length_fixes += 1
                print(f"Fixed line lengths in {file_path}")

    print(f"Applied line length fixes to {length_fixes} files")

    # Step 5: Run ruff --fix for auto-fixable issues
    print("\n🔄 Step 5: Running ruff --fix...")
    try:
        result = subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--fix"],
            capture_output=True,
            text=True,
            check=False
        )
        print("Ruff auto-fix completed")
    except Exception as e:
        print(f"Error running ruff --fix: {e}")

    # Step 6: Final error count
    print("\n📈 Step 6: Final error count...")
    try:
        result = subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--statistics"],
            capture_output=True,
            text=True,
            check=False
        )
        print("Final statistics:")
        print(result.stdout)
    except Exception as e:
        print(f"Error getting final statistics: {e}")

    print("\n✅ Systematic debugging completed!")
    print(f"Total fixes applied: {syntax_fixes + length_fixes}")


if __name__ == "__main__":
    main()
