#!/usr/bin/env python3
"""
Fix line length issues (E501) across the repository.
"""

import re
from pathlib import Path


def fix_line_length_in_file(file_path: Path, max_length: int = 88) -> bool:
    """Fix line length issues in a single file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        original_lines = lines.copy()

        for i, line in enumerate(lines):
            if len(line) > max_length:
                # Try to fix common long line patterns

                # 1. Long string literals - break them
                if '"""' in line or "'''" in line:
                    continue  # Skip docstrings for now

                # 2. Long function calls with multiple parameters
                if "(" in line and ")" in line and "," in line:
                    # Find function call pattern
                    match = re.match(r"^(\s*)(.*?)(\(.*\))(.*)$", line)
                    if match:
                        indent, prefix, params, suffix = match.groups()
                        if len(params) > 40:  # Only break if params are long
                            # Break after opening parenthesis
                            new_indent = indent + "    "
                            param_parts = params[1:-1].split(",")
                            if len(param_parts) > 1:
                                new_lines = [f"{indent}{prefix}("]
                                for j, param in enumerate(param_parts):
                                    param = param.strip()
                                    if j == len(param_parts) - 1:
                                        new_lines.append(f"{new_indent}{param}")
                                    else:
                                        new_lines.append(f"{new_indent}{param},")
                                new_lines.append(f"{indent}){suffix}")

                                # Replace the long line with multiple lines
                                lines[i : i + 1] = new_lines
                                continue

                # 3. Long comments - break them
                if line.strip().startswith("#"):
                    words = line.split()
                    if len(words) > 3:
                        indent_match = re.match(r"^(\s*)", line)
                        indent = indent_match.group(1) if indent_match else ""

                        # Break comment into multiple lines
                        current_line = f"{indent}#"
                        new_lines = []

                        for word in words[1:]:  # Skip the '#'
                            if len(current_line + " " + word) <= max_length:
                                current_line += " " + word
                            else:
                                new_lines.append(current_line)
                                current_line = f"{indent}# {word}"

                        if current_line.strip() != f"{indent}#".strip():
                            new_lines.append(current_line)

                        if len(new_lines) > 1:
                            lines[i : i + 1] = new_lines
                            continue

                # 4. Long import statements
                if line.strip().startswith("from ") and " import " in line:
                    if "," in line:
                        parts = line.split(" import ")
                        if len(parts) == 2:
                            from_part = parts[0]
                            import_part = parts[1]
                            imports = [imp.strip() for imp in import_part.split(",")]

                            if len(imports) > 1:
                                indent_match = re.match(r"^(\s*)", line)
                                indent = indent_match.group(1) if indent_match else ""

                                new_lines = [f"{from_part} import ("]
                                for j, imp in enumerate(imports):
                                    if j == len(imports) - 1:
                                        new_lines.append(f"{indent}    {imp}")
                                    else:
                                        new_lines.append(f"{indent}    {imp},")
                                new_lines.append(f"{indent})")

                                lines[i : i + 1] = new_lines
                                continue

        if lines != original_lines:
            file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True
        return False
    except Exception as e:
        print(f"Error fixing line lengths in {file_path}: {e}")
        return False


def main():
    """Main function to fix line length issues."""
    print("🔧 Fixing line length issues (E501)...")

    # Get all Python files
    python_files = list(Path(".").glob("**/*.py"))

    fixes_applied = 0

    for py_file in python_files:
        if py_file.name.startswith(".") or "fix_line_length_issues" in str(py_file):
            continue

        if fix_line_length_in_file(py_file):
            fixes_applied += 1
            print(f"Fixed line lengths in {py_file}")

    print(f"\n✅ Fixed line length issues in {fixes_applied} files")


if __name__ == "__main__":
    main()
