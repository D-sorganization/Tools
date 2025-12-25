#!/usr/bin/env python3
"""
Fix undefined variable errors in the codebase.
"""

import re
from pathlib import Path


def fix_undefined_i_variables():
    """Fix undefined 'i' variables in loops."""
    file_path = Path("data_processing/data_processor/python/data_processor/Data_Processor_r0.py")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Fix pattern: for _ in range(...): ... i ...
        # Replace _ with i when i is used in the loop body
        lines = content.splitlines()
        modified = False

        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for "for _ in range" patterns
            if re.match(r'\s*for\s+_\s+in\s+range\s*\(', line):
                # Check the next few lines for usage of 'i'
                loop_start = i
                indent_level = len(line) - len(line.lstrip())

                # Find the end of this loop block
                j = i + 1
                uses_i = False
                while j < len(lines):
                    next_line = lines[j]
                    if next_line.strip() == "":
                        j += 1
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= indent_level and next_line.strip():
                        break

                    # Check if this line uses 'i'
                    if re.search(r'\bi\b', next_line):
                        uses_i = True

                    j += 1

                # If 'i' is used in the loop, replace _ with i
                if uses_i:
                    lines[loop_start] = re.sub(r'for\s+_\s+in\s+range', 'for i in range', lines[loop_start])
                    modified = True
                    print(f"Fixed undefined 'i' at line {loop_start + 1}")

            i += 1

        if modified:
            file_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            print(f"Fixed undefined variables in {file_path}")
            return True
        else:
            print(f"No undefined variable fixes needed in {file_path}")
            return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def fix_bare_except_clauses():
    """Fix bare except clauses."""
    file_path = Path("data_processing/data_processor/python/data_processor/Data_Processor_r0.py")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Replace bare except: with except Exception:
        content = re.sub(r'except\s*:', 'except Exception:', content)

        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"Fixed bare except clauses in {file_path}")
            return True
        else:
            print(f"No bare except clauses found in {file_path}")
            return False

    except Exception as e:
        print(f"Error fixing bare except clauses in {file_path}: {e}")
        return False


def main():
    """Main function to fix undefined variables and other issues."""
    print("🔧 Fixing undefined variables and other issues...")

    # Fix undefined 'i' variables
    print("\n📝 Step 1: Fixing undefined 'i' variables...")
    fixed_vars = fix_undefined_i_variables()

    # Fix bare except clauses
    print("\n🚫 Step 2: Fixing bare except clauses...")
    fixed_except = fix_bare_except_clauses()

    print("\n✅ Fixes completed!")
    print(f"Undefined variables fixed: {fixed_vars}")
    print(f"Bare except clauses fixed: {fixed_except}")


if __name__ == "__main__":
    main()
