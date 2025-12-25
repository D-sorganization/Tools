#!/usr/bin/env python3
"""
Comprehensive CI/CD Issues Fix Script

This script addresses all Black, Ruff, and MyPy issues found in the tools repository.
It follows the coding standards defined in AGENTS.md.
"""

import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def fix_ruff_issues() -> bool:
    """Fix all auto-fixable Ruff issues."""
    print("🔧 Fixing Ruff issues...")

    # Run ruff with --fix to auto-fix issues
    exit_code, stdout, stderr = run_command(["ruff", "check", "--fix", "."])

    if exit_code == 0:
        print("✅ All Ruff issues fixed successfully")
        return True
    else:
        print(f"⚠️ Some Ruff issues remain (exit code: {exit_code})")
        if stderr:
            print(f"Stderr: {stderr}")
        return False


def fix_black_formatting() -> bool:
    """Fix all Black formatting issues."""
    print("🎨 Fixing Black formatting...")

    # Run black to format all files
    exit_code, stdout, stderr = run_command(["black", "."])

    if exit_code == 0:
        print("✅ All files formatted with Black successfully")
        return True
    else:
        print(f"❌ Black formatting failed (exit code: {exit_code})")
        if stderr:
            print(f"Stderr: {stderr}")
        return False


def fix_specific_long_lines() -> bool:
    """Fix specific long lines that Black can't handle automatically."""
    print("📏 Fixing specific long lines...")

    files_with_long_lines = [
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py",
        "media_processing/video_processor/python/tests/test_logger_utils_mock.py",
        "media_processing/video_processor/scripts/quality_check.py",
        "media_processing/video_processor/tools/code_quality_check.py",
        "media_processing/video_processor/tools/matlab_utilities/scripts/matlab_quality_check.py",
        "web_applications/unit_converter/tools/code_quality_check.py",
    ]

    changes_made = False

    for file_path in files_with_long_lines:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Fix common long line patterns
            content = fix_long_comments(content)
            content = fix_long_strings(content)
            content = fix_long_function_calls(content)

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed long lines in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def fix_long_comments(content: str) -> str:
    """Fix long comments by breaking them at logical points."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if line.strip().startswith("#") and len(line.rstrip()) > 88:
            indent = len(line) - len(line.lstrip())
            comment_text = line.strip()[1:].strip()

            # Break at logical points
            for break_point in [
                " - ",
                ": ",
                ", ",
                " and ",
                " or ",
                " but ",
                " with ",
                " for ",
            ]:
                if break_point in comment_text and len(comment_text) > 60:
                    parts = comment_text.split(break_point, 1)
                    if 20 < len(parts[0]) < 75 and len(parts[1]) < 75:
                        new_lines.append(
                            " " * indent + "# " + parts[0] + break_point.rstrip()
                        )
                        new_lines.append(" " * indent + "# " + parts[1])
                        break
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def fix_long_strings(content: str) -> str:
    """Fix long string literals by breaking them."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if '"' in line and len(line.rstrip()) > 88:
            # Try to break long strings at logical points
            if any(keyword in line for keyword in ['f"', "text=", "messagebox"]):
                indent = len(line) - len(line.lstrip())

                # Find string content and break it
                quote_matches = list(re.finditer(r'"([^"]*)"', line))
                if quote_matches:
                    for match in quote_matches:
                        content_text = match.group(1)
                        if len(content_text) > 60:
                            # Break at logical points
                            for break_point in [
                                ": ",
                                " - ",
                                ", ",
                                " and ",
                                " or ",
                                " with ",
                            ]:
                                if break_point in content_text:
                                    parts = content_text.split(break_point, 1)
                                    if 20 < len(parts[0]) < 70 and len(parts[1]) < 70:
                                        new_line1 = line.replace(
                                            f'"{content_text}"',
                                            f'"{parts[0]}{break_point.rstrip()}"',
                                        )
                                        new_line2 = " " * (indent + 4) + f'"{parts[1]}"'
                                        if line.endswith(","):
                                            new_line2 += ","
                                        elif line.endswith(")"):
                                            new_line2 += ")"
                                        new_lines.append(new_line1)
                                        new_lines.append(new_line2)
                                        break
                            else:
                                new_lines.append(line)
                            break
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def fix_long_function_calls(content: str) -> str:
    """Fix long function calls and method chains."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if len(line.rstrip()) > 88 and ("(" in line or "." in line):
            # Try to break long function calls
            if line.count("(") == line.count(")") and "," in line:
                indent = len(line) - len(line.lstrip())
                # Break at commas in function calls
                if line.strip().endswith(")") or line.strip().endswith("),"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        new_line = parts[0] + ","
                        new_lines.append(new_line)
                        for part in parts[1:-1]:
                            new_lines.append(" " * (indent + 4) + part.strip() + ",")
                        # Last part
                        last_part = parts[-1].strip()
                        new_lines.append(" " * (indent + 4) + last_part)
                        continue

            new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def remove_unused_variables() -> bool:
    """Fix unused variable issues by renaming them with underscore prefix."""
    print("🧹 Fixing unused variables...")

    # This is handled by ruff --fix, but we can add specific fixes if needed
    files_to_check = ["fix_data_processor_lines.py", "fix_remaining_issues.py"]

    changes_made = False

    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Fix unused loop variables
            content = re.sub(
                r"for i, (.+) in enumerate\(", r"for _i, \1 in enumerate(", content
            )

            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Fixed unused variables in {file_path}")
                changes_made = True

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return changes_made


def verify_fixes() -> bool:
    """Verify that all fixes were successful."""
    print("🔍 Verifying fixes...")

    # Check Ruff
    exit_code, stdout, stderr = run_command(["ruff", "check", "."])
    ruff_ok = exit_code == 0

    # Check Black
    exit_code, stdout, stderr = run_command(["black", "--check", "."])
    black_ok = exit_code == 0

    # Check MyPy
    exit_code, stdout, stderr = run_command(
        ["mypy", "python/src/", "--ignore-missing-imports"]
    )
    mypy_ok = exit_code == 0

    print(f"Ruff: {'✅' if ruff_ok else '❌'}")
    print(f"Black: {'✅' if black_ok else '❌'}")
    print(f"MyPy: {'✅' if mypy_ok else '❌'}")

    return ruff_ok and black_ok and mypy_ok


def main() -> None:
    """Main function to fix all CI/CD issues."""
    print("🚀 Starting CI/CD Issues Fix")
    print("=" * 50)

    success_count = 0
    total_steps = 5

    # Step 1: Fix Ruff issues
    if fix_ruff_issues():
        success_count += 1

    # Step 2: Fix Black formatting
    if fix_black_formatting():
        success_count += 1

    # Step 3: Fix specific long lines
    if fix_specific_long_lines():
        success_count += 1

    # Step 4: Fix unused variables
    if remove_unused_variables():
        success_count += 1

    # Step 5: Verify all fixes
    if verify_fixes():
        success_count += 1
        print("\n🎉 All CI/CD issues have been resolved!")
    else:
        print("\n⚠️ Some issues may still remain. Check the output above.")

    print(f"\nCompleted {success_count}/{total_steps} steps successfully")

    if success_count == total_steps:
        print("✅ Repository is now ready for CI/CD!")
        sys.exit(0)
    else:
        print("❌ Some issues need manual attention")
        sys.exit(1)


if __name__ == "__main__":
    main()
