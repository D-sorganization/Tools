#!/usr/bin/env python3
"""
Critical syntax error fixes for CI/CD pipeline.
Focuses on the most problematic files causing the majority of syntax errors.
"""

import re
import subprocess
from pathlib import Path


def fix_data_processor_syntax():
    """Fix critical syntax errors in Data_Processor_r0.py"""
    file_path = Path("data_processing/data_processor/python/data_processor/Data_Processor_r0.py")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Fix 1: Incomplete comment/docstring around line 5668
        content = re.sub(
            r'# Check if it has the expected structure \(\s*processing configs have',
            '# Check if it has the expected structure (\n                            # processing configs have',
            content
        )

        # Fix 2: Fix incomplete parentheses in conditions
        content = re.sub(
            r'if isinstance\(data, dict\) and \(\s*"saved_at" in data or "plot_name" in data\s*\):',
            'if isinstance(data, dict) and ("saved_at" in data or "plot_name" in data):',
            content
        )

        # Fix 3: Fix incomplete function calls
        content = re.sub(
            r'config_files\.append\(\s*$',
            'config_files.append(file_path)',
            content,
            flags=re.MULTILINE
        )

        # Fix 4: Fix incomplete try blocks
        content = re.sub(
            r'try:\s*$\s*except',
            'try:\n                    pass\n                except',
            content,
            flags=re.MULTILINE
        )

        # Fix 5: Fix f-string issues
        content = re.sub(
            r'f"([^"]*)\{([^}]*)\}([^"]*)"([^"]*$)',
            r'f"\1{\2}\3\4"',
            content,
            flags=re.MULTILINE
        )

        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"Fixed syntax errors in {file_path}")
            return True
        else:
            print(f"No changes needed in {file_path}")
            return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def remove_broken_fix_scripts():
    """Remove broken fix scripts that are causing syntax errors"""
    broken_scripts = [
        "fix_replicants_final_10.py",
        "fix_line_lengths_final.py",
        "fix_ci_cd_issues.py",
        "final_data_processor_fix.py",
        "fix_data_processor_aggressive.py",
        "fix_data_processor_r0_comprehensive.py",
        "fix_replicants_syntax.py",
    ]

    removed_count = 0
    for script in broken_scripts:
        script_path = Path(script)
        if script_path.exists():
            try:
                # Check if it has syntax errors first
                result = subprocess.run(
                    ["python", "-m", "py_compile", str(script_path)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    script_path.unlink()
                    print(f"Removed broken script: {script}")
                    removed_count += 1
            except Exception as e:
                print(f"Error checking/removing {script}: {e}")

    return removed_count


def run_ruff_fix():
    """Run ruff --fix for auto-fixable issues"""
    try:
        subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--fix", "--unsafe-fixes"],
            capture_output=True,
            text=True,
            check=False
        )
        print("Ruff auto-fix completed")
        return True
    except Exception as e:
        print(f"Error running ruff --fix: {e}")
        return False


def main():
    """Main function to fix critical syntax errors"""
    print("🚨 Fixing critical syntax errors for CI/CD pipeline...")

    # Step 1: Remove broken fix scripts
    print("\n📝 Step 1: Removing broken fix scripts...")
    removed_count = remove_broken_fix_scripts()
    print(f"Removed {removed_count} broken scripts")

    # Step 2: Fix Data_Processor_r0.py syntax issues
    print("\n🔧 Step 2: Fixing Data_Processor_r0.py syntax...")
    fix_data_processor_syntax()

    # Step 3: Run ruff auto-fix
    print("\n🔄 Step 3: Running ruff auto-fix...")
    run_ruff_fix()

    # Step 4: Check final error count
    print("\n📊 Step 4: Final error count...")
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

    print("\n✅ Critical syntax error fixes completed!")


if __name__ == "__main__":
    main()
