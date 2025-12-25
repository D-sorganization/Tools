#!/usr/bin/env python3
"""
Fix syntax errors in replicants folder_packer_pro.py file.
"""

import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_syntax_errors():
    """Fix all syntax errors in folder_packer_pro.py."""

    file_path = Path("replicants/python/folder_packer_pro/folder_packer_pro.py")
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Fix broken string literals and indentation issues
    fixes = [
        # Fix broken newline characters in strings
        (r'line \+ "\\n"', r'line + "\n"'),
        # Fix method indentation issues
        (r"\n        def (_\w+)\(self,", r"\n\n    def \1(self,"),
        # Fix broken string literals in insert statements
        (
            r'insert\("1\.0", "Package appears to be encrypted or corrupted\.\\n"\)',
            r'insert("1.0", "Package appears to be encrypted or corrupted.\n")',
        ),
        # Fix broken f-string literals
        (
            r'f"Created: \{manifest_data\.get\(\'created_at\', \'Unknown\'\)\}\\n"',
            r'f"Created: {manifest_data.get(\'created_at\', \'Unknown\')}\n"',
        ),
        (
            r'f"Files: \{len\(manifest_data\.get\(\'files\', \[\]\)\)\}\\n"',
            r'f"Files: {len(manifest_data.get(\'files\', []))}\n"',
        ),
        (
            r'f"Metadata: \{manifest_data\.get\(\'metadata\', \{\}\)\}\\n\\n"',
            r'f"Metadata: {manifest_data.get(\'metadata\', {})}\n\n"',
        ),
        # Fix broken method definitions
        (
            r'(\s+)def (_\w+)\(self,([^)]*)\) -> ([^:]+):\s*"""([^"]+)"""\s*([^}]+)}',
            r'\1def \2(self,\3) -> \4:\n\1    """\5"""\n\1    \6',
        ),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Write the fixed content
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Applied syntax fixes")


def validate_python_syntax():
    """Validate that the Python file has correct syntax."""

    file_path = Path("replicants/python/folder_packer_pro/folder_packer_pro.py")

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Try to compile the code
        compile(content, str(file_path), "exec")
        logger.info("✅ Python syntax is valid")
        return True

    except SyntaxError as e:
        logger.error(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        logger.error(f"Text: {e.text}")
        return False
    except Exception as e:
        logger.error(f"❌ Error validating syntax: {e}")
        return False


def restore_from_backup_if_needed():
    """Restore from backup if syntax is broken."""

    backup_path = Path(
        "replicants/python/folder_packer_pro/folder_packer_pro.py.backup"
    )
    file_path = Path("replicants/python/folder_packer_pro/folder_packer_pro.py")

    if not validate_python_syntax():
        if backup_path.exists():
            logger.info("Restoring from backup due to syntax errors...")
            with open(backup_path, encoding="utf-8") as f:
                backup_content = f.read()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(backup_content)

            logger.info("Restored from backup")
        else:
            logger.error("No backup available!")


def main():
    """Main function to fix syntax errors."""
    logger.info("Starting syntax error fixes...")

    # First, validate current syntax
    if validate_python_syntax():
        logger.info("File already has valid syntax")
        return

    # Try to fix syntax errors
    fix_syntax_errors()

    # Validate the fixes
    if not validate_python_syntax():
        restore_from_backup_if_needed()

    logger.info("Syntax fix process completed")


if __name__ == "__main__":
    main()
