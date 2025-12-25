#!/usr/bin/env python3
"""
Fix emoji encoding issues in Data_Processor_r0.py that prevent Black formatting.
"""

import re
from pathlib import Path


def fix_emoji_issues() -> bool:
    """Fix emoji characters that cause Black formatting issues."""
    file_path = Path(
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
    )

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Remove or replace problematic emoji characters
        emoji_replacements = {
            "⚙️": "CONFIG",
            "📊": "DATA",
            "🔧": "TOOLS",
            "🎯": "TARGET",
            "📁": "FILES",
            "📈": "CHARTS",
            "📋": "LIST",
            "❓": "HELP",
            "🌊": "FILTER",
            "📏": "MEASURE",
            "🛡️": "SECURE",
            "🔢": "NUMBERS",
            "📐": "CALC",
            "📉": "TREND",
            "🔄": "PROCESS",
            "⚡": "FAST",
            "⚠️": "WARNING",
        }

        # Replace emoji characters with text equivalents
        for emoji, replacement in emoji_replacements.items():
            content = content.replace(emoji, replacement)

        # Remove any remaining emoji or special Unicode characters that might cause
        # issues
        # Keep only ASCII printable characters, newlines, and basic Unicode
        content = re.sub(r"[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF]", "", content)

        # Clean up any malformed lines that might have been created
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip lines that are just emoji or special characters
            if line.strip() and not re.match(r"^[^\w\s]*$", line.strip()):
                cleaned_lines.append(line)
            elif not line.strip():  # Keep empty lines
                cleaned_lines.append(line)

        content = "\n".join(cleaned_lines)

        # Remove any orphaned documentation sections that might be causing issues
        # These appear to be misplaced help text
        doc_sections_to_remove = [
            r"CONFIG BULK PROCESSING MODE:.*?(?=\n\n|\nclass|\ndef|\n    def)",
            r"LIST ESSENTIAL BUTTONS:.*?(?=\n\n|\nclass|\ndef|\n    def)",
            r"FILES SIGNAL LIST MANAGEMENT:.*?(?=\n\n|\nclass|\ndef|\n    def)",
            r"PROCESS TYPICAL WORKFLOW:.*?(?=\n\n|\nclass|\ndef|\n    def)",
            r"TARGET Smart Auto-Zoom System.*?(?=\n\n|\nclass|\ndef|\n    def)",
            r"TOOLS Configuration Management.*?(?=\n\n|\nclass|\ndef|\n    def)",
            r"FAST Performance Improvements.*?(?=\n\n|\nclass|\ndef|\n    def)",
        ]

        for pattern in doc_sections_to_remove:
            content = re.sub(pattern, "", content, flags=re.MULTILINE | re.DOTALL)

        # Clean up multiple empty lines
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed emoji encoding issues in {file_path}")
            return True
        else:
            print("No emoji issues found to fix")
            return False

    except Exception as e:
        print(f"❌ Error fixing emoji issues: {e}")
        return False


def main() -> None:
    """Main function."""
    print("🔧 Fixing emoji encoding issues...")

    if fix_emoji_issues():
        print("✅ Emoji fix completed successfully")
    else:
        print("❌ Emoji fix failed")


if __name__ == "__main__":
    main()
