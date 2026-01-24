#!/usr/bin/env python3
"""
Enhanced comprehensive batch fixer for DRY violations.

This script systematically fixes common DRY violations across the codebase
with better pattern matching and import management.
"""

import re
import sys
from pathlib import Path

# Track total fixes
TOTAL_FIXES = 0


def ensure_import(content: str, import_line: str, fallback: str = "") -> str:
    """Ensure an import exists, adding it after existing imports."""
    if import_line in content:
        return content

    # Find the last import statement
    import_pattern = r"^(import |from .* import )"
    lines = content.split("\n")
    last_import_idx = -1

    for i, line in enumerate(lines):
        if re.match(import_pattern, line.strip()):
            last_import_idx = i

    if last_import_idx >= 0:
        # Add after last import
        insert_idx = last_import_idx + 1
        # Skip blank lines after imports
        while insert_idx < len(lines) and not lines[insert_idx].strip():
            insert_idx += 1

        if fallback:
            new_lines = (
                lines[:insert_idx]
                + [""]
                + [import_line]
                + [fallback]
                + lines[insert_idx:]
            )
        else:
            new_lines = lines[:insert_idx] + [""] + [import_line] + lines[insert_idx:]
        return "\n".join(new_lines)

    # No imports found, add at top
    if fallback:
        return f"{import_line}\n{fallback}\n{content}"
    return f"{import_line}\n{content}"


def fix_json_load_patterns(content: str) -> tuple[str, int]:
    """Replace json.load patterns with safe_read_json."""
    fixes = 0

    # Pattern 1: with open(...) as f: data = json.load(f)
    pattern1 = re.compile(
        r'with\s+open\s*\(\s*([^,)]+)(?:,\s*["\']r["\']?)?(?:,\s*encoding\s*=\s*["\']utf-8["\'])?\s*\)\s+as\s+(\w+):\s*\n\s*(\w+)\s*=\s*json\.load\s*\(\s*\2\s*\)',
        re.MULTILINE,
    )

    def replace1(match):
        nonlocal fixes
        fixes += 1
        file_path = match.group(1).strip()
        var_name = match.group(3)
        return f"{var_name} = safe_read_json({file_path}, default=None)"

    content = pattern1.sub(replace1, content)

    # Pattern 2: data = safe_read_json(..., default=None)
    pattern2 = re.compile(
        r'(\w+)\s*=\s*json\.load\s*\(\s*open\s*\(\s*([^,)]+)(?:,\s*["\']r["\']?)?(?:,\s*encoding\s*=\s*["\']utf-8["\'])?\s*\)\s*\)',
        re.MULTILINE,
    )

    def replace2(match):
        nonlocal fixes
        fixes += 1
        var_name = match.group(1)
        file_path = match.group(2).strip()
        return f"{var_name} = safe_read_json({file_path}, default=None)"

    content = pattern2.sub(replace2, content)

    # Add import if fixes made
    if fixes > 0:
        content = ensure_import(
            content,
            "try:\n    from utils.file_utils import safe_read_json\nexcept ImportError:\n    import json\n    def safe_read_json(path, default=None):\n        try:\n            with open(path, encoding='utf-8') as f:\n                return json.load(f)\n        except Exception:\n            return default",
        )

    return content, fixes


def fix_json_dump_patterns(content: str) -> tuple[str, int]:
    """Replace json.dump patterns with safe_write_json."""
    fixes = 0

    # Pattern: json.dump(data, open(...), ...)
    pattern = re.compile(
        r'json\.dump\s*\(\s*([^,)]+),\s*open\s*\(\s*([^,)]+)(?:,\s*["\']w["\']?)?(?:,\s*encoding\s*=\s*["\']utf-8["\'])?\s*\)(?:,\s*indent\s*=\s*(\d+))?\s*\)',
        re.MULTILINE,
    )

    def replace(match):
        nonlocal fixes
        fixes += 1
        data = match.group(1).strip()
        file_path = match.group(2).strip()
        indent = match.group(3) if match.group(3) else "2"
        return f"safe_write_json({file_path}, {data}, indent={indent})"

    content = pattern.sub(replace, content)

    # Pattern: with open(...) as f: json.dump(data, f, ...)
    pattern2 = re.compile(
        r'with\s+open\s*\(\s*([^,)]+)(?:,\s*["\']w["\']?)?(?:,\s*encoding\s*=\s*["\']utf-8["\'])?\s*\)\s+as\s+(\w+):\s*\n\s*json\.dump\s*\(\s*([^,)]+),\s*\2(?:,\s*indent\s*=\s*(\d+))?\s*\)',
        re.MULTILINE,
    )

    def replace2(match):
        nonlocal fixes
        fixes += 1
        file_path = match.group(1).strip()
        data = match.group(3).strip()
        indent = match.group(4) if match.group(4) else "2"
        return f"safe_write_json({file_path}, {data}, indent={indent})"

    content = pattern2.sub(replace2, content)

    # Add import if fixes made
    if fixes > 0:
        content = ensure_import(
            content,
            "try:\n    from utils.file_utils import safe_write_json\nexcept ImportError:\n    import json\n    def safe_write_json(path, data, indent=2, create_parents=True):\n        Path(path).parent.mkdir(parents=True, exist_ok=True)\n        with open(path, 'w', encoding='utf-8') as f:\n            json.dump(data, f, indent=indent)",
        )

    return content, fixes


def fix_path_join_patterns(content: str) -> tuple[str, int]:
    """Replace os.path.join with Path operations."""
    fixes = 0

    # Pattern: Path(a) / b / c -> Path(a) / b / c
    pattern = re.compile(r"os\.path\.join\s*\(\s*([^)]+)\s*\)")

    def replace(match):
        nonlocal fixes
        fixes += 1
        args = match.group(1)
        # Split by comma, handling strings carefully
        parts = []
        current = ""
        in_string = False
        string_char = None

        for char in args:
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
                current += char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
                current += char
            elif char == "," and not in_string:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            parts.append(current.strip())

        if not parts:
            return match.group(0)

        # Build Path expression
        result = f"Path({parts[0]})"
        for part in parts[1:]:
            result += f" / {part}"

        return result

    content = pattern.sub(replace, content)

    # Pattern: Path(path).exists() -> Path(path).exists()
    pattern2 = re.compile(r"os\.path\.exists\s*\(\s*([^)]+)\s*\)")

    def replace2(match):
        nonlocal fixes
        fixes += 1
        return f"Path({match.group(1)}).exists()"

    content = pattern2.sub(replace2, content)

    # Pattern: Path(path).parent -> Path(path).parent
    pattern3 = re.compile(r"os\.path\.dirname\s*\(\s*([^)]+)\s*\)")

    def replace3(match):
        nonlocal fixes
        fixes += 1
        return f"Path({match.group(1)}).parent"

    content = pattern3.sub(replace3, content)

    # Pattern: Path(path).name -> Path(path).name
    pattern4 = re.compile(r"os\.path\.basename\s*\(\s*([^)]+)\s*\)")

    def replace4(match):
        nonlocal fixes
        fixes += 1
        return f"Path({match.group(1)}).name"

    content = pattern4.sub(replace4, content)

    # Add Path import if fixes made
    if fixes > 0 and "from pathlib import Path" not in content:
        content = ensure_import(content, "from pathlib import Path")

    return content, fixes


def fix_sys_path_patterns(content: str) -> tuple[str, int]:
    """Replace sys.path manipulations with ensure_utils_in_path."""
    fixes = 0

    # Pattern: ensure_utils_in_path() or ensure_utils_in_path()
    pattern = re.compile(
        r"sys\.path\.(?:insert\s*\(\s*0\s*,\s*|append\s*\(\s*)str\s*\(([^)]+)\)\s*\)",
        re.MULTILINE,
    )

    def replace(match):
        nonlocal fixes
        fixes += 1
        return "ensure_utils_in_path()"

    content = pattern.sub(replace, content)

    # Add import if fixes made
    if fixes > 0:
        content = ensure_import(
            content,
            "try:\n    from utils.path_helpers import ensure_utils_in_path\nexcept ImportError:\n    def ensure_utils_in_path():\n        pass",
        )

    return content, fixes


def fix_text_file_patterns(content: str) -> tuple[str, int]:
    """Replace text file read/write patterns with safe_read_text/safe_write_text."""
    fixes = 0

    # Pattern: with open(...) as f: content = f.read()
    pattern1 = re.compile(
        r'with\s+open\s*\(\s*([^,)]+)(?:,\s*["\']r["\']?)?(?:,\s*encoding\s*=\s*["\']utf-8["\'])?\s*\)\s+as\s+(\w+):\s*\n\s*(\w+)\s*=\s*\2\.read\s*\(\s*\)',
        re.MULTILINE,
    )

    def replace1(match):
        nonlocal fixes
        fixes += 1
        file_path = match.group(1).strip()
        var_name = match.group(3)
        return f"{var_name} = safe_read_text({file_path}, default='')"

    content = pattern1.sub(replace1, content)

    # Pattern: with open(...) as f: f.write(...)
    pattern2 = re.compile(
        r'with\s+open\s*\(\s*([^,)]+)(?:,\s*["\']w["\']?)?(?:,\s*encoding\s*=\s*["\']utf-8["\'])?\s*\)\s+as\s+(\w+):\s*\n\s*\2\.write\s*\(\s*([^)]+)\s*\)',
        re.MULTILINE,
    )

    def replace2(match):
        nonlocal fixes
        fixes += 1
        file_path = match.group(1).strip()
        content_var = match.group(3).strip()
        return f"safe_write_text({file_path}, {content_var})"

    content = pattern2.sub(replace2, content)

    # Add import if fixes made
    if fixes > 0:
        content = ensure_import(
            content,
            "try:\n    from utils.file_utils import safe_read_text, safe_write_text\nexcept ImportError:\n    from pathlib import Path\n    def safe_read_text(path, encoding='utf-8', default=''):\n        try:\n            return Path(path).read_text(encoding=encoding)\n        except Exception:\n            return default\n    def safe_write_text(path, content, encoding='utf-8', create_parents=True):\n        p = Path(path)\n        if create_parents:\n            p.parent.mkdir(parents=True, exist_ok=True)\n        p.write_text(content, encoding=encoding)",
        )

    return content, fixes


def fix_csv_patterns(content: str) -> tuple[str, int]:
    """Replace pd.read_csv and df.to_csv with safe_read_csv/safe_write_csv."""
    fixes = 0

    # Pattern: df = safe_read_csv(...)
    pattern1 = re.compile(
        r"(\w+)\s*=\s*pd\.read_csv\s*\(\s*([^,)]+)(?:,\s*([^)]+))?\s*\)", re.MULTILINE
    )

    def replace1(match):
        nonlocal fixes
        fixes += 1
        var_name = match.group(1)
        file_path = match.group(2).strip()
        kwargs = match.group(3) if match.group(3) else ""
        if kwargs:
            return f"{var_name} = safe_read_csv({file_path}, {kwargs})"
        return f"{var_name} = safe_read_csv({file_path})"

    content = pattern1.sub(replace1, content)

    # Pattern: safe_write_csv(df, ...)
    pattern2 = re.compile(
        r"(\w+)\.to_csv\s*\(\s*([^,)]+)(?:,\s*([^)]+))?\s*\)", re.MULTILINE
    )

    def replace2(match):
        nonlocal fixes
        fixes += 1
        df_var = match.group(1)
        file_path = match.group(2).strip()
        kwargs = match.group(3) if match.group(3) else ""
        if kwargs:
            return f"safe_write_csv({df_var}, {file_path}, {kwargs})"
        return f"safe_write_csv({df_var}, {file_path})"

    content = pattern2.sub(replace2, content)

    # Add import if fixes made
    if fixes > 0:
        content = ensure_import(
            content,
            "try:\n    from utils.csv_utils import safe_read_csv, safe_write_csv\nexcept ImportError:\n    import pandas as pd\n    from pathlib import Path\n    def safe_read_csv(path, default=None, **kwargs):\n        try:\n            return pd.read_csv(path, **kwargs)\n        except Exception:\n            return default if default is not None else pd.DataFrame()\n    def safe_write_csv(df, path, create_parents=True, **kwargs):\n        Path(path).parent.mkdir(parents=True, exist_ok=True)\n        df.to_csv(path, **kwargs)",
        )

    return content, fixes


def process_file(file_path: Path) -> int:
    """Process a single file and fix all DRY violations."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original = content
        total_fixes = 0

        # Apply all fixes
        content, fixes = fix_json_load_patterns(content)
        total_fixes += fixes

        content, fixes = fix_json_dump_patterns(content)
        total_fixes += fixes

        content, fixes = fix_path_join_patterns(content)
        total_fixes += fixes

        content, fixes = fix_sys_path_patterns(content)
        total_fixes += fixes

        content, fixes = fix_text_file_patterns(content)
        total_fixes += fixes

        content, fixes = fix_csv_patterns(content)
        total_fixes += fixes

        # Only write if changes made
        if content != original:
            file_path.write_text(content, encoding="utf-8")
            return total_fixes

        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)


try:
    from utils.csv_utils import safe_read_csv, safe_write_csv
except ImportError:
    from pathlib import Path

    import pandas as pd


try:
    from utils.csv_utils import safe_read_csv, safe_write_csv
except ImportError:
    from pathlib import Path

    import pandas as pd



    def safe_read_csv(path, default=None, **kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            return default if default is not None else pd.DataFrame()

    def safe_write_csv(df, path, create_parents=True, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, **kwargs)


def find_python_files(root: Path) -> list[Path]:
    """Find all Python files to process."""
    files = []
    for path in root.rglob("*.py"):
        # Skip certain directories
        if any(
            skip in str(path)
            for skip in [
                "__pycache__",
                ".git",
                "node_modules",
                ".venv",
                "venv",
                "utils",
            ]
        ):
            continue
        files.append(path)
    return files


def main():
    """Main entry point."""
    repo_root = Path("/home/dieterolson/Linux_Tools/Tools")

    # Find Python files
    print("Finding Python files...")
    files = find_python_files(repo_root)
    print(f"Found {len(files)} Python files")

    total_fixes = 0
    fixed_files = 0

    for file_path in sorted(files):
        fixes = process_file(file_path)
        if fixes > 0:
            print(f"Fixed {fixes} violations in {file_path.relative_to(repo_root)}")
            total_fixes += fixes
            fixed_files += 1

    print(f"\nTotal: {fixed_files} files, {total_fixes} violations fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
