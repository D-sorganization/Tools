import os
import re
from pathlib import Path

def refactor_none_checks(file_path: Path):
    content = file_path.read_text(encoding="utf-8")
    
    # 1. if not (x is not None): -> if x is None:
    # Handle both single variable and expressions inside brackets
    # Note: Using non-greedy match for the variable part
    pattern1 = r"if not \(([\w\._\[\]]+) is not None\):"
    replacement1 = r"if \1 is None:"
    
    # 2. if not (x is not None and y is not None): -> if x is None or y is None:
    pattern2 = r"if not \(([\w\._\[\]]+) is not None and ([\w\._\[\]]+) is not None\):"
    replacement2 = r"if \1 is None or \2 is None:"

    # 3. while not (x is not None): -> while x is None:
    pattern3 = r"while not \(([\w\._\[\]]+) is not None\):"
    replacement3 = r"while \1 is None:"

    new_content = content
    new_content = re.sub(pattern1, replacement1, new_content)
    new_content = re.sub(pattern2, replacement2, new_content)
    new_content = re.sub(pattern3, replacement3, new_content)

    if new_content != content:
        file_path.write_text(new_content, encoding="utf-8")
        return True
    return False

def main():
    root = Path("src")
    count = 0
    for path in root.rglob("*.py"):
        if refactor_none_checks(path):
            print(f"Refactored {path}")
            count += 1
    print(f"Total files refactored: {count}")

if __name__ == "__main__":
    main()
