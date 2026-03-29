"""Migrate print() calls to logging in all non-test Python source files."""

import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_file(fpath: str) -> bool:
    """Replace print() with logger.debug() in a single file. Returns True if changed."""
    try:
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return False

    if "print(" not in content:
        return False

    new_content = content

    # Add logging import if missing
    if "import logging" not in new_content:
        if new_content.startswith('"""'):
            end_doc = new_content.find('"""', 3) + 3
            new_content = (
                new_content[:end_doc] + "\n\nimport logging\n" + new_content[end_doc:]
            )
        else:
            new_content = "import logging\n" + new_content

    # Add logger instance if missing
    if "getLogger" not in new_content:
        new_content = re.sub(
            r"(import logging\n)",
            r"\1\nlogger = logging.getLogger(__name__)\n",
            new_content,
            count=1,
        )

    # Replace bare print( with logger.debug(
    new_content = re.sub(r"\bprint\(", "logger.debug(", new_content)

    if new_content != content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


def main() -> None:
    src_dir = r"c:\Users\diete\Repositories\Tools\src"
    fixed = 0
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".mypy_cache", ".git"]]
        for fname in files:
            if not fname.endswith(".py") or "test" in fname.lower():
                continue
            fpath = os.path.join(root, fname)
            if migrate_file(fpath):
                fixed += 1
                logger.info("Fixed: %s", fpath)
    logger.info("Total files migrated: %d", fixed)


if __name__ == "__main__":
    main()
