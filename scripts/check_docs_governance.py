#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "assessments" / "README.md",
    ROOT / "docs" / "adr" / "README.md",
    ROOT / "docs" / "adr" / "ADR_TEMPLATE.md",
    ROOT / "docs" / "governance" / "DOCS_GOVERNANCE.md",
]


def _changed_files() -> list[str]:
    cp = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        return []
    return [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]


def _fail(msg: str) -> int:
    sys.stderr.write(msg + "\n")
    return 1


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        return _fail("Missing docs governance files:\n- " + "\n- ".join(missing))

    changed = _changed_files()
    changed_set = set(changed)

    if any(
        p.startswith("docs/assessments/") and p != "docs/assessments/README.md"
        for p in changed
    ):
        if "docs/assessments/README.md" not in changed_set:
            return _fail(
                "Assessment docs changed without updating docs/assessments/README.md"
            )

    if any(
        p.startswith("docs/adr/")
        and p not in {"docs/adr/README.md", "docs/adr/ADR_TEMPLATE.md"}
        for p in changed
    ):
        if "docs/adr/README.md" not in changed_set:
            return _fail("ADR docs changed without updating docs/adr/README.md")

    sys.stdout.write("docs governance checks passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
