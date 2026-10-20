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


_ADR_README = "docs/adr/README.md"
_ADR_TEMPLATE = "docs/adr/ADR_TEMPLATE.md"
_ADR_README_ERROR = "ADR docs changed without updating docs/adr/README.md"


def _adr_body_changed(changed: list[str]) -> bool:
    return any(
        path.startswith("docs/adr/") and path not in {_ADR_README, _ADR_TEMPLATE}
        for path in changed
    )


def adr_readme_error(changed: list[str], *, index_is_fresh: bool) -> str | None:
    """Return an error when an ADR edit leaves the records index stale.

    A diff that already contains ``docs/adr/README.md`` satisfies the gate.
    A format-only ADR edit also satisfies it when ``check_adr_references``
    reports the records table is already current.
    """
    if not _adr_body_changed(changed):
        return None
    if _ADR_README in changed or index_is_fresh:
        return None
    return _ADR_README_ERROR


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

    index_is_fresh = False
    if _adr_body_changed(changed) and _ADR_README not in changed_set:
        from scripts import check_adr_references

        index_is_fresh = check_adr_references.main(["--root", str(ROOT)]) == 0
    adr_error = adr_readme_error(changed, index_is_fresh=index_is_fresh)
    if adr_error:
        return _fail(adr_error)

    sys.stdout.write("docs governance checks passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
