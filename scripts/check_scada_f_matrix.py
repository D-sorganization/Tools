#!/usr/bin/env python3
"""Keep the P1AM SCADA requirement matrix honest (Tools #4912).

``docs/scada/f_matrix.v1.json`` is the tracker of record for F01-F16 and the
historian children H1-H9, replacing the phantom-complete checklists on #4085,
#4086, #4087, #4088, #4089 and #4046. A tracker of record is only worth having
if it cannot rot, so this script asserts the properties that make it trustworthy:

* every implementing and test path it cites actually exists in the tree;
* a ``landed`` or ``partial`` row cites at least one implementing file *and* at
  least one test, and says what is on main;
* a ``missing`` row cites nothing at all -- it cannot smuggle in evidence;
* every row states its gaps, including ``partial`` rows, because the gap list is
  what replaces a ticked checkbox;
* the published totals agree with a recount of the rows;
* the rendered ``f_matrix.md`` is in step with the JSON.

Run ``python scripts/check_scada_f_matrix.py --check`` (the flag is accepted for
symmetry with the other repo checkers; the default action is the same).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs" / "scada" / "f_matrix.v1.json"
RENDERED_PATH = ROOT / "docs" / "scada" / "f_matrix.md"

SCHEMA_VERSION = "scada-f-matrix/v1"
STATUSES = ("landed", "partial", "missing")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PR_REF_PATTERN = re.compile(r"^#\d{3,5}$")

EXPECTED_SCADA_IDS = tuple(f"F{n:02d}" for n in range(1, 17))
EXPECTED_HISTORIAN_IDS = tuple(f"H{n}" for n in range(1, 10))


def load_matrix() -> dict[str, Any]:
    """Read the matrix, or fail with a message rather than a traceback."""
    if not MATRIX_PATH.is_file():
        raise SystemExit(f"missing matrix: {MATRIX_PATH.relative_to(ROOT)}")
    return cast(dict[str, Any], json.loads(MATRIX_PATH.read_text(encoding="utf-8")))


def _check_identity(matrix: dict[str, Any], fail: list[str]) -> None:
    if matrix.get("schema_version") != SCHEMA_VERSION:
        fail.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if matrix.get("repo") != "D-sorganization/Tools":
        fail.append("repo must be D-sorganization/Tools")
    if matrix.get("audit_issue") != "#4912":
        fail.append("audit_issue must be #4912")
    base = matrix.get("base") or {}
    if base.get("ref") != "origin/main":
        fail.append("base.ref must be origin/main")
    if not SHA_PATTERN.match(str(base.get("sha", ""))):
        fail.append("base.sha must be a full 40-character SHA")
    if set(matrix.get("status_classes", {})) != set(STATUSES):
        fail.append(f"status_classes must define exactly {STATUSES}")
    for key in ("method", "headline"):
        if not str(matrix.get(key, "")).strip():
            fail.append(f"{key} must be a non-empty statement")


def _check_coverage(rows: list[dict[str, Any]], fail: list[str]) -> None:
    ids = [str(row.get("id")) for row in rows]
    if len(set(ids)) != len(ids):
        fail.append("duplicate requirement ids")
    expected = EXPECTED_SCADA_IDS + EXPECTED_HISTORIAN_IDS
    absent = [rid for rid in expected if rid not in ids]
    if absent:
        fail.append(f"requirements with no row: {absent}")
    extra = [rid for rid in ids if rid not in expected]
    if extra:
        fail.append(f"unexpected requirement ids: {extra}")


def _check_row(row: dict[str, Any], fail: list[str]) -> None:
    rid = str(row.get("id"))
    status = row.get("status")
    if status not in STATUSES:
        fail.append(f"{rid}: status {status!r} is not one of {STATUSES}")
        return
    if not str(row.get("title", "")).strip():
        fail.append(f"{rid}: empty title")

    tracker = str(row.get("child") or row.get("epic") or "")
    if not PR_REF_PATTERN.match(tracker):
        fail.append(f"{rid}: epic/child must be an issue ref like #4085")

    files = [str(p) for p in row.get("files", [])]
    tests = [str(p) for p in row.get("tests", [])]
    for path in files + tests:
        if not (ROOT / path).is_file():
            fail.append(f"{rid}: cited path does not exist: {path}")

    for ref in row.get("evidence_prs", []):
        if not PR_REF_PATTERN.match(str(ref)):
            fail.append(f"{rid}: bad evidence PR ref {ref!r}")

    delivered = str(row.get("delivered", "")).strip()
    gaps = [str(g) for g in row.get("gaps", [])]

    if not gaps:
        fail.append(f"{rid}: every row must state its gaps")

    if status == "missing":
        # The whole point of the audit: a missing requirement may not quietly
        # accumulate "evidence" that would let a checkbox be re-ticked.
        if files:
            fail.append(f"{rid}: missing status cites implementing files {files}")
        if tests:
            fail.append(f"{rid}: missing status cites tests {tests}")
        if delivered:
            fail.append(f"{rid}: missing status claims delivered work")
        if row.get("evidence_prs"):
            fail.append(f"{rid}: missing status cites evidence PRs")
    else:
        if not files:
            fail.append(f"{rid}: {status} status cites no implementing file")
        if not tests:
            fail.append(f"{rid}: {status} status cites no test")
        if not delivered:
            fail.append(f"{rid}: {status} status does not say what is on main")


def _check_totals(
    matrix: dict[str, Any], rows: list[dict[str, Any]], fail: list[str]
) -> None:
    totals = matrix.get("totals") or {}
    for key, prefix in (("scada", "F"), ("historian", "H")):
        series = [row for row in rows if str(row.get("id")).startswith(prefix)]
        published = totals.get(key) or {}
        for status in STATUSES:
            counted = sum(1 for row in series if row.get("status") == status)
            if published.get(status) != counted:
                fail.append(
                    f"totals.{key}.{status}: published "
                    f"{published.get(status)!r}, recount {counted}"
                )
        if published.get("total") != len(series):
            fail.append(
                f"totals.{key}.total: published {published.get('total')!r}, "
                f"recount {len(series)}"
            )


def _check_carrier_prs(matrix: dict[str, Any], fail: list[str]) -> None:
    prs = matrix.get("closed_carrier_prs") or []
    if not prs:
        fail.append("closed_carrier_prs must not be empty")
    for pr in prs:
        number = pr.get("number")
        if not isinstance(number, int):
            fail.append(f"carrier PR with a non-integer number: {number!r}")
            continue
        if not SHA_PATTERN.match(str(pr.get("head_oid", ""))):
            fail.append(f"#{number}: head_oid must be a full 40-character SHA")
        if not isinstance(pr.get("reachable"), bool):
            fail.append(f"#{number}: reachable must be a boolean")
        if not str(pr.get("note", "")).strip():
            fail.append(f"#{number}: needs a note recording what it carried")
        count = pr.get("product_files_absent_from_main")
        if not isinstance(count, int) or count < 0:
            fail.append(f"#{number}: product_files_absent_from_main must be >= 0")


def _check_rendered(
    matrix: dict[str, Any], rows: list[dict[str, Any]], fail: list[str]
) -> None:
    if not RENDERED_PATH.is_file():
        fail.append(f"missing rendered matrix: {RENDERED_PATH.relative_to(ROOT)}")
        return
    rendered = RENDERED_PATH.read_text(encoding="utf-8")
    for row in rows:
        rid = str(row.get("id"))
        if f"### {rid} - " not in rendered and f"| {rid} |" not in rendered:
            fail.append(f"{rid}: absent from the rendered f_matrix.md")
    for key, label in (("scada", "SCADA F01-F16"), ("historian", "Historian H1-H9")):
        totals = (matrix.get("totals") or {}).get(key) or {}
        expected_row = (
            f"| {label} | {totals.get('landed')} | {totals.get('partial')} "
            f"| {totals.get('missing')} | {totals.get('total')} |"
        )
        if expected_row not in rendered:
            fail.append(
                f"rendered counts row for {label} is stale; expected: {expected_row}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the matrix (the default action; accepted for symmetry).",
    )
    parser.parse_args(argv)

    matrix = load_matrix()
    rows = cast(list[dict[str, Any]], matrix.get("requirements") or [])

    fail: list[str] = []
    _check_identity(matrix, fail)
    _check_coverage(rows, fail)
    for row in rows:
        _check_row(row, fail)
    _check_totals(matrix, rows, fail)
    _check_carrier_prs(matrix, fail)
    _check_rendered(matrix, rows, fail)

    if fail:
        sys.stderr.write("SCADA F-matrix check FAILED:\n")
        for problem in fail:
            sys.stderr.write(f"  - {problem}\n")
        return 1

    scada = (matrix.get("totals") or {}).get("scada") or {}
    hist = (matrix.get("totals") or {}).get("historian") or {}
    sys.stdout.write(
        "SCADA F-matrix check passed "
        f"({len(rows)} requirements; "
        f"F: {scada.get('landed')} landed / {scada.get('partial')} partial / "
        f"{scada.get('missing')} missing; "
        f"H: {hist.get('landed')} landed / {hist.get('partial')} partial / "
        f"{hist.get('missing')} missing)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
