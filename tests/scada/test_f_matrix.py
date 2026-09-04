"""Contract tests for the P1AM SCADA requirement matrix (Tools #4912).

``docs/scada/f_matrix.v1.json`` is the tracker of record for F01-F16 and H1-H9,
replacing the checklists on #4085/#4086/#4087/#4088/#4089/#4046 -- which showed
38 of 38 boxes ticked while every carrier PR sat closed and unmerged.

The failure mode being defended against is precise: a requirement gets marked
delivered without any code on ``main`` behind it. So these tests do two things.
They run the checker in-process (so the checker itself is covered rather than
merely present), and they inject deliberately corrupted matrices to prove each
guard actually rejects the thing it claims to reject -- a checker whose rules
never fire is indistinguishable from no checker at all.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import check_scada_f_matrix as checker

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "docs" / "scada" / "f_matrix.v1.json"
RENDERED_PATH = ROOT / "docs" / "scada" / "f_matrix.md"

EXPECTED_IDS = [f"F{n:02d}" for n in range(1, 17)] + [f"H{n}" for n in range(1, 10)]


def _matrix() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(MATRIX_PATH.read_text(encoding="utf-8")))


def _rows() -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _matrix()["requirements"])


# --------------------------------------------------------------------------- #
# The committed matrix must pass its own checker                              #
# --------------------------------------------------------------------------- #
def test_committed_matrix_passes_the_checker() -> None:
    assert checker.main([]) == 0


def test_checker_accepts_the_check_flag() -> None:
    """The documented invocation is ``--check``; keep it working."""
    assert checker.main(["--check"]) == 0


# --------------------------------------------------------------------------- #
# Shape of the matrix itself                                                  #
# --------------------------------------------------------------------------- #
def test_every_requirement_has_exactly_one_row() -> None:
    assert [str(row["id"]) for row in _rows()] == EXPECTED_IDS


def test_no_requirement_is_claimed_landed_without_being_reverified() -> None:
    """The audit's central finding, pinned as an assertion.

    If someone later marks a row ``landed``, this test forces them to update it
    here deliberately rather than letting the count drift back to the phantom
    38/38.
    """
    matrix = _matrix()
    assert matrix["totals"]["scada"]["landed"] == 0
    assert matrix["totals"]["historian"]["landed"] == 0


def test_every_cited_path_exists() -> None:
    for row in _rows():
        for path in list(row["files"]) + list(row["tests"]):
            assert (ROOT / str(path)).is_file(), f"{row['id']}: {path}"


def test_missing_rows_cite_no_evidence() -> None:
    for row in _rows():
        if row["status"] != "missing":
            continue
        rid = row["id"]
        assert row["files"] == [], rid
        assert row["tests"] == [], rid
        assert row["evidence_prs"] == [], rid
        assert not str(row["delivered"]).strip(), rid


def test_partial_rows_carry_both_evidence_and_a_gap() -> None:
    partials = [row for row in _rows() if row["status"] == "partial"]
    assert partials, "the matrix should contain partial rows"
    for row in partials:
        rid = row["id"]
        assert row["files"], rid
        assert row["tests"], rid
        assert str(row["delivered"]).strip(), rid
        assert row["gaps"], rid


def test_every_row_states_its_gaps() -> None:
    for row in _rows():
        assert row["gaps"], f"{row['id']} must say what is not delivered"


def test_carrier_prs_record_head_oids_not_branch_tips() -> None:
    """Recovery classification is only reproducible from the head OID.

    A branch tip can carry later stacked merges, which is exactly how an earlier
    audit misattributed 19 files to the wrong PR.
    """
    prs = _matrix()["closed_carrier_prs"]
    numbers = sorted(cast(int, pr["number"]) for pr in prs)
    assert numbers == [4065, 4091, 4093, 4094, 4095, 4449]
    for pr in prs:
        assert checker.SHA_PATTERN.match(str(pr["head_oid"])), pr["number"]


def test_rendered_markdown_is_in_step_with_the_json() -> None:
    rendered = RENDERED_PATH.read_text(encoding="utf-8")
    for row in _rows():
        assert f"### {row['id']} - " in rendered, row["id"]


# --------------------------------------------------------------------------- #
# The guards must actually fire                                               #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def corrupt(monkeypatch: pytest.MonkeyPatch):
    """Feed the checker a mutated copy of the real matrix."""

    def _apply(mutate) -> int:
        data = copy.deepcopy(_matrix())
        mutate(data)
        monkeypatch.setattr(checker, "load_matrix", lambda: data)
        return checker.main([])

    return _apply


def test_guard_rejects_a_missing_row_that_grew_evidence(corrupt) -> None:
    """The regression that matters most: a `missing` row quietly re-ticked."""

    def mutate(data: dict[str, Any]) -> None:
        row = next(r for r in data["requirements"] if r["status"] == "missing")
        row["files"] = ["scripts/check_scada_f_matrix.py"]
        row["delivered"] = "shipped, honestly"

    assert corrupt(mutate) == 1


def test_guard_rejects_a_partial_row_with_no_tests(corrupt) -> None:
    def mutate(data: dict[str, Any]) -> None:
        row = next(r for r in data["requirements"] if r["status"] == "partial")
        row["tests"] = []

    assert corrupt(mutate) == 1


def test_guard_rejects_a_nonexistent_implementing_path(corrupt) -> None:
    def mutate(data: dict[str, Any]) -> None:
        row = next(r for r in data["requirements"] if r["status"] == "partial")
        row["files"] = ["src/p1am_control_system/backend/timescale_writer.py"]

    assert corrupt(mutate) == 1


def test_guard_rejects_totals_that_disagree_with_the_rows(corrupt) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["totals"]["scada"]["landed"] = 16

    assert corrupt(mutate) == 1


def test_guard_rejects_a_dropped_requirement(corrupt) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["requirements"] = [r for r in data["requirements"] if r["id"] != "F12"]
        data["totals"]["scada"]["missing"] -= 1
        data["totals"]["scada"]["total"] -= 1

    assert corrupt(mutate) == 1


def test_guard_rejects_a_row_with_no_gaps(corrupt) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["requirements"][0]["gaps"] = []

    assert corrupt(mutate) == 1


def test_guard_rejects_a_carrier_pr_recorded_without_a_full_sha(corrupt) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["closed_carrier_prs"][0]["head_oid"] = "2259f59"

    assert corrupt(mutate) == 1


def test_guard_rejects_a_stale_rendered_count(corrupt) -> None:
    """Editing the JSON without re-rendering the markdown must fail."""

    def mutate(data: dict[str, Any]) -> None:
        row = next(r for r in data["requirements"] if r["status"] == "partial")
        row["status"] = "missing"
        row["files"] = []
        row["tests"] = []
        row["delivered"] = ""
        row["evidence_prs"] = []
        data["totals"]["scada"]["partial"] -= 1
        data["totals"]["scada"]["missing"] += 1

    assert corrupt(mutate) == 1


# --------------------------------------------------------------------------- #
# Recovery ledger (docs/scada/recovery_ledger.v1.json)                        #
# --------------------------------------------------------------------------- #
LEDGER_PATH = ROOT / "docs" / "scada" / "recovery_ledger.v1.json"
DECISIONS = {"re-land", "obsolete", "needs-owner"}


def _ledger() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(LEDGER_PATH.read_text(encoding="utf-8")))


def test_every_recovered_file_carries_exactly_one_decision() -> None:
    files = _ledger()["files"]
    paths = [str(f["path"]) for f in files]
    assert len(set(paths)) == len(paths), "a file is classified twice"
    for entry in files:
        assert entry["decision"] in DECISIONS, entry["path"]


def test_re_land_files_are_actually_still_missing_from_main() -> None:
    """A `re-land` that already exists is not a re-land -- it is a stale ledger.

    This is the assertion that makes the ledger self-expiring: as the corpus is
    recovered, the rows for landed files must be reclassified rather than left
    claiming work that is already done.
    """
    for entry in _ledger()["files"]:
        if entry["decision"] == "re-land":
            assert not (ROOT / str(entry["path"])).exists(), entry["path"]


def test_every_cluster_states_imports_and_a_rationale() -> None:
    for cluster in _ledger()["clusters"]:
        cid = cluster["id"]
        assert str(cluster["rationale"]).strip(), cid
        assert str(cluster["name"]).strip(), cid
        assert "external_imports" in cluster, cid
        assert cluster["decision"] in DECISIONS, cid


def test_file_decisions_agree_with_their_cluster() -> None:
    ledger = _ledger()
    by_id = {str(c["id"]): c for c in ledger["clusters"]}
    for entry in ledger["files"]:
        cluster = by_id[str(entry["cluster"])]
        assert entry["decision"] == cluster["decision"], entry["path"]


def test_ledger_totals_survive_a_recount() -> None:
    ledger = _ledger()
    files = ledger["files"]
    for decision in DECISIONS:
        counted = sum(1 for f in files if f["decision"] == decision)
        assert ledger["totals"][decision] == counted, decision
    assert ledger["totals"]["files_classified"] == len(files)


def test_the_never_track_artifacts_stay_obsolete() -> None:
    """`dcs_scada.db` is a runtime file main's .gitignore already excludes.

    Re-landing it would reintroduce a tracked file the repo decided not to
    track, so its decision is pinned rather than left to judgement. Same for the
    six accidental `.codex-worktrees` gitlinks, which have no `.gitmodules` and
    would leave main with broken submodules.
    """
    by_path = {str(f["path"]): f for f in _ledger()["files"]}
    assert by_path["dcs_scada.db"]["decision"] == "obsolete"
    gitlinks = [p for p in by_path if p.startswith(".codex-worktrees/")]
    assert len(gitlinks) == 6
    for path in gitlinks:
        assert by_path[path]["decision"] == "obsolete", path


def test_the_licence_gated_clusters_are_not_marked_re_land() -> None:
    """Grafana (AGPLv3) and the TSL-covered Timescale schema need a ruling first."""
    by_path = {str(f["path"]): f for f in _ledger()["files"]}
    encumbered = [
        p
        for p in by_path
        if "/timescale/" in p or "/grafana/" in p or "/deploy/historian/" in p
    ]
    assert encumbered, "expected licence-gated files in the corpus"
    for path in encumbered:
        assert by_path[path]["decision"] == "needs-owner", path


def test_the_managed_bypass_module_needs_an_owner() -> None:
    """Bypass authority on a protection system is never an audit-time decision."""
    by_path = {str(f["path"]): f for f in _ledger()["files"]}
    key = "src/p1am_control_system/backend/protection_management.py"
    assert by_path[key]["decision"] == "needs-owner"
