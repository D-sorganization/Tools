"""Contracts for the divergence ledger and its paired-PR gate (Tools #4915)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_divergence_ledger as mod

ROOT = Path(__file__).resolve().parents[2]


def _ledger(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": mod.SCHEMA_VERSION,
        "title": "t",
        "updated": "2026-09-03",
        "pins": {"tools": "a", "upstreamdrift": "b"},
        "rulings": {},
        "gate": {"rule": "r"},
        "rows": rows,
    }


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "module": "package:ai",
        "tools_path": "src/shared/python/ai",
        "ud_path": "src/shared/python/ai",
        "ruling": "deferred",
        "owner": "seam epic",
        "source_issue": "UpstreamDrift#9406",
        "target_pr": None,
        "status": "pending-inventory",
        "rulings": [],
        "inventory": None,
        "notes": "n",
    }
    base.update(overrides)
    return base


def test_repository_ledger_loads_and_markdown_is_fresh() -> None:
    document = mod.load_ledger(mod.LEDGER_PATH)
    rows = mod.rows_of(document)
    assert len(rows) >= 60
    modules = {row["module"] for row in rows}
    assert "package:launch_monitor" in modules
    assert "package:sidekick" in modules
    assert "rate_of_closure/player_covariation.py" in modules
    d_refs = {
        ref
        for row in rows
        for ref in (
            row["rulings"] if isinstance(row["rulings"], (list, tuple, set)) else []
        )
    }
    assert {"D1", "D15", "D17", "D22", "D23", "D30"} <= d_refs
    assert mod.main(["--check"]) == 0


def test_repository_handoff_points_at_the_ledger() -> None:
    handoff = (ROOT / "AGENT_HANDOFF.md").read_text(encoding="utf-8")
    assert "docs/shared/divergence_ledger.v1.json" in handoff


def test_load_rejects_bad_ruling(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(_ledger([_row(ruling="maybe")])), encoding="utf-8")
    with pytest.raises(mod.LedgerError, match="ruling"):
        mod.load_ledger(path)


def test_gate_ignores_unledgered_files() -> None:
    code, lines = mod.gate(_ledger([_row()]), ["src/other/x.py"], body="")
    assert code == 0
    assert "no ledgered module touched" in lines[0]


def test_gate_requires_pair_for_deferred_row() -> None:
    code, lines = mod.gate(_ledger([_row()]), ["src/shared/python/ai/x.py"], body="")
    assert code == 1
    assert any("UD-PAIR" in line for line in lines)


def test_gate_accepts_pair_reference() -> None:
    body = "Closes #1\n\nUD-PAIR: D-sorganization/UpstreamDrift#9432\n"
    code, _lines = mod.gate(_ledger([_row()]), ["src/shared/python/ai/x.py"], body)
    assert code == 0


@pytest.mark.parametrize(
    "overrides",
    [{"ruling": "tools-canonical"}, {"status": "ud-copy-deleted"}],
)
def test_gate_exempts_tools_canonical_and_deleted_copies(
    overrides: dict[str, object],
) -> None:
    code, _lines = mod.gate(
        _ledger([_row(**overrides)]), ["src/shared/python/ai/x.py"], body=""
    )
    assert code == 0


def test_gate_does_not_enforce_outside_pull_requests() -> None:
    code, lines = mod.gate(_ledger([_row()]), ["src/shared/python/ai/x.py"], None)
    assert code == 0
    assert any("not a pull_request" in line for line in lines)


def test_most_specific_row_governs() -> None:
    rows = [
        _row(),
        _row(
            module="ai/special.py",
            tools_path="src/shared/python/ai/special.py",
            ruling="tools-canonical",
            status="in-sync",
        ),
    ]
    code, _ = mod.gate(_ledger(rows), ["src/shared/python/ai/special.py"], body="")
    assert code == 0
    code, _ = mod.gate(_ledger(rows), ["src/shared/python/ai/other.py"], body="")
    assert code == 1


def test_pr_body_from_event(tmp_path: Path) -> None:
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"body": "UD-PAIR: x"}}))
    assert mod.pr_body_from_event(str(event)) == "UD-PAIR: x"
    assert mod.pr_body_from_event(None) is None
    event.write_text(json.dumps({"ref": "refs/heads/main"}))
    assert mod.pr_body_from_event(str(event)) is None
