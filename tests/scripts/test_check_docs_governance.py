"""Contracts for scripts/check_docs_governance.py."""

from __future__ import annotations

from scripts.check_docs_governance import adr_readme_error

_ADR = "docs/adr/ADR-0050-plant-historian-timescaledb.md"
_README = "docs/adr/README.md"


def test_format_only_adr_edit_passes_when_records_index_is_fresh() -> None:
    assert adr_readme_error([_ADR], index_is_fresh=True) is None


def test_adr_edit_fails_when_records_index_is_stale_and_readme_is_untouched() -> None:
    assert (
        adr_readme_error([_ADR], index_is_fresh=False)
        == "ADR docs changed without updating docs/adr/README.md"
    )


def test_readme_in_the_diff_satisfies_the_gate_before_the_index_check() -> None:
    assert adr_readme_error([_ADR, _README], index_is_fresh=False) is None


def test_no_adr_body_change_does_not_require_the_readme() -> None:
    assert adr_readme_error(["docs/README.md"], index_is_fresh=False) is None
