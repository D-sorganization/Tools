"""Bounded JSON persistence for localized attribution authority."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.ui.pyqt6.localized_attribution_archive import (
    ARCHIVED_AUTHORITY_DISCLAIMER,
    MAX_AUTHORITY_JSON_BYTES,
    authority_from_json,
    authority_to_json,
    read_authority_json,
    write_authority_json,
)
from rate_of_closure.variation.localized_attribution import (
    attribution_authority_from_dict,
)
from shared.python.contracts import ContractViolationError

FIXTURE = Path(__file__).parent / "fixtures" / "localized_attribution_authority_v1.json"


def _authority():  # type: ignore[no-untyped-def]
    return attribution_authority_from_dict(json.loads(FIXTURE.read_text("utf-8")))


def test_authority_json_is_canonical_finite_and_round_trips(tmp_path: Path) -> None:
    authority = _authority()

    text = authority_to_json(authority)
    path = tmp_path / "authority.json"
    write_authority_json(path, authority)

    assert text == path.read_text("utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert text == json.dumps(
        json.loads(text), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert authority_from_json(text) == authority
    assert read_authority_json(path) == authority
    assert "not rerun or provenance-verified" in ARCHIVED_AUTHORITY_DISCLAIMER


def test_reader_rejects_oversized_bytes_before_json_parse(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b"{" + b" " * MAX_AUTHORITY_JSON_BYTES)

    with pytest.raises(ContractViolationError, match="byte cap"):
        read_authority_json(path)


@pytest.mark.parametrize("text", ["not-json", '{"schema_id":NaN}'])
def test_reader_rejects_invalid_or_nonfinite_json(text: str) -> None:
    with pytest.raises((ContractViolationError, ValueError)):
        authority_from_json(text)
