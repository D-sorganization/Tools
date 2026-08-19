from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import pytest

from codemap import _ts_common as ts_common
from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

# Scoped to this module only; a session-wide skip hook silenced the whole
# suite here once already (issue #4497).
pytestmark = CODEMAP_DEPS_SKIP


@pytest.fixture(autouse=True)
def reset_parser_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts_common, "_LANG_CACHE", {})
    monkeypatch.setattr(ts_common, "_PARSER_CACHE", {})
    monkeypatch.setattr(ts_common, "_MISSING_WARNED", set())


def test_node_helpers_decode_children_and_lines() -> None:
    source = b"alpha beta\nbad\n"
    beta_start = source.index(b"beta")
    beta = FakeNode(
        "identifier",
        start_byte=beta_start,
        end_byte=beta_start + len(b"beta"),
        start_point=(0, 6),
        end_point=(0, 10),
    )
    parent = FakeNode(
        "parent",
        start_point=(4, 0),
        end_point=(6, 3),
        children=[FakeNode("comment"), beta],
    )

    assert ts_common.to_bytes(source) is source
    assert ts_common.to_bytes("plain text") == b"plain text"
    assert ts_common.text_of(beta, source) == "beta"
    assert ts_common.first_child(parent, "identifier") is beta
    assert ts_common.first_child(parent, "missing") is None
    assert ts_common.line_range(parent) == (5, 7)


def test_get_parser_returns_none_for_unknown_language() -> None:
    assert ts_common.get_parser("unsupported") is None
    assert ts_common._PARSER_CACHE == {}


def test_get_parser_caches_successful_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_language_module = types.ModuleType("fake_tree_sitter_language")
    fake_language_module.language = lambda: "raw-language"
    fake_tree_sitter = types.ModuleType("tree_sitter")
    fake_tree_sitter.Language = FakeLanguage
    fake_tree_sitter.Parser = FakeParser
    monkeypatch.setitem(
        ts_common.LANG_MODULES,
        "fake",
        ("fake_tree_sitter_language", "language"),
    )
    monkeypatch.setitem(sys.modules, "fake_tree_sitter_language", fake_language_module)
    monkeypatch.setitem(sys.modules, "tree_sitter", fake_tree_sitter)

    parser = ts_common.get_parser("fake")
    cached = ts_common.get_parser("fake")

    assert parser is cached
    assert isinstance(parser, FakeParser)
    assert parser.language.raw == "raw-language"
    assert ts_common._LANG_CACHE["fake"].raw == "raw-language"
    assert ts_common._PARSER_CACHE["fake"] is parser


def test_get_parser_logs_and_caches_missing_language_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(
        ts_common.LANG_MODULES,
        "missing",
        ("missing_tree_sitter_language_for_test", "language"),
    )
    caplog.set_level("WARNING", logger=ts_common.logger.name)

    assert ts_common.get_parser("missing") is None
    assert ts_common.get_parser("missing") is None

    messages = [record.message for record in caplog.records]
    assert messages == [
        "codemap: language 'missing' unavailable "
        "(No module named 'missing_tree_sitter_language_for_test'); skipping"
    ]
    assert ts_common._PARSER_CACHE["missing"] is None
    assert ts_common._MISSING_WARNED == {"missing_tree_sitter_language_for_test"}


def test_get_parser_logs_and_caches_initialization_failure_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_language_module = types.ModuleType("broken_tree_sitter_language")
    fake_language_module.language = lambda: "raw-language"
    fake_tree_sitter = types.ModuleType("tree_sitter")
    fake_tree_sitter.Language = RaisingLanguage
    fake_tree_sitter.Parser = FakeParser
    monkeypatch.setitem(
        ts_common.LANG_MODULES,
        "broken",
        ("broken_tree_sitter_language", "language"),
    )
    monkeypatch.setitem(
        sys.modules,
        "broken_tree_sitter_language",
        fake_language_module,
    )
    monkeypatch.setitem(sys.modules, "tree_sitter", fake_tree_sitter)
    caplog.set_level("WARNING", logger=ts_common.logger.name)

    assert ts_common.get_parser("broken") is None
    assert ts_common.get_parser("broken") is None

    messages = [record.message for record in caplog.records]
    assert messages == [
        "codemap: failed to initialise 'broken' parser: cannot wrap language"
    ]
    assert ts_common._PARSER_CACHE["broken"] is None
    assert ts_common._MISSING_WARNED == {"broken_tree_sitter_language"}


@dataclass
class FakeNode:
    type: str
    start_byte: int = 0
    end_byte: int = 0
    start_point: tuple[int, int] = (0, 0)
    end_point: tuple[int, int] = (0, 0)
    children: list[FakeNode] = field(default_factory=list)


@dataclass
class FakeLanguage:
    raw: str


class RaisingLanguage:
    def __init__(self, raw: str) -> None:
        raise RuntimeError("cannot wrap language")


@dataclass
class FakeParser:
    language: FakeLanguage
