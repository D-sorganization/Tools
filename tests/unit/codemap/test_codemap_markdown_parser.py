from __future__ import annotations

from dataclasses import dataclass, field

from codemap._ts_common import ParsedSymbol

from codemap import _lang_markdown as markdown
from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

# Scoped to this module only; a session-wide skip hook silenced the whole
# suite here once already (issue #4497).
pytestmark = CODEMAP_DEPS_SKIP


def test_extract_emits_atx_headings_with_truncated_symbol_fields(monkeypatch) -> None:
    long_title = "x" * 90
    source = f"# Project\n\nIntro text\n\n## {long_title}\n".encode()
    second_heading_start = source.index(b"## ")
    root = FakeNode(
        "document",
        children=[
            FakeNode(
                "atx_heading",
                start_byte=0,
                end_byte=len(b"# Project"),
                start_point=(0, 0),
                end_point=(0, 9),
            ),
            FakeNode(
                "atx_heading",
                start_byte=second_heading_start,
                end_byte=len(source.rstrip()),
                start_point=(4, 0),
                end_point=(4, 93),
            ),
        ],
    )
    monkeypatch.setattr(markdown, "get_parser", lambda lang_id: FakeParser(root))

    result = markdown.extract(
        "README.md",
        source,
    )

    assert result.language == "markdown"
    assert result.imports == []
    assert [symbol.kind for symbol in result.symbols] == ["heading", "heading"]
    assert [symbol.name for symbol in result.symbols] == ["Project", "x" * 80]
    assert result.symbols[1].qualified == long_title
    assert result.symbols[1].sig == long_title
    assert result.symbols[0].start_line == 1
    assert result.symbols[1].start_line == 5


def test_extract_returns_empty_result_when_markdown_parser_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(markdown, "get_parser", lambda lang_id: None)

    result = markdown.extract("README.md", "# Missing parser\n")

    assert result.language == "markdown"
    assert result.imports == []
    assert result.symbols == []


@dataclass
class FakeNode:
    type: str
    start_byte: int = 0
    end_byte: int = 0
    start_point: tuple[int, int] = (0, 0)
    end_point: tuple[int, int] = (0, 0)
    children: list[FakeNode] = field(default_factory=list)


@dataclass
class FakeTree:
    root_node: FakeNode


class FakeParser:
    def __init__(self, root_node: FakeNode) -> None:
        self.root_node = root_node

    def parse(self, source: bytes) -> FakeTree:
        self.source = source
        return FakeTree(self.root_node)


def test_walk_uses_raw_heading_text_and_skips_blank_headings() -> None:
    source = b"## Raw heading\n###   \n"
    raw_heading = FakeNode(
        type="atx_heading",
        start_byte=0,
        end_byte=len(b"## Raw heading"),
        start_point=(0, 0),
        end_point=(0, 14),
    )
    blank_heading = FakeNode(
        type="atx_heading",
        start_byte=len(b"## Raw heading\n"),
        end_byte=len(source),
        start_point=(1, 0),
        end_point=(1, 6),
    )
    root = FakeNode("document", children=[raw_heading, blank_heading])
    symbols: list[ParsedSymbol] = []

    markdown._walk(root, source, symbols)

    assert symbols == [
        ParsedSymbol(
            kind="heading",
            name="Raw heading",
            qualified="Raw heading",
            sig="Raw heading",
            docstring="",
            start_line=1,
            end_line=1,
        )
    ]
