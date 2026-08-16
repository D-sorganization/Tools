from __future__ import annotations

from dataclasses import dataclass, field

from codemap._ts_common import ParsedSymbol

from codemap import _lang_rust as rust_parser
from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

# Scoped to this module only; a session-wide skip hook silenced the whole
# suite here once already (issue #4497).
pytestmark = CODEMAP_DEPS_SKIP


def test_extract_emits_imports_functions_structs_modules_and_impls(
    monkeypatch,
) -> None:
    source_text = (
        "use std::fs;\n"
        "use crate::module::{Thing};\n"
        "\n"
        "fn top(value: i32) -> i32 {\n"
        "    value\n"
        "}\n"
        "\n"
        "struct Worker {\n"
        "    value: i32,\n"
        "}\n"
        "\n"
        "impl Worker {\n"
        "    fn run(&self) -> i32 {\n"
        "        self.value\n"
        "    }\n"
        "}\n"
        "\n"
        "impl {\n"
        "    fn loose() {}\n"
        "}\n"
        "\n"
        "mod nested {\n"
        "    fn inner() {}\n"
        "    struct Inner;\n"
        "    impl Inner {\n"
        "        fn inside() {}\n"
        "    }\n"
        "}\n"
    )
    source = source_text.encode()
    root = FakeNode(
        "source_file",
        children=[
            _line_node(source, "use_declaration", b"use std::fs;"),
            _line_node(source, "use_declaration", b"use crate::module::{Thing};"),
            _function_node(source, b"fn top", b"top"),
            _struct_node(source, b"struct Worker", b"Worker"),
            FakeNode(
                "impl_item",
                children=[
                    _text_node(source, "type_identifier", b"Worker"),
                    FakeNode(
                        "declaration_list",
                        children=[_function_node(source, b"fn run", b"run")],
                    ),
                ],
            ),
            FakeNode(
                "impl_item",
                children=[
                    FakeNode(
                        "declaration_list",
                        children=[_function_node(source, b"fn loose", b"loose")],
                    )
                ],
            ),
            FakeNode(
                "mod_item",
                children=[
                    _text_node(source, "identifier", b"nested"),
                    FakeNode(
                        "declaration_list",
                        children=[
                            _function_node(source, b"fn inner", b"inner"),
                            _struct_node(source, b"struct Inner", b"Inner"),
                            FakeNode(
                                "impl_item",
                                children=[
                                    _text_node(source, "type_identifier", b"Inner"),
                                    FakeNode(
                                        "declaration_list",
                                        children=[
                                            _function_node(
                                                source,
                                                b"fn inside",
                                                b"inside",
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    monkeypatch.setattr(rust_parser, "get_parser", lambda lang_id: FakeParser(root))

    result = rust_parser.extract("sample.rs", source)

    assert result.language == "rust"
    assert result.imports == ["use std::fs", "use crate::module::{Thing}"]
    assert [
        (symbol.kind, symbol.name, symbol.qualified, symbol.sig)
        for symbol in result.symbols
    ] == [
        ("function", "top", "top", "fn top(value: i32) -> i32"),
        ("struct", "Worker", "Worker", "struct Worker"),
        ("function", "run", "Worker::run", "fn run(&self) -> i32"),
        ("function", "loose", "loose", "fn loose() {}"),
        ("function", "inner", "nested::inner", "fn inner() {}"),
        ("struct", "Inner", "nested::Inner", "struct Inner"),
        ("function", "inside", "nested::Inner::inside", "fn inside() {}"),
    ]
    assert all(symbol.docstring == "" for symbol in result.symbols)


def test_extract_returns_empty_result_when_rust_parser_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rust_parser, "get_parser", lambda lang_id: None)

    result = rust_parser.extract("missing.rs", "fn missing() {}")

    assert result.language == "rust"
    assert result.imports == []
    assert result.symbols == []


def test_walk_skips_incomplete_items_and_missing_bodies() -> None:
    source = b"module\nfn nested() {}\n"
    symbols: list[ParsedSymbol] = []

    rust_parser._walk(
        FakeNode(
            "source_file",
            children=[
                FakeNode("function_item"),
                FakeNode("struct_item"),
                FakeNode(
                    "impl_item",
                    children=[FakeNode("type_identifier", start_byte=0, end_byte=6)],
                ),
                FakeNode("mod_item"),
                FakeNode(
                    "mod_item",
                    children=[
                        FakeNode("identifier", start_byte=0, end_byte=6),
                        FakeNode(
                            "declaration_list",
                            children=[_function_node(source, b"fn nested", b"nested")],
                        ),
                    ],
                ),
            ],
        ),
        source,
        "Outer",
        symbols,
    )

    assert [(symbol.name, symbol.qualified) for symbol in symbols] == [
        ("nested", "Outer::module::nested")
    ]


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


def _line_node(source: bytes, type_name: str, needle: bytes) -> FakeNode:
    start = source.index(needle)
    end = start + len(needle)
    return FakeNode(
        type_name,
        start_byte=start,
        end_byte=end,
        start_point=_point_for_offset(source, start),
        end_point=_point_for_offset(source, end),
    )


def _text_node(
    source: bytes,
    type_name: str,
    needle: bytes,
    start: int = 0,
) -> FakeNode:
    offset = source.index(needle, start)
    return FakeNode(
        type_name,
        start_byte=offset,
        end_byte=offset + len(needle),
        start_point=_point_for_offset(source, offset),
        end_point=_point_for_offset(source, offset + len(needle)),
    )


def _function_node(source: bytes, start: bytes, name: bytes) -> FakeNode:
    start_offset = source.index(start)
    end_offset = source.find(b"\n", start_offset)
    if end_offset == -1:
        end_offset = len(source)
    return FakeNode(
        "function_item",
        start_byte=start_offset,
        end_byte=end_offset,
        start_point=_point_for_offset(source, start_offset),
        end_point=_point_for_offset(source, end_offset),
        children=[_text_node(source, "identifier", name, start_offset)],
    )


def _struct_node(source: bytes, start: bytes, name: bytes) -> FakeNode:
    start_offset = source.index(start)
    end_offset = source.find(b"\n", start_offset)
    if end_offset == -1:
        end_offset = len(source)
    return FakeNode(
        "struct_item",
        start_byte=start_offset,
        end_byte=end_offset,
        start_point=_point_for_offset(source, start_offset),
        end_point=_point_for_offset(source, end_offset),
        children=[_text_node(source, "type_identifier", name, start_offset)],
    )


def _point_for_offset(source: bytes, offset: int) -> tuple[int, int]:
    prefix = source[:offset]
    line = prefix.count(b"\n")
    last_newline = prefix.rfind(b"\n")
    column = offset if last_newline == -1 else offset - last_newline - 1
    return line, column
