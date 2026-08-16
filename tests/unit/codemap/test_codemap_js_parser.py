from __future__ import annotations

from dataclasses import dataclass, field

from codemap._ts_common import ParsedSymbol

from codemap import _lang_js as js_parser
from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

# Scoped to this module only; a session-wide skip hook silenced the whole
# suite here once already (issue #4497).
pytestmark = CODEMAP_DEPS_SKIP


def test_extract_javascript_emits_imports_and_symbols(monkeypatch) -> None:
    source_text = (
        'import tools from "pkg";\n'
        "function top(value) {\n"
        "  return value;\n"
        "}\n"
        "export function exported() {}\n"
        "declare function declared(): void;\n"
        "class Worker {\n"
        "  run(input) {\n"
        "    return input;\n"
        "  }\n"
        "}\n"
        "abstract class Base {\n"
        "  execute() {}\n"
        "}\n"
        "const arrowed = () => true;\n"
        "let expressed = function () { return true; };\n"
        "const ignored = 1;\n"
    )
    source = source_text.encode()
    top_function = _function_node(source, b"function top", b"top")
    exported_function = _function_node(source, b"function exported", b"exported")
    declared_function = _function_node(source, b"function declared", b"declared")
    worker_class = _class_node(
        source,
        b"class Worker",
        b"Worker",
        "type_identifier",
        [_method_node(source, b"run(input)", b"run", "property_identifier")],
    )
    base_class = _class_node(
        source,
        b"class Base",
        b"Base",
        "identifier",
        [_method_node(source, b"execute()", b"execute", "identifier")],
        node_type="abstract_class_declaration",
    )
    root = FakeNode(
        "program",
        children=[
            FakeNode(
                "import_statement",
                children=[_text_node(source, "string", b'"pkg"')],
            ),
            top_function,
            FakeNode("export_statement", children=[exported_function]),
            FakeNode("ambient_declaration", children=[declared_function]),
            worker_class,
            base_class,
            _variable_function(source, b"const arrowed", b"arrowed", "arrow_function"),
            _variable_function(
                source,
                b"let expressed",
                b"expressed",
                "function_expression",
                declaration_type="variable_declaration",
            ),
            FakeNode(
                "lexical_declaration",
                children=[
                    FakeNode(
                        "variable_declarator",
                        children=[
                            _text_node(source, "identifier", b"ignored"),
                            _text_node(source, "number", b"1"),
                        ],
                    )
                ],
            ),
        ],
    )
    monkeypatch.setattr(js_parser, "get_parser", lambda lang_id: FakeParser(root))

    result = js_parser.extract_javascript("sample.js", source)

    assert result.language == "javascript"
    assert result.imports == ["pkg"]
    assert [
        (symbol.kind, symbol.name, symbol.qualified, symbol.sig)
        for symbol in result.symbols
    ] == [
        ("function", "top", "top", "function top(value) {"),
        ("function", "exported", "exported", "function exported() {}"),
        ("function", "declared", "declared", "function declared(): void;"),
        ("class", "Worker", "Worker", "class Worker"),
        ("method", "run", "Worker.run", "run(input) {"),
        ("class", "Base", "Base", "class Base"),
        ("method", "execute", "Base.execute", "execute() {}"),
        ("function", "arrowed", "arrowed", "const arrowed = () => true;"),
        (
            "function",
            "expressed",
            "expressed",
            "let expressed = function () { return true; };",
        ),
    ]
    assert all(symbol.docstring == "" for symbol in result.symbols)


def test_extract_typescript_uses_extension_language_and_same_symbol_walk(
    monkeypatch,
) -> None:
    source = b"const typed = function () { return 1; };\n"
    root = FakeNode(
        "program",
        children=[
            _variable_function(
                source,
                b"const typed",
                b"typed",
                "function",
            )
        ],
    )
    seen_languages: list[str] = []

    def fake_get_parser(lang_id: str) -> FakeParser:
        seen_languages.append(lang_id)
        return FakeParser(root)

    monkeypatch.setattr(js_parser, "get_parser", fake_get_parser)

    ts_result = js_parser.extract_typescript("component.ts", source)
    tsx_result = js_parser.extract_typescript("component.tsx", source)

    assert seen_languages == ["typescript", "tsx"]
    assert ts_result.language == "typescript"
    assert tsx_result.language == "tsx"
    assert [symbol.qualified for symbol in ts_result.symbols] == ["typed"]
    assert [symbol.qualified for symbol in tsx_result.symbols] == ["typed"]


def test_extract_returns_empty_result_when_parser_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(js_parser, "get_parser", lambda lang_id: None)

    js_result = js_parser.extract_javascript("missing.js", "function missing() {}")
    ts_result = js_parser.extract_typescript("missing.ts", "const missing = () => 1")
    tsx_result = js_parser.extract_typescript("missing.tsx", "const el = <div />")

    assert js_result.language == "javascript"
    assert ts_result.language == "typescript"
    assert tsx_result.language == "tsx"
    assert js_result.imports == ts_result.imports == tsx_result.imports == []
    assert js_result.symbols == ts_result.symbols == tsx_result.symbols == []


def test_walk_skips_incomplete_nodes_and_recurses_into_prefixed_exports() -> None:
    source = b"Broken\nfunction nested() {}\n"
    symbols: list[ParsedSymbol] = []
    js_parser._walk(
        FakeNode(
            "program",
            children=[
                FakeNode("function_declaration"),
                FakeNode("class_declaration"),
                FakeNode(
                    "class_declaration",
                    children=[
                        FakeNode("type_identifier", start_byte=0, end_byte=6),
                        FakeNode(
                            "class_body",
                            children=[FakeNode("method_definition")],
                        ),
                    ],
                ),
                FakeNode(
                    "lexical_declaration",
                    children=[
                        FakeNode("variable_declarator"),
                        FakeNode(
                            "variable_declarator",
                            children=[
                                FakeNode("identifier", start_byte=0, end_byte=6),
                                FakeNode("number", start_byte=0, end_byte=1),
                            ],
                        ),
                    ],
                ),
                FakeNode(
                    "export_statement",
                    children=[_function_node(source, b"function nested", b"nested")],
                ),
            ],
        ),
        source,
        "Outer",
        symbols,
    )

    assert [(symbol.name, symbol.qualified) for symbol in symbols] == [
        ("Broken", "Outer.Broken"),
        ("nested", "Outer.nested"),
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
        "function_declaration",
        start_byte=start_offset,
        end_byte=end_offset,
        start_point=_point_for_offset(source, start_offset),
        end_point=_point_for_offset(source, end_offset),
        children=[_text_node(source, "identifier", name, start_offset)],
    )


def _class_node(
    source: bytes,
    start: bytes,
    name: bytes,
    name_type: str,
    methods: list[FakeNode],
    node_type: str = "class_declaration",
) -> FakeNode:
    start_offset = source.index(start)
    end_offset = source.find(b"}\n", start_offset)
    if end_offset == -1:
        end_offset = len(source)
    else:
        end_offset += 1
    return FakeNode(
        node_type,
        start_byte=start_offset,
        end_byte=end_offset,
        start_point=_point_for_offset(source, start_offset),
        end_point=_point_for_offset(source, end_offset),
        children=[
            _text_node(source, name_type, name, start_offset),
            FakeNode("class_body", children=methods),
        ],
    )


def _method_node(
    source: bytes,
    start: bytes,
    name: bytes,
    name_type: str,
) -> FakeNode:
    start_offset = source.index(start)
    end_offset = source.find(b"\n", start_offset)
    if end_offset == -1:
        end_offset = len(source)
    return FakeNode(
        "method_definition",
        start_byte=start_offset,
        end_byte=end_offset,
        start_point=_point_for_offset(source, start_offset),
        end_point=_point_for_offset(source, end_offset),
        children=[_text_node(source, name_type, name, start_offset)],
    )


def _variable_function(
    source: bytes,
    start: bytes,
    name: bytes,
    value_type: str,
    declaration_type: str = "lexical_declaration",
) -> FakeNode:
    start_offset = source.index(start)
    end_offset = source.find(b"\n", start_offset)
    if end_offset == -1:
        end_offset = len(source)
    value_offset = source.index(name, start_offset) + len(name)
    return FakeNode(
        declaration_type,
        start_byte=start_offset,
        end_byte=end_offset,
        start_point=_point_for_offset(source, start_offset),
        end_point=_point_for_offset(source, end_offset),
        children=[
            FakeNode(
                "variable_declarator",
                children=[
                    _text_node(source, "identifier", name, start_offset),
                    FakeNode(value_type, start_byte=value_offset, end_byte=end_offset),
                ],
            )
        ],
    )


def _point_for_offset(source: bytes, offset: int) -> tuple[int, int]:
    prefix = source[:offset]
    line = prefix.count(b"\n")
    last_newline = prefix.rfind(b"\n")
    column = offset if last_newline == -1 else offset - last_newline - 1
    return line, column
