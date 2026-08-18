from __future__ import annotations

from dataclasses import dataclass, field

from codemap._ts_common import ParsedSymbol

from codemap import _lang_python as python_parser
from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

# Scoped to this module only; a session-wide skip hook silenced the whole
# suite here once already (issue #4497).
pytestmark = CODEMAP_DEPS_SKIP


def test_extract_emits_imports_symbols_docstrings_signatures_and_calls(
    monkeypatch,
) -> None:
    source_text = (
        "import os\n"
        "from . import local\n"
        "from package.sub import thing\n"
        "\n"
        "@decorator\n"
        "def top(value):\n"
        '    """Top doc.\n'
        "    more\n"
        '    """\n'
        "    print(value)\n"
        "    helper()\n"
        "    value.method()\n"
        "\n"
        "class Worker:\n"
        "    'Worker doc.'\n"
        "    def run(self):\n"
        '        """Run doc."""\n'
        "        return top(self)\n"
    )
    source = source_text.encode()
    top_function = FakeNode(
        "function_definition",
        start_byte=source.index(b"def top"),
        end_byte=source.index(b"class Worker"),
        children=[
            _text_node(source, "identifier", b"top"),
            _text_node(source, ":", b':\n    """Top doc.'),
            FakeNode(
                "block",
                children=[
                    FakeNode(
                        "expression_statement",
                        children=[
                            _text_node(
                                source,
                                "string",
                                b'"""Top doc.\n    more\n    """',
                            )
                        ],
                    ),
                    FakeNode(
                        "call",
                        children=[_text_node(source, "identifier", b"print")],
                    ),
                    FakeNode(
                        "call",
                        children=[_text_node(source, "identifier", b"helper")],
                    ),
                    FakeNode(
                        "call",
                        children=[_text_node(source, "attribute", b"value.method")],
                    ),
                ],
            ),
        ],
    )
    run_method = FakeNode(
        "function_definition",
        start_byte=source.index(b"def run"),
        end_byte=len(source),
        children=[
            _text_node(source, "identifier", b"run"),
            _text_node(source, ":", b':\n        """Run doc.'),
            FakeNode(
                "block",
                children=[
                    FakeNode(
                        "expression_statement",
                        children=[_text_node(source, "string", b'"""Run doc."""')],
                    ),
                    FakeNode(
                        "call",
                        children=[_text_node(source, "identifier", b"top", -1)],
                    ),
                ],
            ),
        ],
    )
    worker_class = FakeNode(
        "class_definition",
        start_byte=source.index(b"class Worker"),
        end_byte=len(source),
        children=[
            _text_node(source, "identifier", b"Worker"),
            _text_node(source, ":", b":\n    'Worker doc.'"),
            FakeNode(
                "block",
                children=[
                    FakeNode(
                        "expression_statement",
                        children=[_text_node(source, "string", b"'Worker doc.'")],
                    ),
                    run_method,
                ],
            ),
        ],
    )
    root = FakeNode(
        "module",
        children=[
            FakeNode(
                "import_statement",
                children=[_text_node(source, "dotted_name", b"os")],
            ),
            FakeNode(
                "import_from_statement",
                children=[_text_node(source, "dotted_name", b"local")],
            ),
            FakeNode(
                "import_from_statement",
                children=[_text_node(source, "dotted_name", b"package.sub")],
            ),
            FakeNode(
                "decorated_definition",
                children=[FakeNode("decorator"), top_function],
            ),
            worker_class,
        ],
    )
    monkeypatch.setattr(python_parser, "get_parser", lambda lang_id: FakeParser(root))

    result = python_parser.extract("sample.py", source)

    assert result.language == "python"
    assert result.imports == ["os", "local", "package.sub"]
    assert [
        (symbol.kind, symbol.name, symbol.qualified, symbol.sig, symbol.docstring)
        for symbol in result.symbols
    ] == [
        ("function", "top", "top", "def top(value):", "Top doc."),
        ("class", "Worker", "Worker", "class Worker:", "Worker doc."),
        ("method", "run", "Worker.run", "def run(self):", "Run doc."),
    ]
    assert result.symbols[0].calls_out == ["helper", "print", "value.method"]
    assert result.symbols[2].calls_out == ["top"]


def test_extract_returns_empty_result_when_python_parser_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(python_parser, "get_parser", lambda lang_id: None)

    result = python_parser.extract("sample.py", "def missing_parser(): pass")

    assert result.language == "python"
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


def _text_node(
    source: bytes,
    type_name: str,
    needle: bytes,
    start: int = 0,
) -> FakeNode:
    offset = source.index(needle, start) if start >= 0 else source.rindex(needle)
    return FakeNode(type_name, start_byte=offset, end_byte=offset + len(needle))


def test_docstring_covers_empty_comment_and_non_string_bodies() -> None:
    empty_string = FakeNode("string", start_byte=0, end_byte=2)
    empty_expr = FakeNode("expression_statement", children=[empty_string])
    assert (
        python_parser._docstring(FakeNode("block", children=[empty_expr]), b"''") == ""
    )

    empty_expression = FakeNode("expression_statement")
    assert (
        python_parser._docstring(
            FakeNode("block", children=[empty_expression]),
            b"",
        )
        == ""
    )

    comment_only = FakeNode("block", children=[FakeNode("comment")])
    assert python_parser._docstring(comment_only, b"# comment") == ""

    non_comment_first = FakeNode("block", children=[FakeNode("pass_statement")])
    assert python_parser._docstring(non_comment_first, b"pass") == ""

    assert python_parser._docstring(None, b"") == ""


def test_private_helpers_cover_missing_names_calls_imports_and_blocks() -> None:
    source = b"fallback:\nmodule\n"
    symbols: list[ParsedSymbol] = []

    python_parser._walk_def(FakeNode("function_definition"), source, "", symbols)
    assert symbols == []

    function_definition = FakeNode(
        "function_definition",
        start_byte=0,
        end_byte=len(b"fallback:"),
        children=[
            FakeNode("identifier", start_byte=0, end_byte=len(b"fallback")),
            FakeNode(":", start_byte=len(b"fallback"), end_byte=len(b"fallback:")),
        ],
    )
    python_parser._walk_def(function_definition, source, "", symbols)

    class_without_body = FakeNode(
        "class_definition",
        start_byte=0,
        end_byte=len(b"fallback"),
        children=[
            FakeNode("identifier", start_byte=0, end_byte=len(b"fallback")),
            FakeNode(":", start_byte=len(b"fallback"), end_byte=len(b"fallback")),
        ],
    )
    python_parser._walk_def(class_without_body, source, "", symbols)

    assert [(symbol.kind, symbol.qualified, symbol.sig) for symbol in symbols] == [
        ("function", "fallback", "fallback:"),
        ("class", "fallback", "fallback"),
    ]

    calls: list[str] = []
    python_parser._collect_calls(
        FakeNode(
            "block",
            children=[
                FakeNode("call"),
                FakeNode(
                    "call",
                    children=[FakeNode("subscript", start_byte=0, end_byte=8)],
                ),
            ],
        ),
        source,
        calls,
    )
    assert calls == ["fallback"]

    imports = python_parser._imports(
        FakeNode(
            "module",
            children=[
                FakeNode("import_statement", children=[FakeNode("identifier")]),
                FakeNode("import_from_statement"),
                FakeNode(
                    "import_from_statement",
                    children=[FakeNode("relative_import", start_byte=10, end_byte=16)],
                ),
            ],
        ),
        source,
    )
    assert imports == ["module"]

    walked: list[ParsedSymbol] = []
    python_parser._walk(
        FakeNode(
            "module",
            children=[
                FakeNode("decorated_definition"),
                FakeNode(
                    "block",
                    children=[
                        FakeNode(
                            "function_definition",
                            start_byte=0,
                            end_byte=len(b"fallback"),
                            children=[
                                FakeNode("identifier", start_byte=0, end_byte=8),
                                FakeNode(":", start_byte=8, end_byte=8),
                            ],
                        )
                    ],
                ),
            ],
        ),
        source,
        "",
        walked,
    )
    assert [symbol.qualified for symbol in walked] == ["fallback"]
