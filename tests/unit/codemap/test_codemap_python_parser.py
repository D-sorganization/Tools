from __future__ import annotations

from dataclasses import dataclass, field

from codemap import _lang_python as python_parser
from codemap._ts_common import ParsedSymbol


def test_extract_emits_imports_symbols_docstrings_signatures_and_calls() -> None:
    source = (
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

    result = python_parser.extract("sample.py", source.encode())

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
