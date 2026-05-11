"""Tree-sitter parser wrappers for Python, Rust, TypeScript, JavaScript, Markdown.

Each language exposes ``extract_symbols(path, source) -> list[Symbol]``.
``dispatch(path, source)`` picks the right one by extension.

Lazy-import: a missing language package logs a warning once and that
extension is silently skipped on subsequent files (rather than blowing up
the whole indexer).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public Symbol record (intermediate; persisted via indexer / api.Symbol).
# ---------------------------------------------------------------------------


@dataclass
class ParsedSymbol:
    """Lightweight symbol record produced by parsers."""

    kind: str
    name: str
    qualified: str
    sig: str
    docstring: str
    start_line: int  # 1-indexed inclusive
    end_line: int  # 1-indexed inclusive
    calls_out: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    """All symbols + imports extracted from a single file."""

    language: str
    imports: list[str]
    symbols: list[ParsedSymbol]


# ---------------------------------------------------------------------------
# Lazy language registry.
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_LANG_CACHE: dict[str, object] = {}
_PARSER_CACHE: dict[str, object] = {}
_MISSING_WARNED: set[str] = set()

_LANG_MODULES: dict[str, tuple[str, str]] = {
    # language_id -> (pip module name, attribute name returning ts language ptr)
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "rust": ("tree_sitter_rust", "language"),
    "markdown": ("tree_sitter_markdown", "language"),
}

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".md": "markdown",
    ".markdown": "markdown",
}


def _get_parser(lang_id: str):
    """Return a cached tree-sitter Parser for ``lang_id`` or None if unavailable."""
    with _LOCK:
        if lang_id in _PARSER_CACHE:
            return _PARSER_CACHE[lang_id]
        spec = _LANG_MODULES.get(lang_id)
        if spec is None:
            return None
        module_name, attr = spec
        try:
            mod = __import__(module_name)
            from tree_sitter import Language, Parser  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import paths exercised at runtime
            if module_name not in _MISSING_WARNED:
                _MISSING_WARNED.add(module_name)
                logger.warning(
                    "codemap: language '%s' unavailable (%s); skipping",
                    lang_id,
                    exc,
                )
            _PARSER_CACHE[lang_id] = None  # type: ignore[assignment]
            return None
        try:
            raw = getattr(mod, attr)()
            language = Language(raw)
            parser = Parser(language)
        except Exception as exc:  # pragma: no cover
            if module_name not in _MISSING_WARNED:
                _MISSING_WARNED.add(module_name)
                logger.warning(
                    "codemap: failed to initialise '%s' parser: %s",
                    lang_id,
                    exc,
                )
            _PARSER_CACHE[lang_id] = None  # type: ignore[assignment]
            return None
        _LANG_CACHE[lang_id] = language
        _PARSER_CACHE[lang_id] = parser
        return parser


def _bytes(source: str | bytes) -> bytes:
    if isinstance(source, bytes):
        return source
    return source.encode("utf-8", errors="replace")


def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _first_child(node, type_name: str):
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def _line_of(node) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


# ---------------------------------------------------------------------------
# Python.
# ---------------------------------------------------------------------------


def _py_docstring(body_node, source: bytes) -> str:
    if body_node is None:
        return ""
    for c in body_node.children:
        if c.type == "expression_statement":
            inner = c.children[0] if c.children else None
            if inner is not None and inner.type == "string":
                raw = _text(inner, source).strip()
                # Strip quotes (handle triple and single).
                for q in ('"""', "'''", '"', "'"):
                    if raw.startswith(q) and raw.endswith(q):
                        raw = raw[len(q) : -len(q)]
                        break
                return raw.strip().splitlines()[0] if raw.strip() else ""
            return ""
        if c.type not in ("comment",):
            return ""
    return ""


def _py_signature(def_node, source: bytes) -> str:
    # First line of the def — name + params.
    end = def_node.start_byte
    for c in def_node.children:
        if c.type == ":":
            end = c.end_byte
            break
    sig = source[def_node.start_byte : end].decode("utf-8", errors="replace")
    return sig.splitlines()[0].strip()


def _py_calls(node, source: bytes, out: list[str]) -> None:
    if node.type == "call":
        func = _first_child(node, "attribute") or _first_child(node, "identifier")
        if func is None and node.children:
            func = node.children[0]
        if func is not None:
            out.append(_text(func, source))
    for c in node.children:
        _py_calls(c, source, out)


def _py_walk(node, source: bytes, prefix: str, out: list[ParsedSymbol]) -> None:
    for child in node.children:
        if child.type == "decorated_definition":
            inner = child.children[-1] if child.children else None
            if inner is not None:
                _py_walk_def(inner, source, prefix, out)
        elif child.type in ("function_definition", "class_definition"):
            _py_walk_def(child, source, prefix, out)
        elif child.type == "block":
            _py_walk(child, source, prefix, out)


def _py_walk_def(node, source: bytes, prefix: str, out: list[ParsedSymbol]) -> None:
    name_node = _first_child(node, "identifier")
    if name_node is None:
        return
    name = _text(name_node, source)
    qualified = f"{prefix}.{name}" if prefix else name
    body = _first_child(node, "block")
    start, end = _line_of(node)
    if node.type == "function_definition":
        calls: list[str] = []
        if body is not None:
            _py_calls(body, source, calls)
        out.append(
            ParsedSymbol(
                kind="method" if "." in qualified else "function",
                name=name,
                qualified=qualified,
                sig=_py_signature(node, source),
                docstring=_py_docstring(body, source),
                start_line=start,
                end_line=end,
                calls_out=sorted(set(calls))[:64],
            )
        )
    elif node.type == "class_definition":
        out.append(
            ParsedSymbol(
                kind="class",
                name=name,
                qualified=qualified,
                sig=_py_signature(node, source),
                docstring=_py_docstring(body, source),
                start_line=start,
                end_line=end,
            )
        )
        if body is not None:
            _py_walk(body, source, qualified, out)


def _py_imports(root, source: bytes) -> list[str]:
    out: list[str] = []
    for c in root.children:
        if c.type == "import_statement":
            for sub in c.children:
                if sub.type == "dotted_name":
                    out.append(_text(sub, source))
        elif c.type == "import_from_statement":
            mod = _first_child(c, "dotted_name") or _first_child(c, "relative_import")
            if mod is not None:
                out.append(_text(mod, source))
    return out


def extract_python(path: str, source: str | bytes) -> ParseResult:
    parser = _get_parser("python")
    if parser is None:
        return ParseResult("python", [], [])
    src = _bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _py_walk(tree.root_node, src, "", symbols)
    imports = _py_imports(tree.root_node, src)
    return ParseResult("python", imports, symbols)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript.
# ---------------------------------------------------------------------------


def _js_walk(node, source: bytes, prefix: str, out: list[ParsedSymbol]) -> None:
    for child in node.children:
        t = child.type
        if t == "function_declaration":
            name_node = _first_child(child, "identifier")
            if name_node is None:
                continue
            name = _text(name_node, source)
            qualified = f"{prefix}.{name}" if prefix else name
            start, end = _line_of(child)
            out.append(
                ParsedSymbol(
                    kind="function",
                    name=name,
                    qualified=qualified,
                    sig=_text(child, source).splitlines()[0],
                    docstring="",
                    start_line=start,
                    end_line=end,
                )
            )
        elif t in ("class_declaration", "abstract_class_declaration"):
            name_node = _first_child(child, "type_identifier") or _first_child(
                child, "identifier"
            )
            if name_node is None:
                continue
            name = _text(name_node, source)
            qualified = f"{prefix}.{name}" if prefix else name
            start, end = _line_of(child)
            out.append(
                ParsedSymbol(
                    kind="class",
                    name=name,
                    qualified=qualified,
                    sig=f"class {name}",
                    docstring="",
                    start_line=start,
                    end_line=end,
                )
            )
            body = _first_child(child, "class_body")
            if body is not None:
                for member in body.children:
                    if member.type == "method_definition":
                        mname_node = _first_child(
                            member, "property_identifier"
                        ) or _first_child(member, "identifier")
                        if mname_node is None:
                            continue
                        mname = _text(mname_node, source)
                        mstart, mend = _line_of(member)
                        out.append(
                            ParsedSymbol(
                                kind="method",
                                name=mname,
                                qualified=f"{qualified}.{mname}",
                                sig=_text(member, source).splitlines()[0],
                                docstring="",
                                start_line=mstart,
                                end_line=mend,
                            )
                        )
        elif t in ("export_statement", "ambient_declaration"):
            _js_walk(child, source, prefix, out)
        elif t in ("lexical_declaration", "variable_declaration"):
            for sub in child.children:
                if sub.type == "variable_declarator":
                    name_node = _first_child(sub, "identifier")
                    val = sub.children[-1] if sub.children else None
                    if name_node is None or val is None:
                        continue
                    if val.type in (
                        "arrow_function",
                        "function_expression",
                        "function",
                    ):
                        name = _text(name_node, source)
                        start, end = _line_of(child)
                        out.append(
                            ParsedSymbol(
                                kind="function",
                                name=name,
                                qualified=f"{prefix}.{name}" if prefix else name,
                                sig=_text(child, source).splitlines()[0],
                                docstring="",
                                start_line=start,
                                end_line=end,
                            )
                        )


def _js_imports(root, source: bytes) -> list[str]:
    out: list[str] = []
    for c in root.children:
        if c.type == "import_statement":
            src_node = _first_child(c, "string")
            if src_node is not None:
                out.append(_text(src_node, source).strip("'\""))
    return out


def extract_javascript(path: str, source: str | bytes) -> ParseResult:
    parser = _get_parser("javascript")
    if parser is None:
        return ParseResult("javascript", [], [])
    src = _bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _js_walk(tree.root_node, src, "", symbols)
    return ParseResult("javascript", _js_imports(tree.root_node, src), symbols)


def extract_typescript(path: str, source: str | bytes) -> ParseResult:
    lang_id = "tsx" if str(path).endswith(".tsx") else "typescript"
    parser = _get_parser(lang_id)
    if parser is None:
        return ParseResult(lang_id, [], [])
    src = _bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _js_walk(tree.root_node, src, "", symbols)
    return ParseResult(lang_id, _js_imports(tree.root_node, src), symbols)


# ---------------------------------------------------------------------------
# Rust.
# ---------------------------------------------------------------------------


def _rs_walk(node, source: bytes, prefix: str, out: list[ParsedSymbol]) -> None:
    for child in node.children:
        t = child.type
        if t == "function_item":
            name_node = _first_child(child, "identifier")
            if name_node is None:
                continue
            name = _text(name_node, source)
            qualified = f"{prefix}::{name}" if prefix else name
            start, end = _line_of(child)
            out.append(
                ParsedSymbol(
                    kind="function",
                    name=name,
                    qualified=qualified,
                    sig=_text(child, source).splitlines()[0].rstrip(" {"),
                    docstring="",
                    start_line=start,
                    end_line=end,
                )
            )
        elif t == "struct_item":
            name_node = _first_child(child, "type_identifier")
            if name_node is None:
                continue
            name = _text(name_node, source)
            qualified = f"{prefix}::{name}" if prefix else name
            start, end = _line_of(child)
            out.append(
                ParsedSymbol(
                    kind="struct",
                    name=name,
                    qualified=qualified,
                    sig=f"struct {name}",
                    docstring="",
                    start_line=start,
                    end_line=end,
                )
            )
        elif t == "impl_item":
            type_node = _first_child(child, "type_identifier")
            sub_prefix = prefix
            if type_node is not None:
                sub_prefix = (
                    f"{prefix}::{_text(type_node, source)}"
                    if prefix
                    else _text(type_node, source)
                )
            body = _first_child(child, "declaration_list")
            if body is not None:
                _rs_walk(body, source, sub_prefix, out)
        elif t == "mod_item":
            name_node = _first_child(child, "identifier")
            if name_node is None:
                continue
            name = _text(name_node, source)
            sub_prefix = f"{prefix}::{name}" if prefix else name
            body = _first_child(child, "declaration_list")
            if body is not None:
                _rs_walk(body, source, sub_prefix, out)


def _rs_imports(root, source: bytes) -> list[str]:
    out: list[str] = []
    for c in root.children:
        if c.type == "use_declaration":
            out.append(_text(c, source).strip().rstrip(";"))
    return out


def extract_rust(path: str, source: str | bytes) -> ParseResult:
    parser = _get_parser("rust")
    if parser is None:
        return ParseResult("rust", [], [])
    src = _bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _rs_walk(tree.root_node, src, "", symbols)
    return ParseResult("rust", _rs_imports(tree.root_node, src), symbols)


# ---------------------------------------------------------------------------
# Markdown — extract headings as "symbols".
# ---------------------------------------------------------------------------


def _md_walk(node, source: bytes, out: list[ParsedSymbol]) -> None:
    for c in node.children:
        if c.type in ("atx_heading", "setext_heading"):
            text_node = _first_child(c, "inline") or _first_child(c, "heading_content")
            if text_node is None:
                # Fall back: take the whole heading text.
                title = _text(c, source).strip("# \n")
            else:
                title = _text(text_node, source).strip()
            if not title:
                continue
            start, end = _line_of(c)
            out.append(
                ParsedSymbol(
                    kind="heading",
                    name=title[:80],
                    qualified=title[:200],
                    sig=title[:200],
                    docstring="",
                    start_line=start,
                    end_line=end,
                )
            )
        _md_walk(c, source, out)


def extract_markdown(path: str, source: str | bytes) -> ParseResult:
    parser = _get_parser("markdown")
    if parser is None:
        return ParseResult("markdown", [], [])
    src = _bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _md_walk(tree.root_node, src, symbols)
    return ParseResult("markdown", [], symbols)


# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------


_EXTRACTORS: dict[str, Callable[[str, str | bytes], ParseResult]] = {
    "python": extract_python,
    "javascript": extract_javascript,
    "typescript": extract_typescript,
    "tsx": extract_typescript,
    "rust": extract_rust,
    "markdown": extract_markdown,
}


def language_for(path: str | Path) -> str | None:
    """Return the language id for ``path``, or None if unsupported."""
    ext = Path(path).suffix.lower()
    return EXT_TO_LANG.get(ext)


def dispatch(path: str | Path, source: str | bytes) -> ParseResult | None:
    """Parse ``source`` by extension. Returns None for unsupported languages."""
    lang = language_for(path)
    if lang is None:
        return None
    extractor = _EXTRACTORS.get(lang)
    if extractor is None:
        return None
    return extractor(str(path), source)


__all__ = [
    "EXT_TO_LANG",
    "ParseResult",
    "ParsedSymbol",
    "dispatch",
    "extract_javascript",
    "extract_markdown",
    "extract_python",
    "extract_rust",
    "extract_typescript",
    "language_for",
]
