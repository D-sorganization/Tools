"""Tree-sitter parsers for Python, Rust, TypeScript, JavaScript, Markdown.

Thin dispatcher over the per-language modules in ``_lang_*``. Each language
exposes ``extract_*(path, source) -> ParseResult`` and shares a small set
of helpers from ``_ts_common`` (lazy import + parser cache + node helpers).

``dispatch(path, source)`` picks the right extractor by extension; missing
language packages are silently skipped after a one-shot warning so a bare
interpreter without every tree-sitter wheel won't crash the indexer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import _lang_js as _js
from . import _lang_markdown as _md
from . import _lang_python as _py
from . import _lang_rust as _rs
from ._ts_common import ParsedSymbol, ParseResult, to_bytes

# Re-export ParseResult / ParsedSymbol so callers can keep importing from
# ``codemap.parsers`` without knowing about the internal split.
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

# Per-language entry points (preserve the names the test suite imports).
extract_python = _py.extract
extract_javascript = _js.extract_javascript
extract_typescript = _js.extract_typescript
extract_rust = _rs.extract
extract_markdown = _md.extract


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
    if lang == "typescript":
        resource = _qt_translation(source)
        if resource is not None:
            return resource
    extractor = _EXTRACTORS.get(lang)
    if extractor is None:
        return None
    return extractor(str(path), source)


def _qt_translation(source: str | bytes) -> ParseResult | None:
    """Recognize Qt's XML .ts resources without hiding malformed source files."""
    data = to_bytes(source).lstrip(b"\xef\xbb\xbf \t\r\n")
    if not data.startswith((b"<?xml", b"<!DOCTYPE TS", b"<TS")):
        return None
    from defusedxml import ElementTree
    from defusedxml.common import DefusedXmlException

    try:
        root = ElementTree.fromstring(data)
    except (ElementTree.ParseError, DefusedXmlException):
        return ParseResult("qt-translation", [], [], complete=False)
    if root.tag != "TS":
        return None
    return ParseResult("qt-translation", [], [])
