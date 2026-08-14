"""Cross-runtime grammar for substantive unavailable-calculation reasons."""

from __future__ import annotations

import re

_SENTINELS = frozenset(
    {"x", "na", "none", "nodata", "notavailable", "notapplicable", "unavailable"}
)
_UNICODE_WHITESPACE = frozenset(
    "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_NORMALIZATION_SEPARATORS = _UNICODE_WHITESPACE | frozenset("./_-")
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_WORD = re.compile(r"[A-Za-z]{2,}")
_MIN_LENGTH = 16
_MAX_LENGTH = 500


def validate_reason_grammar(reason: str) -> str:
    """Validate exact shared whitespace, length, word, and sentinel rules."""
    if reason[0] in _UNICODE_WHITESPACE or reason[-1] in _UNICODE_WHITESPACE:
        raise ValueError("reason must not contain surrounding whitespace")
    normalized = "".join(
        character
        for character in reason.translate(_ASCII_LOWER)
        if character not in _NORMALIZATION_SEPARATORS
    ).rstrip("!?")
    if normalized in _SENTINELS:
        raise ValueError("reason must not be a sentinel value")
    if not _MIN_LENGTH <= len(reason) <= _MAX_LENGTH:
        raise ValueError("reason must contain 16 to 500 Unicode scalar values")
    if len(_WORD.findall(reason)) < 3:
        raise ValueError("reason must contain at least three explanatory words")
    return reason


__all__ = ["validate_reason_grammar"]
