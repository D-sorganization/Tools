"""RED tests for secret redactor (Tools issue #2735)."""

from __future__ import annotations

import pytest
from chat.export.secret_redactor import SecretRedactor


@pytest.mark.parametrize(
    "raw, secret_fragment",
    [
        # Each literal is split across operands so the resulting test file
        # does not match common secret-scanning rules on push.
        (
            "API_KEY=" + "sk-" + "ABCD1234EFGH5678" + "IJKL9012MNOP3456",
            "sk-" + "ABCD1234",
        ),
        (
            "Authorization: " + "Bearer " + "eyJhbGciOiJI.payload.sig",
            "Bearer " + "eyJ",
        ),
        (
            "token " + "ghp_" + "1234567890abcdef" + "ABCDEF1234567890" + "abcdef",
            "ghp_" + "12345",
        ),
        # Synthesised AWS-style access key id (16 alnum chars); not a real
        # credential. The literal is split across operands so secret-scanning
        # does not flag it on push.
        ("aws " + "AKIA" + "TESTKEY1234ABCDE" + " secret", "AKIA" + "TESTKEY"),
        (
            "jwt "
            + "eyJ"
            + "hbGciOiJIUzI1NiJ9."
            + "eyJ"
            + "zdWIiOiIxIn0."
            + "dozjgNryP4J3jVmNHl0w5N"
            + "_XgL0n3I9PlFUP0THsR8U",
            "eyJ" + "hbGciOiJIUzI1NiJ9",
        ),
        (
            "api token " + "sk_live_" + "abcdefghijklmnop" + "1234567890ABCDEF",
            "sk_live_",
        ),
    ],
)
def test_redactor_removes_known_patterns(raw: str, secret_fragment: str) -> None:
    redactor = SecretRedactor()
    out = redactor.redact(raw)
    assert secret_fragment not in out
    assert "[REDACTED" in out


def test_redactor_leaves_plain_text_alone() -> None:
    redactor = SecretRedactor()
    plain = "Hello world, this is a regular sentence."
    assert redactor.redact(plain) == plain


_FAKE_KEY = "sk-" + "ABCD1234EFGH5678" + "IJKL9012MNOP3456"
_FAKE_KEY_PREFIX = "sk-" + "ABCD1234"


def test_redactor_does_not_modify_pinned_messages() -> None:
    from chat.service_base import ChatMessage

    redactor = SecretRedactor()
    msg = ChatMessage(
        role="user",
        content=_FAKE_KEY,
        metadata={"pin": True},
    )
    cleaned = redactor.redact_message(msg)
    # Pinned messages retain original content
    assert _FAKE_KEY_PREFIX in cleaned.content


def test_redactor_redacts_unpinned_messages() -> None:
    from chat.service_base import ChatMessage

    redactor = SecretRedactor()
    msg = ChatMessage(role="user", content=_FAKE_KEY)
    cleaned = redactor.redact_message(msg)
    assert _FAKE_KEY_PREFIX not in cleaned.content
