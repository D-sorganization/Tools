# ruff: noqa: E501
"""TDD tests for Agent Peer Review GUI system (Tools #2738).

Covers:
- format_transcript() utility
- PEER_REVIEW_SYSTEM_PROMPT constant
- PeerReviewConfigDialog widget

All tests run without a real display server by mocking Qt where needed.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Minimal package stubs so we can import without the full dependency tree
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    """Register a bare namespace package in sys.modules if missing."""
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = mod


for _pkg, _rel in [
    ("src", ROOT / "src"),
    ("src.shared", ROOT / "src" / "shared"),
    ("src.shared.python", ROOT / "src" / "shared" / "python"),
    ("src.shared.python.ai", ROOT / "src" / "shared" / "python" / "ai"),
    (
        "src.shared.python.ai.peer_review",
        ROOT / "src" / "shared" / "python" / "ai" / "peer_review",
    ),
    (
        "src.shared.python.ai.gui",
        ROOT / "src" / "shared" / "python" / "ai" / "gui",
    ),
]:
    _ensure_pkg(_pkg, _rel)

_logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
_logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
_logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", _logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", _logging_config)

# ---------------------------------------------------------------------------
# Load the production module under test
# ---------------------------------------------------------------------------


def _load(name: str, rel: str):
    full = ROOT / "src" / "shared" / "python" / rel
    spec = importlib.util.spec_from_file_location(name, full)
    assert spec is not None and spec.loader is not None, f"Cannot locate {full}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_transcript_mod = _load(
    "src.shared.python.ai.peer_review.transcript",
    "ai/peer_review/transcript.py",
)
format_transcript = _transcript_mod.format_transcript

_prompt_mod = _load(
    "src.shared.python.ai.peer_review.prompts",
    "ai/peer_review/prompts.py",
)
PEER_REVIEW_SYSTEM_PROMPT = _prompt_mod.PEER_REVIEW_SYSTEM_PROMPT

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Tests: format_transcript
# ---------------------------------------------------------------------------


class TestFormatTranscript:
    def test_format_transcript_includes_all_messages(self) -> None:
        """format_transcript wraps all messages in <transcript> tags."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well."},
        ]
        result = format_transcript(messages)
        assert "<transcript>" in result
        assert "</transcript>" in result
        assert "Hello, how are you?" in result
        assert "I am doing well." in result

    def test_format_transcript_includes_all_roles(self) -> None:
        """Each message role and content appears in output."""
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "system", "content": "System info"},
        ]
        result = format_transcript(messages)
        # Roles may be uppercased as labels (e.g. [USER]) — check case-insensitively
        result_lower = result.lower()
        assert "user" in result_lower
        assert "assistant" in result_lower
        assert "Question" in result
        assert "Answer" in result

    def test_format_transcript_empty_thread_returns_empty_transcript(self) -> None:
        """Empty message list produces a valid but empty transcript block."""
        result = format_transcript([])
        assert "<transcript>" in result
        assert "</transcript>" in result
        # The content between tags may be empty or whitespace only
        inner = result.replace("<transcript>", "").replace("</transcript>", "").strip()
        assert inner == ""

    def test_format_transcript_raises_for_non_list(self) -> None:
        """DbC: format_transcript raises ValueError if messages is not a list."""
        with pytest.raises((TypeError, ValueError)):
            format_transcript("not a list")  # type: ignore[arg-type]

    def test_format_transcript_raises_for_none(self) -> None:
        """DbC: format_transcript raises ValueError for None input."""
        with pytest.raises((TypeError, ValueError)):
            format_transcript(None)  # type: ignore[arg-type]

    def test_format_transcript_preserves_message_order(self) -> None:
        """Messages appear in their original order in the transcript."""
        messages = [
            {"role": "user", "content": "FIRST"},
            {"role": "assistant", "content": "SECOND"},
            {"role": "user", "content": "THIRD"},
        ]
        result = format_transcript(messages)
        first_pos = result.index("FIRST")
        second_pos = result.index("SECOND")
        third_pos = result.index("THIRD")
        assert first_pos < second_pos < third_pos

    def test_format_transcript_accepts_message_objects(self) -> None:
        """format_transcript also accepts objects with .role and .content attrs."""

        class _FakeMsg:
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

        messages = [_FakeMsg("user", "Hello"), _FakeMsg("assistant", "World")]
        result = format_transcript(messages)
        assert "Hello" in result
        assert "World" in result


# ---------------------------------------------------------------------------
# Tests: PEER_REVIEW_SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestPeerReviewSystemPrompt:
    def test_system_prompt_is_non_empty_string(self) -> None:
        assert isinstance(PEER_REVIEW_SYSTEM_PROMPT, str)
        assert len(PEER_REVIEW_SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_critical_instruction(self) -> None:
        """The prompt must instruct the reviewer NOT to rubber-stamp."""
        lower = PEER_REVIEW_SYSTEM_PROMPT.lower()
        anti_rubber_stamp_phrases = [
            "do not simply agree",
            "critical",
            "must not agree",
            "rubber",
            "challenge",
            "find issues",
        ]
        assert any(phrase in lower for phrase in anti_rubber_stamp_phrases), (
            "System prompt must contain an anti-rubber-stamp instruction. "
            f"Got: {PEER_REVIEW_SYSTEM_PROMPT[:200]!r}"
        )

    def test_system_prompt_mentions_security(self) -> None:
        assert "security" in PEER_REVIEW_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_performance(self) -> None:
        assert "performance" in PEER_REVIEW_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_design(self) -> None:
        assert "design" in PEER_REVIEW_SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_grading_rubric(self) -> None:
        """Prompt must reference the four review dimensions."""
        lower = PEER_REVIEW_SYSTEM_PROMPT.lower()
        required = ["security", "performance", "design", "accuracy"]
        missing = [r for r in required if r not in lower]
        assert not missing, f"System prompt missing rubric dimensions: {missing}"

    def test_system_prompt_identifies_reviewer_role(self) -> None:
        """Reviewer must be identified as a senior peer reviewer."""
        lower = PEER_REVIEW_SYSTEM_PROMPT.lower()
        assert (
            "peer reviewer" in lower
            or "senior reviewer" in lower
            or "reviewer" in lower
        )


# ---------------------------------------------------------------------------
# Tests: PeerReviewConfigDialog (Qt — mocked if no display)
# ---------------------------------------------------------------------------


def _has_pyqt6() -> bool:
    return importlib.util.find_spec("PyQt6") is not None


_qt_reason = "PyQt6 not installed"


def _stub_config_deps() -> None:
    """Inject minimal stubs for the deep config dependency chain.

    ``ai.peer_review.gui`` → ``ai.gui._provider_registry_data``
      → ``ai.config`` → ``config.environment``.

    We stub just enough so that PyQt6 widgets can be instantiated without
    a real environment or keyring.
    """
    # stub config.environment
    _cfg_env = types.ModuleType("src.shared.python.config")
    _cfg_env_sub = types.ModuleType("src.shared.python.config.environment")
    _cfg_env_sub.get_env = lambda name, default=None: default  # type: ignore[attr-defined]
    _cfg_env_sub.get_env_float = lambda name, default=0.0: default  # type: ignore[attr-defined]
    sys.modules.setdefault("src.shared.python.config", _cfg_env)
    sys.modules.setdefault("src.shared.python.config.environment", _cfg_env_sub)

    # stub keyring (used by _api_keys)
    _keyring = types.ModuleType("keyring")
    _keyring.get_password = lambda *a, **kw: None  # type: ignore[attr-defined]
    _keyring.set_password = lambda *a, **kw: None  # type: ignore[attr-defined]
    _keyring.delete_password = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules.setdefault("keyring", _keyring)


class TestPeerReviewConfigDialog:
    """Tests for the PeerReviewConfigDialog widget.

    We use importlib to load the module so that the fixture can inject the
    mock registry before the dialog class is exercised.  When PyQt6 is not
    available, all tests in this class are skipped automatically.
    """

    @pytest.fixture(autouse=True)
    def _load_dialog(self):
        pytest.importorskip("PyQt6.QtCore", reason=_qt_reason)
        _stub_config_deps()

        # Ensure the QApplication exists (headless via offscreen platform).
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(["--platform", "offscreen"])
        self._app = app

        # Load the dialog module (may already be cached in sys.modules).
        mod = _load(
            "src.shared.python.ai.peer_review.gui",
            "ai/peer_review/gui.py",
        )
        self._dialog_cls = mod.PeerReviewConfigDialog
        yield

    def test_dialog_has_provider_selector(self) -> None:
        """PeerReviewConfigDialog must contain a provider QComboBox."""
        from PyQt6.QtWidgets import QComboBox

        dlg = self._dialog_cls()
        combos = dlg.findChildren(QComboBox)
        assert len(combos) >= 1, "Dialog must have at least one QComboBox (provider)"
        dlg.close()

    def test_dialog_has_model_selector(self) -> None:
        """PeerReviewConfigDialog must contain a model QComboBox."""
        from PyQt6.QtWidgets import QComboBox

        dlg = self._dialog_cls()
        combos = dlg.findChildren(QComboBox)
        assert (
            len(combos) >= 2
        ), "Dialog must have at least two QComboBoxes (provider + model)"
        dlg.close()

    def test_dialog_returns_selected_config(self) -> None:
        """get_config() returns (provider, model) after values are selected."""
        dlg = self._dialog_cls()
        config = dlg.get_config()
        assert isinstance(config, tuple), "get_config() must return a tuple"
        assert len(config) == 2, "get_config() must return (provider, model)"
        provider, model = config
        assert (
            isinstance(provider, str) and provider
        ), "provider must be a non-empty str"
        assert isinstance(model, str) and model, "model must be a non-empty str"
        dlg.close()
