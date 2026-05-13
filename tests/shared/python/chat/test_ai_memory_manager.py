"""Tests for shared AI chat memory persistence and prompt context."""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(ROOT / "src")]
sys.modules.setdefault("src", src_pkg)

ai_pkg = types.ModuleType("src.shared.python.ai")
ai_pkg.__path__ = [str(ROOT / "src" / "shared" / "python" / "ai")]
sys.modules.setdefault("src.shared.python.ai", ai_pkg)

adapters_pkg = types.ModuleType("src.shared.python.ai.adapters")
adapters_pkg.__path__ = [str(ROOT / "src" / "shared" / "python" / "ai" / "adapters")]
sys.modules.setdefault("src.shared.python.ai.adapters", adapters_pkg)

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger
logging_config.setup_logging = lambda *args, **kwargs: None
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

from src.shared.python.ai.adapters.base import BaseAgentAdapter
from src.shared.python.ai.memory_manager import (
    MemoryManager,
    build_memory_prompt_section,
    extract_memory_candidates,
    load_agents_md,
)
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
)

if TYPE_CHECKING:
    from src.shared.python.ai.adapters.base import ToolDeclaration


def test_memory_manager_persists_user_memory_json_atomically(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    manager.set_preference("tone", "concise")

    reloaded = MemoryManager(tmp_path)

    assert reloaded.memory["preferences"] == {"tone": "concise"}
    assert (tmp_path / "user_memory.json").is_file()


def test_archived_digest_extracts_only_explicit_user_memory(tmp_path: Path) -> None:
    context = ConversationContext(session_id="session_a")
    context.add_user_message("Please remember I prefer concise PR summaries.")
    context.add_assistant_message("Stored.")
    context.add_user_message("What is the current branch?")

    manager = MemoryManager(tmp_path)

    assert manager.digest_archived_contexts([context]) == 1
    assert manager.digest_archived_contexts([context]) == 0
    assert manager.memory["memories"][0]["content"] == (
        "Please remember I prefer concise PR summaries."
    )


def test_extract_memory_candidates_ignores_assistant_and_generic_chat() -> None:
    context = ConversationContext(session_id="session_b")
    context.add_user_message("Summarize the failing tests.")
    context.add_assistant_message("Remember to use pytest.")
    context.add_user_message("Never include secret values in logs.")

    candidates = extract_memory_candidates(context)

    assert [candidate.content for candidate in candidates] == [
        "Never include secret values in logs."
    ]


def test_prompt_section_includes_bounded_memory_and_agents_md(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("Always run focused tests.", encoding="utf-8")
    prompt_memory = {
        "preferences": {"verbosity": "short"},
        "memories": [{"content": "Prefer PRs over local-only changes."}],
    }

    section = build_memory_prompt_section(
        prompt_memory=prompt_memory,
        agents_md=load_agents_md(tmp_path),
    )

    assert "Always run focused tests." in section
    assert "verbosity: short" in section
    assert "Prefer PRs over local-only changes." in section


def test_base_adapter_uses_project_root_agents_and_prompt_memory(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("Use DbC for contracts.", encoding="utf-8")
    context = ConversationContext()
    context.metadata["project_root"] = str(tmp_path)
    context.metadata["prompt_memory"] = {
        "preferences": {"style": "direct"},
        "memories": [{"content": "Always keep work on PRs."}],
    }

    prompt = _Adapter().build_system_prompt([], "advanced", context)

    assert "Use DbC for contracts." in prompt
    assert "style: direct" in prompt
    assert "Always keep work on PRs." in prompt


class _Adapter(BaseAgentAdapter):
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        raise NotImplementedError

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        yield AgentChunk(content="")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supported=frozenset(),
            max_tokens=0,
            model_name="test",
        )

    def validate_connection(self) -> tuple[bool, str]:
        return True, "ok"
