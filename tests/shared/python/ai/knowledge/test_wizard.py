"""Sidekick Wizards: per-product knowledge injected into every chat turn (#5346)."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.python.ai.adapters.ollama_adapter import OllamaAdapter
from shared.python.ai.gui._panel_tools import register_panel_tools
from shared.python.ai.knowledge import build_pack, load_manifest
from shared.python.ai.knowledge.wizard import (
    STALE_BANNER,
    KnowledgeContext,
    WizardConfigError,
    WizardKnowledge,
    load_wizard_config,
)
from shared.python.ai.system_prompts import build_system_prompt
from shared.python.ai.types import ConversationContext
from shared.python.ai.wizards import knowledge_for_context, reset_wizards, wizard_for

from .conftest import commit_all, git_repo

WIZARD_YML = """\
key: fixture_product
name: Fixture Wizard
description: the expert on the Fixture Product's features and results
capabilities:
  - Explaining how the swing solver integrates the equations of motion
k: 3
"""

MANIFEST_YML = """\
id: fixture-product
title: Fixture Product
sources:
  - repo: FixtureProduct
    authority: product
    include: ['docs/**/*.md']
"""

SOLVER_DOC = """\
# Swing Solver

The swing solver integrates the equations of motion with a symplectic
Verlet scheme at 1 kHz.

# Export

Results export to CSV and JSON.
"""


@pytest.fixture(autouse=True)
def _fresh_wizard_cache() -> None:
    reset_wizards()


@pytest.fixture
def host(tmp_path: Path) -> Path:
    """A product checkout with ``knowledge/wizard.yml`` and a built pack."""
    root = git_repo(tmp_path / "FixtureProduct")
    (root / "docs").mkdir()
    (root / "docs" / "solver.md").write_text(SOLVER_DOC, "utf-8")
    (root / "knowledge").mkdir()
    (root / "knowledge" / "wizard.yml").write_text(WIZARD_YML, "utf-8")
    (root / "knowledge" / "pack.yml").write_text(MANIFEST_YML, "utf-8")
    commit_all(root)
    manifest = load_manifest(root / "knowledge" / "pack.yml")
    build_pack(manifest, {"FixtureProduct": root}, root / ".knowledge" / "pack.sqlite")
    return root


def _context(root: Path, question: str) -> ConversationContext:
    context = ConversationContext()
    context.metadata["project_root"] = str(root)
    context.add_message("user", question)
    return context


def test_config_reads_wizard_yml_with_defaults(host: Path) -> None:
    config = load_wizard_config(host)
    assert config is not None
    assert (config.key, config.name, config.k) == (
        "fixture_product",
        "Fixture Wizard",
        3,
    )
    assert config.pack == host / ".knowledge" / "pack.sqlite"
    assert config.manifest == host / "knowledge" / "pack.yml"
    assert config.roots == {"FixtureProduct": host}


def test_no_wizard_yml_means_no_wizard(tmp_path: Path) -> None:
    assert load_wizard_config(tmp_path) is None
    assert wizard_for(tmp_path) is None


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("name: X\n", "key"),
        ("key: Bad Key\nname: X\n", "key"),
        ("key: x\n", "name"),
        ("key: x\nname: X\nk: 0\n", "k"),
        ("key: x\nname: X\nsurprise: 1\n", "unknown"),
    ],
)
def test_invalid_wizard_yml_is_rejected(tmp_path: Path, text: str, needle: str) -> None:
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "wizard.yml").write_text(text, "utf-8")
    with pytest.raises(WizardConfigError, match=needle):
        load_wizard_config(tmp_path)


def test_context_carries_cited_passages(host: Path) -> None:
    wizard = WizardKnowledge(load_wizard_config(host))
    knowledge = wizard.context_for("how does the solver integrate the equations?")
    assert isinstance(knowledge, KnowledgeContext)
    assert knowledge.stale is False
    assert knowledge.passages[0].anchor == "swing-solver"
    rendered = knowledge.render()
    assert "Fixture Wizard" in rendered
    assert "[1] FixtureProduct:docs/solver.md#swing-solver @ " in rendered
    assert "Verlet" in rendered
    assert STALE_BANNER not in rendered


def test_stale_pack_renders_the_banner(host: Path) -> None:
    doc = host / "docs" / "solver.md"
    doc.write_text(SOLVER_DOC + "\nNow at 2 kHz.\n", "utf-8")
    knowledge = WizardKnowledge(load_wizard_config(host)).context_for("solver")
    assert knowledge is not None and knowledge.stale is True
    assert STALE_BANNER in knowledge.render()


def test_freshness_is_cached_between_turns(host: Path) -> None:
    now = [0.0]
    wizard = WizardKnowledge(load_wizard_config(host), clock=lambda: now[0])
    assert wizard.context_for("solver").stale is False
    (host / "docs" / "solver.md").write_text("# Changed\n\nNew.\n", "utf-8")
    assert wizard.context_for("solver").stale is False  # within the TTL
    now[0] += 10_000
    assert wizard.context_for("solver").stale is True


def test_missing_pack_degrades_to_no_knowledge(host: Path) -> None:
    (host / ".knowledge" / "pack.sqlite").unlink()
    wizard = WizardKnowledge(load_wizard_config(host))
    assert wizard.available is False
    assert wizard.context_for("solver") is None


def test_unrelated_question_injects_nothing_but_keeps_the_banner(host: Path) -> None:
    wizard = WizardKnowledge(load_wizard_config(host))
    assert wizard.context_for("zzzz qqqq") is None
    (host / "docs" / "solver.md").write_text("# Changed\n\nNew.\n", "utf-8")
    stale = WizardKnowledge(load_wizard_config(host)).context_for("zzzz qqqq")
    assert stale is not None and stale.passages == () and STALE_BANNER in stale.render()


def test_wizard_registers_its_app_context(host: Path) -> None:
    assert wizard_for(host) is not None
    preamble = build_system_prompt(app_context="fixture_product")
    assert "Fixture Wizard" in preamble
    assert "equations of motion" in preamble


def test_every_turn_gets_knowledge_through_the_adapter(host: Path) -> None:
    context = _context(host, "Which integration scheme does the solver use?")
    prompt = OllamaAdapter().build_system_prompt([], "beginner", context)
    assert "FixtureProduct:docs/solver.md#swing-solver" in prompt
    assert "Verlet" in prompt


def test_host_without_wizard_keeps_todays_prompt(tmp_path: Path) -> None:
    plain = ConversationContext()
    plain.add_message("user", "Which integration scheme does the solver use?")
    with_root = _context(tmp_path, "Which integration scheme does the solver use?")
    adapter = OllamaAdapter()
    assert adapter.build_system_prompt([], "beginner", with_root) == (
        adapter.build_system_prompt([], "beginner", plain)
    )
    assert knowledge_for_context(with_root) is None


def test_knowledge_uses_the_latest_user_message(host: Path) -> None:
    context = _context(host, "Which integration scheme does the solver use?")
    context.add_message("assistant", "Verlet.")
    context.add_message("user", "How do I export results to CSV?")
    knowledge = knowledge_for_context(context)
    assert knowledge is not None
    assert knowledge.passages[0].anchor == "export"


class _Registry:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def register(self, name: str, description: str, category: object):  # noqa: ANN201
        def decorator(fn):  # noqa: ANN001, ANN202
            self.tools[name] = fn
            return fn

        return decorator


class _EmptyRag:
    def query(self, _query: str) -> list[object]:
        return []


def test_search_tool_prefers_the_wizard_pack(host: Path) -> None:
    registry = _Registry()
    register_panel_tools(registry, _EmptyRag(), project_root=host)
    out = registry.tools["search_knowledge_base"]("Verlet scheme")
    assert "FixtureProduct:docs/solver.md#swing-solver" in out


def test_search_tool_falls_back_to_the_rag_store(tmp_path: Path) -> None:
    registry = _Registry()
    register_panel_tools(registry, _EmptyRag(), project_root=tmp_path)
    assert registry.tools["search_knowledge_base"]("Verlet") == (
        "No relevant information found."
    )


def test_multi_repo_roots_resolve_relative_to_the_host(tmp_path: Path) -> None:
    host = git_repo(tmp_path / "Host")
    (host / "knowledge").mkdir()
    (host / "knowledge" / "wizard.yml").write_text(
        "key: host\nname: Host Wizard\nroots:\n  Host: .\n  Docs: ../Docs\n", "utf-8"
    )
    config = load_wizard_config(host)
    assert config.roots == {"Host": host, "Docs": (host / ".." / "Docs").resolve()}
    commit_all(host)
