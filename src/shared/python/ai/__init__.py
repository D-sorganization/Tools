"""AI Assistant Integration Layer for Golf Modeling Suite.

This package provides an agent-agnostic AI assistant architecture that
enables natural language interaction, guided workflows, and educational
content delivery for users of all skill levels.

Architecture:
    AgentInterfaceProtocol (AIP) <- Provider Adapters <- User's LLM
                |
                v
    ToolRegistry + WorkflowEngine + EducationSystem
                |
                v
    Scientific Validator (enforces physics consistency)

Design Principles:
    1. Agent-Agnostic: Works with any LLM provider (OpenAI, Anthropic, Ollama)
    2. Zero Developer Cost: Users provide their own API keys
    3. Educational Focus: Teaches while executing
    4. Scientific Integrity: AI never bypasses validation
    5. Privacy-First: API keys in OS keyring, no data to developers

Example:
    >>> from shared.python.ai import ToolRegistry, WorkflowEngine, EducationSystem
    >>> registry = ToolRegistry()
    >>> engine = WorkflowEngine(registry)
    >>> edu = EducationSystem()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .education import EducationSystem, GlossaryEntry
    from .exceptions import (
        AIConnectionError,
        AIError,
        AIProviderError,
        AIRateLimitError,
        AITimeoutError,
        ScientificValidationError,
        ToolExecutionError,
        WorkflowError,
    )
    from .memory_manager import (
        MemoryCandidate,
        MemoryManager,
        build_memory_prompt_section,
        extract_memory_candidates,
        load_agents_md,
    )
    from .tool_registry import (
        Tool,
        ToolCategory,
        ToolParameter,
        ToolRegistry,
        get_global_registry,
    )
    from .types import (
        AgentChunk,
        AgentResponse,
        ConversationContext,
        ExpertiseLevel,
        Message,
        ProviderCapabilities,
        ProviderCapability,
        ToolCall,
        ToolResult,
    )
    from .workflow_definitions import (
        create_c3d_import_workflow,
        create_cross_engine_validation_workflow,
        create_drift_control_decomposition_workflow,
        create_first_analysis_workflow,
        create_inverse_dynamics_workflow,
    )
    from .workflow_engine import (  # noqa: I001
        RecoveryStrategy,
        StepResult,
        StepStatus,
        ValidationResult,
        Workflow,
        WorkflowEngine,
        WorkflowExecution,
        WorkflowStep,
    )


# Mapping of exported name -> (module, attribute) for lazy resolution.
_LAZY: dict[str, tuple[str, str]] = {
    # education
    "EducationSystem": (".education", "EducationSystem"),
    "GlossaryEntry": (".education", "GlossaryEntry"),
    # exceptions
    "AIConnectionError": (".exceptions", "AIConnectionError"),
    "AIError": (".exceptions", "AIError"),
    "AIProviderError": (".exceptions", "AIProviderError"),
    "AIRateLimitError": (".exceptions", "AIRateLimitError"),
    "AITimeoutError": (".exceptions", "AITimeoutError"),
    "ScientificValidationError": (".exceptions", "ScientificValidationError"),
    "ToolExecutionError": (".exceptions", "ToolExecutionError"),
    "WorkflowError": (".exceptions", "WorkflowError"),
    # memory_manager
    "MemoryCandidate": (".memory_manager", "MemoryCandidate"),
    "MemoryManager": (".memory_manager", "MemoryManager"),
    "build_memory_prompt_section": (".memory_manager", "build_memory_prompt_section"),
    "extract_memory_candidates": (".memory_manager", "extract_memory_candidates"),
    "load_agents_md": (".memory_manager", "load_agents_md"),
    # tool_registry
    "Tool": (".tool_registry", "Tool"),
    "ToolCategory": (".tool_registry", "ToolCategory"),
    "ToolParameter": (".tool_registry", "ToolParameter"),
    "ToolRegistry": (".tool_registry", "ToolRegistry"),
    "get_global_registry": (".tool_registry", "get_global_registry"),
    # types
    "AgentChunk": (".types", "AgentChunk"),
    "AgentResponse": (".types", "AgentResponse"),
    "ConversationContext": (".types", "ConversationContext"),
    "ExpertiseLevel": (".types", "ExpertiseLevel"),
    "Message": (".types", "Message"),
    "ProviderCapabilities": (".types", "ProviderCapabilities"),
    "ProviderCapability": (".types", "ProviderCapability"),
    "ToolCall": (".types", "ToolCall"),
    "ToolResult": (".types", "ToolResult"),
    # workflow_definitions
    "create_c3d_import_workflow": (
        ".workflow_definitions",
        "create_c3d_import_workflow",
    ),
    "create_cross_engine_validation_workflow": (
        ".workflow_definitions",
        "create_cross_engine_validation_workflow",
    ),
    "create_drift_control_decomposition_workflow": (
        ".workflow_definitions",
        "create_drift_control_decomposition_workflow",
    ),
    "create_first_analysis_workflow": (
        ".workflow_definitions",
        "create_first_analysis_workflow",
    ),
    "create_inverse_dynamics_workflow": (
        ".workflow_definitions",
        "create_inverse_dynamics_workflow",
    ),
    # workflow_engine
    "RecoveryStrategy": (".workflow_engine", "RecoveryStrategy"),
    "StepResult": (".workflow_engine", "StepResult"),
    "StepStatus": (".workflow_engine", "StepStatus"),
    "ValidationResult": (".workflow_engine", "ValidationResult"),
    "Workflow": (".workflow_engine", "Workflow"),
    "WorkflowEngine": (".workflow_engine", "WorkflowEngine"),
    "WorkflowExecution": (".workflow_engine", "WorkflowExecution"),
    "WorkflowStep": (".workflow_engine", "WorkflowStep"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution for heavy submodule symbols."""
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        import importlib

        mod = importlib.import_module(mod_path, package=__name__)
        value = getattr(mod, attr)
        # Cache in module globals for subsequent fast access.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Types
    "AgentChunk",
    "AgentResponse",
    "ConversationContext",
    "ExpertiseLevel",
    "Message",
    "ProviderCapabilities",
    "ProviderCapability",
    "ToolCall",
    "ToolResult",
    # Tool Registry
    "Tool",
    "ToolCategory",
    "ToolParameter",
    "ToolRegistry",
    "get_global_registry",
    # Workflow Engine
    "RecoveryStrategy",
    "StepResult",
    "StepStatus",
    "ValidationResult",
    "Workflow",
    "WorkflowEngine",
    "WorkflowExecution",
    "WorkflowStep",
    "create_c3d_import_workflow",
    "create_cross_engine_validation_workflow",
    "create_drift_control_decomposition_workflow",
    "create_first_analysis_workflow",
    "create_inverse_dynamics_workflow",
    # Education
    "EducationSystem",
    "GlossaryEntry",
    # Exceptions
    "AIError",
    "AIProviderError",
    "AIConnectionError",
    "AIRateLimitError",
    "AITimeoutError",
    "ScientificValidationError",
    "WorkflowError",
    "ToolExecutionError",
    # Memory
    "MemoryCandidate",
    "MemoryManager",
    "build_memory_prompt_section",
    "extract_memory_candidates",
    "load_agents_md",
]

__version__ = "0.1.0"
