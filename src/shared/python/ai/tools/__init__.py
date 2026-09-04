"""AI Tools module for agent capabilities."""

from shared.python.ai.tools.agent_control import (
    AgentActionResult,
    AgentController,
    EngineStatus,
    create_agent_tools_for_registry,
)
from shared.python.ai.tools.cli_tools import (
    ClaudeCodeTool,
    CLIExecutionResult,
    CLIToolConfig,
    CLIToolManager,
    CodexCLITool,
    ShellTool,
    create_cli_tools_for_registry,
)
from shared.python.ai.tools.codemap_tools import (
    CODEMAP_TOOL_NAMES,
    register_codemap_tools,
)
from shared.python.ai.tools.sidekick_analytics import (
    SIDEKICK_ANALYTICS_TOOL_NAME,
    register_sidekick_analytics_tools,
    summarize_simulation_run,
)

__all__ = [
    # CLI Tools
    "ClaudeCodeTool",
    "CodexCLITool",
    "ShellTool",
    "CLIToolManager",
    "CLIToolConfig",
    "CLIExecutionResult",
    "create_cli_tools_for_registry",
    # Agent Control
    "AgentController",
    "AgentActionResult",
    "EngineStatus",
    "create_agent_tools_for_registry",
    # Codemap
    "CODEMAP_TOOL_NAMES",
    "register_codemap_tools",
    # Sidekick analytics
    "SIDEKICK_ANALYTICS_TOOL_NAME",
    "register_sidekick_analytics_tools",
    "summarize_simulation_run",
]
