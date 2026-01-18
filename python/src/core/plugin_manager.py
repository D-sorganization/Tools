import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    path: str
    type: str
    desc: str
    category: str


class PluginManager:
    """Manages tool discovery and loading."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.tools_file = repo_root / "tools.json"
        self.tools: dict[str, list[Tool]] = {}

    def load_tools(self) -> dict[str, list[Tool]]:
        """Load tools from tools.json."""
        if not self.tools_file.exists():
            logger.error(f"Tools file not found: {self.tools_file}")
            return {}

        try:
            with open(self.tools_file, encoding="utf-8") as f:
                data = json.load(f)

            self.tools = {}
            for category, items in data.items():
                tool_list = []
                for item in items:
                    try:
                        tool = Tool(
                            name=item["name"],
                            path=item["path"],
                            type=item["type"],
                            desc=item["desc"],
                            category=category,
                        )
                        tool_list.append(tool)
                    except KeyError as e:
                        logger.warning(
                            f"Skipping invalid tool entry in {category}: {e}"
                        )

                if tool_list:
                    self.tools[category] = tool_list

            return self.tools
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
            return {}

    def get_tool_by_name(self, name: str) -> Tool | None:
        """Find a tool by name."""
        for category in self.tools.values():
            for tool in category:
                if tool.name == name:
                    return tool
        return None
