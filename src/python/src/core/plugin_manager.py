import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..utils.file_utils import safe_read_json

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

    def validate_tool_path(self, tool_path: str) -> tuple[bool, str | None]:
        """
        Validate that a tool path exists and is within repository root.

        Args:
            tool_path: Relative path string from tool configuration

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(tool_path)
            # Resolve to absolute path relative to repo root
            full_path = (self.repo_root / path).resolve()
            repo_root_abs = self.repo_root.resolve()

            # Check path exists
            if not full_path.exists():
                return False, f"Tool file not found: {full_path}"

            # Check path is within repository (prevent path traversal)
            try:
                full_path.relative_to(repo_root_abs)
            except ValueError:
                return (
                    False,
                    f"Security Alert: Path outside repository: {full_path}",
                )

            # Check path is a file (not a directory)
            if not full_path.is_file():
                return False, f"Path is not a file: {full_path}"

            return True, None
        except (TypeError, ValueError, OSError, RuntimeError) as e:
            return False, f"Invalid path format: {tool_path} ({e})"

    def load_tools(self) -> dict[str, list[Tool]]:
        """
        Load tools from tools.json with path validation.

        Tools with invalid paths are logged but not included in the result.
        """
        if not self.tools_file.exists():
            logger.error(
                f"Tools file not found at {self.tools_file}. "
                f"Create a tools.json file in the repository root ({self.repo_root}) "
                "or verify that the installation is correct."
            )
            return {}

        # Use shared utility for safe JSON reading
        data = safe_read_json(self.tools_file, default={})
        if not data:
            return {}

        try:
            self.tools = {}
            for category, items in data.items():
                tool_list = []
                for item in items:
                    try:
                        tool_path = item["path"]
                        # Validate path exists and is within repository (issue #236)
                        is_valid, error_msg = self.validate_tool_path(tool_path)
                        if not is_valid:
                            logger.warning(
                                f"Skipping tool '{item.get('name', 'Unknown')}' "
                                f"in {category}: {error_msg}"
                            )
                            continue

                        tool = Tool(
                            name=item["name"],
                            path=tool_path,
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
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to load tools: {e}")
            return {}

    def get_tool_by_name(self, name: str) -> Tool | None:
        """Find a tool by name."""
        for category in self.tools.values():
            for tool in category:
                if tool.name == name:
                    return tool
        return None

    def scan_for_tools(self) -> dict[str, list[Tool]]:
        """
        Scan repository for tools with tool_manifest.json files.
        This provides automatic discovery without manual tools.json editing.

        Returns:
            Dictionary mapping categories to lists of discovered tools.
        """
        discovered_tools: dict[str, list[Tool]] = {}

        # Common tool directories to scan
        tool_dirs = [
            self.repo_root / "tools",
            self.repo_root / "web_applications",
            self.repo_root / "data_processing",
            self.repo_root / "scientific_modeling",
            self.repo_root / "media_processing",
        ]

        for tool_dir in tool_dirs:
            if not tool_dir.exists():
                continue

            # Recursively search for tool_manifest.json files
            for manifest_path in tool_dir.rglob("tool_manifest.json"):
                # Use shared utility for safe JSON reading
                manifest_data = safe_read_json(manifest_path, default={})
                if not manifest_data:
                    continue

                try:
                    # Extract tool information from manifest
                    tool_name = manifest_data.get("name", manifest_path.parent.name)
                    tool_path = manifest_data.get("path")
                    if not tool_path:
                        # Try to find main entry point
                        main_files = list(manifest_path.parent.glob("*.py"))
                        if main_files:
                            tool_path = str(main_files[0].relative_to(self.repo_root))
                        else:
                            continue
                    else:
                        # Make path relative to repo root
                        if not Path(tool_path).is_absolute():
                            tool_path = str(
                                (manifest_path.parent / tool_path).relative_to(
                                    self.repo_root
                                )
                            )

                    tool_type = manifest_data.get("type", "python")
                    tool_desc = manifest_data.get(
                        "description", manifest_data.get("desc", "")
                    )
                    tool_category = manifest_data.get("category", "Development Tools")

                    # Validate path exists and is within repository (issue #236)
                    is_valid, error_msg = self.validate_tool_path(tool_path)
                    if not is_valid:
                        logger.warning(
                            f"Skipping discovered tool '{tool_name}': {error_msg}"
                        )
                        continue

                    tool = Tool(
                        name=tool_name,
                        path=tool_path,
                        type=tool_type,
                        desc=tool_desc,
                        category=tool_category,
                    )

                    if tool_category not in discovered_tools:
                        discovered_tools[tool_category] = []
                    discovered_tools[tool_category].append(tool)

                    logger.info(
                        f"Discovered tool: {tool_name} in {manifest_path.parent}"
                    )

                except (json.JSONDecodeError, KeyError, Exception) as e:
                    logger.warning(f"Failed to parse manifest at {manifest_path}: {e}")
                    continue

        return discovered_tools

    def load_tools_with_discovery(self) -> dict[str, list[Tool]]:
        """
        Load tools from tools.json and merge with discovered tools from manifests.
        This provides backward compatibility while enabling automatic discovery.
        """
        # Load from tools.json (existing method)
        json_tools = self.load_tools()

        # Discover tools from manifests
        discovered_tools = self.scan_for_tools()

        # Merge results (discovered tools take precedence for duplicates)
        merged_tools = json_tools.copy()
        for category, tools in discovered_tools.items():
            if category not in merged_tools:
                merged_tools[category] = []

            # Add discovered tools, avoiding duplicates by name
            existing_names = {tool.name for tool in merged_tools[category]}
            for tool in tools:
                if tool.name not in existing_names:
                    merged_tools[category].append(tool)
                    existing_names.add(tool.name)

        self.tools = merged_tools
        return merged_tools
