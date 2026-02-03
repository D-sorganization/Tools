"""Core launcher implementation for GUI applications."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from importlib.util import find_spec
from pathlib import Path

logger = logging.getLogger(__name__)


class GUIType(Enum):
    """Supported GUI framework types."""

    PYQT6 = "pyqt6"
    REACT = "react"
    TKINTER = "tkinter"
    BROWSER = "browser"


@dataclass
class LaunchConfig:
    """Configuration for launching a GUI application."""

    tool_name: str
    gui_type: GUIType
    module_path: str | None = None
    web_path: str | None = None
    entry_point: str | None = None
    dependencies: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None
    port: int = 3000
    auto_open_browser: bool = True


@dataclass
class DependencyStatus:
    """Result of dependency check."""

    ok: bool
    missing: list[str]
    guidance: dict[str, str]


def check_python_dependencies(
    packages: list[str],
    spec_finder: Callable[[str], object] = find_spec,
) -> DependencyStatus:
    """Check if required Python packages are available.

    Args:
        packages: List of package names to check
        spec_finder: Function to check module availability (for testing)

    Returns:
        DependencyStatus with results
    """
    install_hints = {
        "PyQt6": "pip install PyQt6",
        "numpy": "pip install numpy",
        "pandas": "pip install pandas",
        "matplotlib": "pip install matplotlib",
        "scipy": "pip install scipy",
    }

    missing = [pkg for pkg in packages if spec_finder(pkg) is None]

    return DependencyStatus(
        ok=not missing,
        missing=missing,
        guidance={pkg: install_hints.get(pkg, f"pip install {pkg}") for pkg in missing},
    )


def check_node_dependencies(web_path: Path) -> DependencyStatus:
    """Check if Node.js dependencies are installed.

    Args:
        web_path: Path to the web project directory

    Returns:
        DependencyStatus with results
    """
    node_modules = web_path / "node_modules"
    package_json = web_path / "package.json"

    missing = []
    guidance = {}

    if not package_json.exists():
        missing.append("package.json")
        guidance["package.json"] = "Web project not properly configured"

    if not node_modules.exists():
        missing.append("node_modules")
        guidance["node_modules"] = f"Run: cd {web_path} && npm install"

    return DependencyStatus(
        ok=not missing,
        missing=missing,
        guidance=guidance,
    )


class GUILauncher:
    """Unified launcher for PyQt6 and React GUI applications."""

    def __init__(
        self,
        config: LaunchConfig | None = None,
        *,
        tool_name: str = "",
        gui_type: GUIType = GUIType.PYQT6,
        **kwargs,
    ) -> None:
        """Initialize the launcher.

        Args:
            config: Full launch configuration
            tool_name: Name of the tool (if not using config)
            gui_type: Type of GUI to launch (if not using config)
            **kwargs: Additional configuration options
        """
        if config:
            self.config = config
        else:
            self.config = LaunchConfig(
                tool_name=tool_name,
                gui_type=gui_type,
                **kwargs,
            )

        self._process: subprocess.Popen | None = None

    def check_dependencies(self) -> DependencyStatus:
        """Check all required dependencies.

        Returns:
            DependencyStatus indicating what's missing
        """
        if self.config.gui_type == GUIType.PYQT6:
            base_deps = ["PyQt6"] + self.config.dependencies
            return check_python_dependencies(base_deps)

        elif self.config.gui_type == GUIType.REACT:
            if self.config.web_path:
                web_path = Path(self.config.web_path)
                return check_node_dependencies(web_path)
            return DependencyStatus(ok=False, missing=["web_path"], guidance={})

        elif self.config.gui_type == GUIType.TKINTER:
            return check_python_dependencies(["tkinter"] + self.config.dependencies)

        return DependencyStatus(ok=True, missing=[], guidance={})

    def launch(self) -> int:
        """Launch the GUI application.

        Returns:
            Exit code (0 for success)
        """
        status = self.check_dependencies()
        if not status.ok:
            self._print_missing_deps(status)
            return 1

        try:
            if self.config.gui_type == GUIType.PYQT6:
                return self._launch_pyqt6()
            elif self.config.gui_type == GUIType.REACT:
                return self._launch_react()
            elif self.config.gui_type == GUIType.TKINTER:
                return self._launch_tkinter()
            elif self.config.gui_type == GUIType.BROWSER:
                return self._launch_browser()
            else:
                logger.error(f"Unsupported GUI type: {self.config.gui_type}")
                return 1
        except Exception as e:
            logger.error(f"Failed to launch {self.config.tool_name}: {e}")
            return 1

    def _launch_pyqt6(self) -> int:
        """Launch a PyQt6 application."""
        if self.config.module_path:
            cmd = [sys.executable, "-m", self.config.module_path]
        elif self.config.entry_point:
            cmd = [sys.executable, self.config.entry_point]
        else:
            logger.error("No module_path or entry_point specified for PyQt6 app")
            return 1

        env = os.environ.copy()
        env.update(self.config.env_vars)

        logger.info(f"Launching PyQt6 app: {self.config.tool_name}")
        return subprocess.call(cmd, env=env)

    def _launch_react(self) -> int:
        """Launch a React/Vite web application."""
        if not self.config.web_path:
            logger.error("No web_path specified for React app")
            return 1

        web_path = Path(self.config.web_path)

        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("npm is not installed or not in PATH")
            return 1

        logger.info(f"Starting React dev server for: {self.config.tool_name}")
        logger.info(f"  Directory: {web_path}")
        logger.info(f"  Port: {self.config.port}")

        env = os.environ.copy()
        env.update(self.config.env_vars)
        env["PORT"] = str(self.config.port)

        # Start the dev server
        cmd = ["npm", "run", "dev"]

        self._process = subprocess.Popen(
            cmd,
            cwd=web_path,
            env=env,
            shell=True,
        )

        if self.config.auto_open_browser:
            import time

            time.sleep(2)  # Give the server time to start
            webbrowser.open(f"http://localhost:{self.config.port}")

        try:
            return self._process.wait()
        except KeyboardInterrupt:
            self._process.terminate()
            return 0

    def _launch_tkinter(self) -> int:
        """Launch a Tkinter application."""
        if self.config.entry_point:
            cmd = [sys.executable, self.config.entry_point]
        elif self.config.module_path:
            cmd = [sys.executable, "-m", self.config.module_path]
        else:
            logger.error("No entry_point or module_path specified for Tkinter app")
            return 1

        logger.info(f"Launching Tkinter app: {self.config.tool_name}")
        return subprocess.call(cmd)

    def _launch_browser(self) -> int:
        """Launch a static HTML file in the browser."""
        if not self.config.web_path:
            logger.error("No web_path specified for browser app")
            return 1

        file_path = Path(self.config.web_path)
        if not file_path.exists():
            logger.error(f"HTML file not found: {file_path}")
            return 1

        logger.info(f"Opening in browser: {file_path}")
        webbrowser.open(f"file://{file_path.absolute()}")
        return 0

    def _print_missing_deps(self, status: DependencyStatus) -> None:
        """Print missing dependency information."""
        print("Missing dependencies detected:")
        for pkg in status.missing:
            hint = status.guidance.get(pkg, "")
            print(f"  - {pkg}: {hint}")
        print("\nInstall the missing packages and try again.")

    def stop(self) -> None:
        """Stop the running application (for React dev servers)."""
        if self._process:
            self._process.terminate()
            self._process = None


def create_launcher(
    tool_name: str,
    gui_type: GUIType,
    **kwargs,
) -> GUILauncher:
    """Factory function to create a launcher with common configuration.

    Args:
        tool_name: Name of the tool
        gui_type: Type of GUI framework
        **kwargs: Additional configuration options

    Returns:
        Configured GUILauncher instance
    """
    config = LaunchConfig(tool_name=tool_name, gui_type=gui_type, **kwargs)
    return GUILauncher(config=config)
