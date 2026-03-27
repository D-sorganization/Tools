"""Python scripting environment (MATLAB-like backend).

Provides an interactive console environment that can be embedded into applications
or run standalone, featuring namespace management, persistent user scripts, and
stdout capturing.
"""

import contextlib
import io
import math
import os
import sys
import traceback
from typing import Any

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import scipy

    HAS_SCIPY = True
except ImportError:
    scipy = None
    HAS_SCIPY = False


class ConsoleEnvironment:
    """Manages the interactive Python namespace and execution for the CLI."""

    def __init__(
        self,
        default_namespace: dict[str, Any] | None = None,
        user_lib_path: str = "~/.shared_console_user_funcs.py",
    ) -> None:
        """Initialize the console environment.

        Args:
            default_namespace: Variables/modules to inject into the namespace.
            user_lib_path: Path to the user's persistent functions file.
        """
        self._initial_namespace = default_namespace or {}
        self.namespace: dict[str, Any] = {}
        self._user_lib_path = os.path.expanduser(user_lib_path)
        self.reset()

    def reset(self) -> None:
        """Reset the namespace to its initial state.

        Follows DbC by ensuring the namespace is clear before population.
        """
        self.namespace.clear()

        # Add basic dependencies
        self.namespace["np"] = np
        self.namespace["math"] = math
        if HAS_PANDAS:
            self.namespace["pd"] = pd
        if HAS_SCIPY:
            self.namespace["scipy"] = scipy

        # Inject injected defaults
        self.namespace.update(self._initial_namespace)

        # Provide specialized help function that works within the console
        self.namespace["help"] = self._custom_help

        # Load user functions if they exist
        self.refresh_user_functions()

    def _custom_help(self, obj: Any) -> None:
        """Custom help function that writes to stdout, redirectable by the CLI."""
        import pydoc

        sys.stdout.write(pydoc.render_doc(obj) + "\n")

    def set_user_library_path(self, path: str) -> None:
        """Set the path for saving/loading user custom functions."""
        # Precondition check
        if not path:
            raise ValueError("Library path cannot be empty.")
        self._user_lib_path = path

    def save_user_code(self, code: str) -> None:
        """Save raw python code to the user library path."""
        with open(self._user_lib_path, "w", encoding="utf-8") as f:
            f.write(code)

    def append_user_code(self, code: str) -> None:
        """Append raw python code to the user library path."""
        with open(self._user_lib_path, "a", encoding="utf-8") as f:
            f.write("\n" + code + "\n")

    def get_user_code(self) -> str:
        """Get the current user code."""
        if not os.path.exists(self._user_lib_path):
            return ""
        with open(self._user_lib_path, encoding="utf-8") as f:
            return f.read()

    def refresh_user_functions(self) -> None:
        """Reload user functions into the namespace."""
        if not os.path.exists(self._user_lib_path):
            return

        with open(self._user_lib_path, encoding="utf-8") as f:
            code = f.read()

        try:
            # Execute within current namespace so imports/functions are persistent
            exec(code, self.namespace)  # nosec B102
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"Error loading user library: {e}\n")

    def execute(self, source: str) -> tuple[str, str]:
        """Execute a block of source code, capturing stdout and stderr.

        Args:
            source: Raw python command string.

        Returns:
            (stdout_output, stderr_output)
        """
        assert source is not None, "source must be provided"
        if not source.strip():
            return "", ""

        out_buf = io.StringIO()
        err_buf = io.StringIO()

        try:
            with (
                contextlib.redirect_stdout(out_buf),
                contextlib.redirect_stderr(err_buf),
            ):
                # Try to evaluate as an expression first for REPL output behavior
                try:
                    code_obj = compile(source, "<console>", "eval")
                    res = eval(code_obj, self.namespace)  # nosec B307
                    if res is not None:
                        sys.stdout.write(repr(res) + "\n")
                except SyntaxError:
                    # Not an expression, try exec
                    code_obj = compile(source, "<console>", "exec")
                    exec(code_obj, self.namespace)  # nosec B102

        except Exception:  # noqa: BLE001
            # Format exception similar to REPL
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_traceback:
                # Skip the context wrapper internal frames
                tb_lines = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
                # Keep the last portion regarding user code
                err_buf.write("".join(tb_lines[-3:]))
            elif exc_type:
                err_buf.write(f"{exc_type.__name__}: {exc_value}\n")

        return out_buf.getvalue(), err_buf.getvalue()
