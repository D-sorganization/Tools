"""Regression coverage for shared AI/chat dependency imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_shared_ai_dependencies_import_without_test_stubs(
    tmp_path: Path,
) -> None:
    """Adapter imports should not require sys.modules stubs from tests."""
    repo_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )
    src_package = tmp_path / "src"
    src_package.mkdir()
    (src_package / "__init__.py").write_text(
        f"__path__ = [{str(repo_root / 'src')!r}]\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    path_entries = [
        str(tmp_path),
        str(repo_root),
        str(repo_root / "src" / "shared" / "python"),
        str(repo_root / "src" / "python" / "src"),
        str(repo_root / "src"),
    ]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(path_entries + ([existing] if existing else []))

    probe = "\n".join(
        [
            "from src.shared.python.ai.adapters.factory import AdapterFactory",
            "from src.shared.python.ai.config import get_ollama_timeout",
            "from src.shared.python.config.environment import get_env_float",
            "from src.shared.python.logging_pkg.logging_config import get_logger",
            "assert AdapterFactory.__name__ == 'AdapterFactory'",
            "assert get_ollama_timeout() == 120.0",
            "assert get_env_float('TOOLS_MISSING_FLOAT', 1.5) == 1.5",
            "assert get_logger('tools.import_probe').name == 'tools.import_probe'",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
