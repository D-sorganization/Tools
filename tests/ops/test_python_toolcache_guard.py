from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / ".github" / "scripts" / "clean-python-toolcache.sh"
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"


def test_pyo3_guard_precedes_setup_python_in_rust_job() -> None:
    guard = GUARD.read_text(encoding="utf-8")
    assert '"/opt/hostedtoolcache"' in guard
    assert 'sysconfig.get_config_var("LDLIBRARY")' in guard
    assert '"--require-link-library"' in guard

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    rust_steps = workflow["jobs"]["rust-quality-gate"]["steps"]
    clean_index = next(
        index
        for index, step in enumerate(rust_steps)
        if step.get("name") == "Force-clean incomplete Python tool cache for PyO3"
    )
    assert rust_steps[clean_index]["run"].endswith("'3.11' --require-link-library")
    assert str(rust_steps[clean_index + 1].get("uses", "")).startswith(
        "actions/setup-python@"
    )


def test_rust_quality_gate_allows_cold_cache_completion() -> None:
    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    rust_job = workflow["jobs"]["rust-quality-gate"]

    # This lane compiles two test configurations, cargo-audit, a Python wheel,
    # two WASM packages, and benchmarks. Fifteen minutes has twice canceled a
    # healthy cold-cache run while cargo-audit was still compiling.
    assert int(rust_job["timeout-minutes"]) >= 45


def _write_fake_python(arch_dir: Path, *, with_link_library: bool) -> None:
    python = arch_dir / "bin" / "python"
    library_dir = arch_dir / "lib"
    python.parent.mkdir(parents=True)
    library_dir.mkdir()
    python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *sys.version_info* ]]; then echo 3.11.15; exit 0; fi\n'
        'if [[ "$*" == *sysconfig* ]]; then\n'
        f"  echo '{library_dir}'\n"
        "  echo 'libpython3.11.so'\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$*\" == *'pip --version'* ]]; then\n"
        "  echo 'pip 25.0 from /fixture/pip (python 3.11)'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    if with_link_library:
        (library_dir / "libpython3.11.so").write_bytes(b"fixture")


def _run_guard(cache_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON_TOOLCACHE_ROOTS"] = str(cache_root)
    return subprocess.run(
        ["bash", str(GUARD), "3.11", *args],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


@pytest.mark.skipif(os.name == "nt", reason="Linux tool-cache contract")
def test_pyo3_guard_keeps_cache_with_a_link_library(tmp_path: Path) -> None:
    arch_dir = tmp_path / "Python" / "3.11.15" / "x64"
    _write_fake_python(arch_dir, with_link_library=True)

    result = _run_guard(tmp_path, "--require-link-library")

    assert result.returncode == 0, result.stderr
    assert arch_dir.is_dir()
    assert f"Directory {arch_dir} is healthy." in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="Linux tool-cache contract")
def test_pyo3_guard_removes_cache_without_a_link_library(tmp_path: Path) -> None:
    arch_dir = tmp_path / "Python" / "3.11.15" / "x64"
    _write_fake_python(arch_dir, with_link_library=False)
    complete_marker = arch_dir.with_name("x64.complete")
    complete_marker.write_text("fixture", encoding="utf-8")

    result = _run_guard(tmp_path, "--require-link-library")

    assert result.returncode == 0, result.stderr
    assert not arch_dir.exists()
    assert not complete_marker.exists()
    assert "Python link library is missing" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="Linux tool-cache contract")
def test_general_guard_does_not_require_a_link_library(tmp_path: Path) -> None:
    arch_dir = tmp_path / "Python" / "3.11.15" / "x64"
    _write_fake_python(arch_dir, with_link_library=False)

    result = _run_guard(tmp_path)

    assert result.returncode == 0, result.stderr
    assert arch_dir.is_dir()
