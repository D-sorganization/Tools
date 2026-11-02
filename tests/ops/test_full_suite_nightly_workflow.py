from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "full-suite-nightly.yml"


def _step_script(step_name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["full-suite"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == step_name)


def _build_install_line() -> str:
    """Return the nightly's `pip install` line that provides `build`."""
    return next(
        line
        for line in _step_script("Install Python dependencies").splitlines()
        if "python -m pip install" in line and '"build' in line
    )


def _dependency_name(spec: str) -> str:
    return (
        spec.split(";", maxsplit=1)[0]
        .split("[", maxsplit=1)[0]
        .split(">=", maxsplit=1)[0]
        .split("==", maxsplit=1)[0]
        .strip()
    )


def test_full_suite_test_extra_declares_collection_runtime_dependencies() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert 'pip install -e ".[dev,test]"' in workflow
    assert "|| pip install -e ." not in workflow

    test_deps = config["project"]["optional-dependencies"]["test"]
    dep_names = {_dependency_name(dep) for dep in test_deps}

    assert {
        "ezdxf",
        "fastapi",
        "httpx",
        "opencv-python-headless",
        "pymodbus",
        "python-multipart",
        "requests",
        "sqlmodel",
    }.issubset(dep_names)


def test_full_suite_nightly_hard_fails_missing_collection_dependencies() -> None:
    script = _step_script("Verify full-suite collection dependencies")

    assert "missing full-suite dependencies" in script
    assert "silently skip affected tests" in script
    for module_name in ("cv2", "ezdxf", "pymodbus", "requests", "sqlmodel"):
        assert module_name in script


def test_full_suite_nightly_runs_on_github_hosted_runner() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["jobs"]["full-suite"]["runs-on"] == "ubuntu-latest"


def test_full_suite_nightly_disables_xdist_on_fleet_runners() -> None:
    script = _step_script("Run full suite")

    assert "-n 0" in script
    assert "-n auto" not in script
    assert "--dist no" in script
    assert "--dist loadscope" not in script
    assert "full_suite_results.xml" in script


def test_full_suite_nightly_keeps_collection_floor_guard() -> None:
    script = _step_script("Summarize full-suite results")

    assert "total < 500" in script
    assert "collection broken" in script
    assert "sys.exit(1)" in script


def test_full_suite_nightly_provisions_chromium_before_running_the_suite() -> None:
    """The nightly collects the unit-converter browser tests too (#5911)."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["full-suite"]["steps"]
    names = [step.get("name") for step in steps]
    browser_step = "Provision bundled Chromium for unit-converter regressions"

    assert browser_step in names
    assert (
        names.index("Install Python dependencies")
        < names.index(browser_step)
        < names.index("Run full suite")
    )
    assert "continue-on-error" not in steps[names.index(browser_step)]

    script = _step_script(browser_step)
    assert 'PLAYWRIGHT_BROWSERS_PATH="$RUNNER_TEMP/playwright-browsers"' in script
    assert '"$GITHUB_ENV"' in script
    assert 'if [ "$(id -u)" -eq 0 ]; then' in script
    assert "sudo -n true" in script
    assert "flock /tmp/d-sorg-apt-install.lock" in script
    assert "-m playwright install-deps chromium" in script
    assert "-m playwright install chromium" in script
    assert "::error::" in script
    assert "exit 1" in script


def test_full_suite_nightly_keeps_libpython_reachable_under_sudo() -> None:
    """AGENT_TOOLSDIRECTORY moves setup-python's interpreter into the workspace,
    so it needs setup-python's LD_LIBRARY_PATH; sudo resets the environment and
    the interpreter died with "libpython3.11.so.1.0: cannot open" (#5948)."""
    script = _step_script("Provision bundled Chromium for unit-converter regressions")
    assert "sudo -n flock" not in script, "sudo must not run python bare"
    assert 'sudo -n env "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" flock' in script


def test_full_suite_nightly_installs_the_ci_standard_build_pin() -> None:
    """The Sidekick wheel test runs `python -m build`; ci-standard installs a
    pinned `build` but the nightly did not, so the test failed with
    "No module named build" (#5953)."""
    ci_standard = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
        encoding="utf-8"
    )
    pin = re.search(r'"build[<>=][^"]*"', ci_standard)
    assert pin, "ci-standard.yml must keep a pinned build install"
    assert pin.group(0) in _build_install_line()


def test_full_suite_nightly_installs_the_build_system_requires() -> None:
    """`python -m build --no-isolation` builds against the current environment,
    so the wheel tests failed with "Missing dependencies: wheel" until the
    nightly installed pyproject's [build-system] requires too (#5957)."""
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    requires = pyproject["build-system"]["requires"]
    install = _build_install_line()
    for requirement in requires:
        assert f'"{requirement}"' in install, (
            f"{requirement} must be installed with build"
        )
