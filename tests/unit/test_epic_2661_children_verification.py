"""Verification tests for Epic #2661 children (Tools #2928).

Enumerates all 30 child issues of Epic #2661 (Sidekick universal calc/chat
roadmap) and asserts that the expected implementation files exist on the
current branch. This test exists to prevent phantom-close regressions.

Known status at time of writing (2026-05-17 audit):
- #2673 (Jupyter tabs): Original close phantom. Superseded by #2875-#2877.
  Verified by #2930 + #2940 (both CLOSED). Phased implementation exists.
- #2674 (Notes tab): Phantom close. Follow-up #2931 (OPEN, in progress).
  Branch fix/issue-2931-markdown-notes-tab has PR #2974.
- #2747 (Reporting context): Phantom close. Fixed, follow-up #2936 CLOSED.
- #2834 (Obsidian Phase 2): Phantom close. Real impl verified by #2938.
- #2874 (Jupyter impl missing): Superseded by #2875-#2877.
- #2682 (Symbolic solver): Branch fix/issue-2934-symbolic-solver has impl.

Cross-references: #2661, #2928
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDEKICK = REPO_ROOT / "src" / "shared" / "python" / "sidekick"
SIDEBAR = SIDEKICK / "ui" / "tools_sidebar"
INTEGRATION = REPO_ROOT / "tests" / "integration" / "sidekick"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _exists(rel_path: str) -> bool:
    """Return True if the relative path (from repo root) exists."""
    return (REPO_ROOT / rel_path).is_file() or (REPO_ROOT / rel_path).is_dir()


# ---------------------------------------------------------------------------
# Tests — one per child issue group
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_2662_tab_context_menus() -> None:
    """#2662: Tab workflow controls moved to right-click menus."""
    assert (
        SIDEBAR / "tab_context_menu.py"
    ).is_file(), "tab_context_menu.py missing — #2662 may be phantom-closed"
    assert (
        SIDEBAR / "tab_context_menu.py"
    ).stat().st_size > 500, "tab_context_menu.py appears to be a stub (< 500 bytes)"


@pytest.mark.unit
def test_2663_tab_settings_panels() -> None:
    """#2663: Per-tab persistent settings panels."""
    assert (SIDEBAR / "tab_settings_panel.py").is_file()
    assert (SIDEBAR / "tab_settings_panel.py").stat().st_size > 500


@pytest.mark.unit
def test_2664_rename_tabs() -> None:
    """#2664: Users can rename tabs with persistent custom display names."""
    assert (SIDEBAR / "tab_display_names.py").is_file()
    assert (SIDEBAR / "tab_display_names.py").stat().st_size > 200


@pytest.mark.unit
def test_2665_state_profiles() -> None:
    """#2665: State profiles with save, load, clear data."""
    assert (SIDEBAR / "state_profiles.py").is_file()
    assert (SIDEBAR / "state_profiles.py").stat().st_size > 500


@pytest.mark.unit
def test_2666_theme_support() -> None:
    """#2666: Inherited parent themes and custom color/font themes."""
    assert (SIDEBAR / "theme_settings.py").is_file()
    assert (SIDEBAR / "theme_settings.py").stat().st_size > 200


@pytest.mark.unit
def test_2667_calculator_workspaces() -> None:
    """#2667: Separate calculator-local workspaces from global variables."""
    assert (SIDEBAR / "calculator_workspace.py").is_file()
    assert (SIDEBAR / "calculator_workspace.py").stat().st_size > 500


@pytest.mark.unit
def test_2668_matlab_workspace() -> None:
    """#2668: MATLAB-like workspace save, load, clear, variable management."""
    assert (SIDEBAR / "workspace_commands.py").is_file()
    assert (SIDEBAR / "workspace_commands.py").stat().st_size > 500


@pytest.mark.unit
def test_2669_help_panels() -> None:
    """#2669: Help panels and hover hints for every tab and icon."""
    assert (SIDEBAR / "help_content.py").is_file()
    assert (SIDEBAR / "help_content.py").stat().st_size > 500


@pytest.mark.unit
def test_2670_function_generator_tab() -> None:
    """#2670: Function Generator as a first-class tab."""
    assert (SIDEBAR / "default_tabs.py").is_file()
    # default_tabs.py should define the standard tab set including FuncGen
    content = (SIDEBAR / "default_tabs.py").read_text(encoding="utf-8")
    assert (
        "function" in content.lower()
        or "funcgen" in content.lower()
        or "func_gen" in content.lower()
    ), "default_tabs.py does not appear to include Function Generator tab"


@pytest.mark.unit
def test_2671_data_processor_tab() -> None:
    """#2671: Data Processor as a first-class tab."""
    assert (SIDEBAR / "data_processor_tab.py").is_file()
    assert (SIDEBAR / "data_processor_tab.py").stat().st_size > 500


@pytest.mark.unit
def test_2672_tab_visibility_settings() -> None:
    """#2672: Settings for default visible/hidden tabs."""
    assert (SIDEBAR / "tab_visibility.py").is_file()
    assert (SIDEBAR / "tab_visibility.py").stat().st_size > 200


@pytest.mark.unit
def test_2673_jupyter_tab_phased_implementation() -> None:
    """#2673: Jupyter notebook tabs — phased re-implementation (#2875-#2877).

    Original close was a phantom. Phased re-implementation (#2875-#2877)
    landed the availability check + widget skeleton. Follow-up #2930 and
    #2940 are both CLOSED indicating verification was completed.
    """
    # The phased implementation should have a jupyter_tab subpackage
    jupyter_dir = SIDEBAR / "jupyter_tab"
    assert (
        jupyter_dir.is_dir()
    ), "jupyter_tab/ directory missing — phased Jupyter implementation not landed"
    assert (jupyter_dir / "widget.py").is_file(), "jupyter_tab/widget.py missing"
    assert (
        jupyter_dir / "availability.py"
    ).is_file(), "jupyter_tab/availability.py missing (soft-dependency guard)"


@pytest.mark.unit
@pytest.mark.xfail(
    not (SIDEKICK / "ui" / "tools_sidebar" / "notes_tab.py").is_file()
    and not (SIDEKICK / "notes_tab.py").is_file(),
    reason=(
        "Notes tab (#2674) was phantom-closed. Follow-up #2931 is in progress "
        "(branch fix/issue-2931-markdown-notes-tab, PR #2974). "
        "This xfail will be removed when the PR merges to main."
    ),
    strict=False,
)
def test_2674_notes_tab() -> None:
    """#2674: Visual markdown Notes tab with note cards and colors.

    Known phantom-close. Follow-up: #2931 (OPEN).
    """
    notes_in_sidebar = (SIDEBAR / "notes_tab.py").is_file()
    notes_in_sidekick = (SIDEKICK / "notes_tab.py").is_file()
    assert notes_in_sidebar or notes_in_sidekick, (
        "notes_tab.py missing — #2674 phantom-close confirmed, "
        "see follow-up issue #2931"
    )


@pytest.mark.unit
def test_2675_shared_calculator_workspace_contract() -> None:
    """#2675: Shared calculator tab workspace and expression execution contract."""
    assert (SIDEBAR / "calculator_workspace.py").is_file()
    # Check that the workspace contract protocol is defined
    workspace_contract = (
        REPO_ROOT / "src" / "shared" / "python" / "sidekick" / "workspace_contract.py"
    )
    assert (
        workspace_contract.is_file()
    ), "workspace_contract.py missing — #2675 shared contract not implemented"


@pytest.mark.unit
def test_2676_host_integration() -> None:
    """#2676: Proven shared host integration across downstream consumers."""
    assert (
        INTEGRATION / "test_sidekick_host_integration.py"
    ).is_file(), "Integration test file missing for #2676"
    content = (INTEGRATION / "test_sidekick_host_integration.py").read_text(
        encoding="utf-8"
    )
    assert len(content) > 500, "Host integration test appears to be a stub"


@pytest.mark.unit
def test_2677_startup_imports() -> None:
    """#2677: Configurable startup imports and user dependency settings."""
    assert (SIDEBAR / "calculator_startup.py").is_file()
    assert (SIDEBAR / "calculator_startup.py").stat().st_size > 200


@pytest.mark.unit
def test_2678_command_history() -> None:
    """#2678: Command history navigation with up-arrow previews."""
    assert (SIDEBAR / "command_history.py").is_file()
    assert (SIDEBAR / "command_history.py").stat().st_size > 200


@pytest.mark.unit
def test_2679_help_and_predictive_text() -> None:
    """#2679: Usage help, tips, and optional predictive text in calculator."""
    # This overlaps with help_content.py and calculator_assist.py
    assert (SIDEBAR / "help_content.py").is_file()
    assist = SIDEBAR / "calculator_assist.py"
    # Either help_content or a dedicated assist file should cover predictive text
    assert assist.is_file() or (SIDEBAR / "help_content.py").stat().st_size > 1000


@pytest.mark.unit
def test_2680_arrays_and_matrices() -> None:
    """#2680: Arrays, matrices, and MATLAB-like variable previews."""
    assert (SIDEBAR / "calculator_runtime.py").is_file()
    assert (SIDEBAR / "calculator_runtime.py").stat().st_size > 500


@pytest.mark.unit
def test_2681_plotting_tab() -> None:
    """#2681: Plotting tab for equation and workspace results."""
    assert (SIDEBAR / "calculator_plotting.py").is_file()
    assert (SIDEBAR / "calculator_plotting.py").stat().st_size > 500


@pytest.mark.unit
@pytest.mark.xfail(
    not (SIDEKICK / "symbolic_engine.py").is_file(),
    reason=(
        "Symbolic solver (#2682) in progress on branch fix/issue-2934-symbolic-solver. "
        "Will be removed after PR merges."
    ),
    strict=False,
)
def test_2682_symbolic_solver() -> None:
    """#2682: Symbolic solver, guided workflows, LaTeX equation rendering.

    Being implemented on branch fix/issue-2934-symbolic-solver.
    """
    assert (
        SIDEKICK / "symbolic_engine.py"
    ).is_file(), "symbolic_engine.py missing — #2682 not yet fully landed"


@pytest.mark.unit
def test_2683_workspace_save_load() -> None:
    """#2683: Local workspace save/load wired into calculator tab settings."""
    assert (SIDEBAR / "calculator_workspace.py").is_file()
    content = (SIDEBAR / "calculator_workspace.py").read_text(encoding="utf-8")
    assert (
        "save" in content.lower()
        or "persist" in content.lower()
        or "checkpoint" in content.lower()
    ), "calculator_workspace.py does not appear to implement save/load"


@pytest.mark.unit
def test_2684_rotation_converter_tab() -> None:
    """#2684: Rotation Converter as an optional tab.

    The Rotation Converter tab is registered in default_tabs.py (not runtime_tabs.py).
    """
    assert (SIDEBAR / "default_tabs.py").is_file()
    content = (SIDEBAR / "default_tabs.py").read_text(encoding="utf-8")
    assert (
        "rotation" in content.lower() or "ROTATION_CONVERTER" in content
    ), "default_tabs.py does not appear to include Rotation Converter tab"


@pytest.mark.unit
def test_2685_file_explorer_open() -> None:
    """#2685: File explorer: open files with Windows default program."""
    assert (SIDEBAR / "project_file_explorer.py").is_file()
    content = (SIDEBAR / "project_file_explorer.py").read_text(encoding="utf-8")
    assert (
        "open" in content.lower()
        or "launch" in content.lower()
        or "startfile" in content.lower()
    )


@pytest.mark.unit
def test_2686_file_explorer_navigation() -> None:
    """#2686: Common locations sidebar and back/forward/up navigation."""
    assert (SIDEBAR / "file_navigation.py").is_file()
    assert (SIDEBAR / "file_navigation.py").stat().st_size > 200


@pytest.mark.unit
def test_2687_chat_duplicate() -> None:
    """#2687: Redock flow and duplicating chat tabs."""
    assert (SIDEBAR / "tab_popout.py").is_file()
    assert (SIDEBAR / "tab_popout.py").stat().st_size > 200


@pytest.mark.unit
def test_2688_chat_history_settings() -> None:
    """#2688: Chat history, settings, memory management, modes, permissions."""
    assert (SIDEBAR / "settings.py").is_file()
    assert (SIDEBAR / "settings.py").stat().st_size > 500


@pytest.mark.unit
def test_2689_terminal_theme() -> None:
    """#2689: Terminal inherited theme default and custom color settings."""
    assert (SIDEBAR / "os_terminal.py").is_file()
    content = (SIDEBAR / "os_terminal.py").read_text(encoding="utf-8")
    assert "theme" in content.lower() or "color" in content.lower()


@pytest.mark.unit
def test_2690_workspace_matlab_cli() -> None:
    """#2690: MATLAB-like command line for creating and editing variables."""
    assert (SIDEBAR / "workspace_commands.py").is_file()
    content = (SIDEBAR / "workspace_commands.py").read_text(encoding="utf-8")
    assert "variable" in content.lower() or "workspace" in content.lower()


@pytest.mark.unit
def test_2691_data_explorer_tab() -> None:
    """#2691: Data Explorer tab for inspecting data files."""
    assert (SIDEBAR / "data_explorer_tab.py").is_file()
    assert (SIDEBAR / "data_explorer_tab.py").stat().st_size > 500


# ---------------------------------------------------------------------------
# Summary test: count implemented vs phantom
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_epic_2661_implementation_summary() -> None:
    """Report the implementation status of all #2661 children.

    This test always passes — it exists to surface the summary so it appears
    in the test output and can be used to track progress.
    """
    files_to_check = [
        ("tab_context_menu.py", "#2662"),
        ("tab_settings_panel.py", "#2663"),
        ("tab_display_names.py", "#2664"),
        ("state_profiles.py", "#2665"),
        ("theme_settings.py", "#2666"),
        ("calculator_workspace.py", "#2667/#2668/#2675/#2683"),
        ("workspace_commands.py", "#2668/#2690"),
        ("help_content.py", "#2669/#2679"),
        ("default_tabs.py", "#2670"),
        ("data_processor_tab.py", "#2671"),
        ("tab_visibility.py", "#2672"),
        ("jupyter_tab/widget.py", "#2673"),
        ("calculator_startup.py", "#2677"),
        ("command_history.py", "#2678"),
        ("calculator_runtime.py", "#2680"),
        ("calculator_plotting.py", "#2681"),
        ("runtime_tabs.py", "#2684"),
        ("project_file_explorer.py", "#2685"),
        ("file_navigation.py", "#2686"),
        ("tab_popout.py", "#2687"),
        ("settings.py", "#2688"),
        ("os_terminal.py", "#2689"),
        ("data_explorer_tab.py", "#2691"),
    ]

    present = [f for f, _ in files_to_check if (SIDEBAR / f).is_file()]
    missing_core = [
        (f, issue) for f, issue in files_to_check if not (SIDEBAR / f).is_file()
    ]

    # Known pending items (in-progress PRs)
    pending = {
        "#2674": "#2931 (PR #2974, branch fix/issue-2931-markdown-notes-tab)",
        "#2682": "#2934 (branch fix/issue-2934-symbolic-solver)",
    }

    import warnings

    warnings.warn(
        f"Epic #2661 implementation summary: "
        f"{len(present)}/{len(files_to_check)} sidebar files present on main. "
        f"In-progress: {pending}. "
        f"Files still missing: {missing_core}",
        UserWarning,
        stacklevel=2,
    )
    assert len(present) + len(missing_core) == len(
        files_to_check
    ), "Epic #2661 summary inventory lost or duplicated file entries"
