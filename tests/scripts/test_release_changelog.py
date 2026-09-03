"""Release bump gating and delta changelog rendering (Tools #4910, RM #1507)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import release_changelog as rc  # noqa: E402


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("feat(putting): add green view (#1)", "minor"),
        ("feat: bare", "minor"),
        ("fix(ci): thing", "patch"),
        ("perf: faster", "patch"),
        ("feat(api)!: drop endpoint", "major"),
        ("fix: x\n\nBREAKING CHANGE: y", "major"),
        ("chore(release): bump version to v1.15.0 (#4865)", "none"),
        ("chore(deps): bump numpy", "none"),
        ("docs: readme", "none"),
        ("ci: runner", "none"),
        ("test: add", "none"),
        ("refactor(x): tidy", "none"),
        ("@ (#4851)", "none"),
        ("Merge branch main", "none"),
    ],
)
def test_bump_for_subject(subject: str, expected: str) -> None:
    assert rc.bump_for_subject(subject) == expected


def test_push_bumps_only_for_feat_fix_perf_at_head() -> None:
    delta = ["feat: earlier feature", "fix: earlier fix"]
    assert (
        rc.decide_bump(
            event="push",
            head_subject="chore(release): bump version to v1.15.0 (#4865)",
            actor="github-actions[bot]",
            force_bump="",
            delta_subjects=delta,
        )
        == "none"
    )
    assert (
        rc.decide_bump(
            event="push",
            head_subject="docs: tweak (#9)",
            actor="dieter",
            force_bump="",
            delta_subjects=delta,
        )
        == "none"
    ), "a docs merge must not bump even though the delta has a feat"
    assert (
        rc.decide_bump(
            event="push",
            head_subject="fix(p1am): interlock (#4928)",
            actor="dieter",
            force_bump="",
            delta_subjects=delta,
        )
        == "patch"
    )
    assert (
        rc.decide_bump(
            event="push",
            head_subject="feat(x): y (#1)",
            actor="dependabot[bot]",
            force_bump="",
            delta_subjects=delta,
        )
        == "none"
    ), "bot pushes never bump"


def test_push_ignores_force_bump_input() -> None:
    assert (
        rc.decide_bump(
            event="push",
            head_subject="docs: x",
            actor="me",
            force_bump="major",
            delta_subjects=[],
        )
        == "none"
    )


def test_dispatch_honours_force_then_auto_detects() -> None:
    assert (
        rc.decide_bump(
            event="workflow_dispatch",
            head_subject="docs: x",
            actor="me",
            force_bump="minor",
            delta_subjects=[],
        )
        == "minor"
    )
    assert (
        rc.decide_bump(
            event="workflow_dispatch",
            head_subject="docs: x",
            actor="me",
            force_bump="",
            delta_subjects=["docs: a", "fix: b", "feat: c"],
        )
        == "minor"
    )
    assert (
        rc.decide_bump(
            event="workflow_dispatch",
            head_subject="docs: x",
            actor="me",
            force_bump="",
            delta_subjects=["docs: a", "chore: b"],
        )
        == "none"
    )


@pytest.mark.parametrize(
    ("actor", "expected"),
    [
        ("dependabot[bot]", True),
        ("github-actions[bot]", True),
        ("d-sorgclaudeagent[bot]", True),
        ("dependabot", True),
        ("dieter", False),
        ("", False),
        (None, False),
    ],
)
def test_is_bot_actor(actor: str | None, expected: bool) -> None:
    assert rc.is_bot_actor(actor) is expected


def test_next_version() -> None:
    assert rc.next_version("1.15.0", "none") == "1.15.0"
    assert rc.next_version("1.15.0", "patch") == "1.15.1"
    assert rc.next_version("1.15.3", "minor") == "1.16.0"
    assert rc.next_version("1.15.3", "major") == "2.0.0"


def test_empty_subject_rows_resolve_to_the_pr_title_never_at_sign() -> None:
    titles = {4851: "feat(putting, #4800 P6): Qt Putting tab", 4887: ""}
    resolve = titles.get
    assert (
        rc.render_entry_line("@ (#4851)", resolve)
        == "- feat(putting, #4800 P6): Qt Putting tab (#4851)"
    )
    # Title lookup empty -> a descriptive placeholder, still never "- @".
    assert rc.render_entry_line("@ (#4887)", resolve) == "- PR #4887 (#4887)"
    # Lookup failure (None) -> same placeholder.
    assert rc.render_entry_line("- (#77)", lambda _n: None) == "- PR #77 (#77)"
    # No PR number and no text: dropped.
    assert rc.render_entry_line("@", resolve) is None
    # Bump commits are dropped, ordinary subjects pass through unchanged.
    assert (
        rc.render_entry_line("chore(release): bump version to v1.14.0 (#4859)", resolve)
        is None
    )
    assert rc.render_entry_line("fix(ci): x (#12)", resolve) == "- fix(ci): x (#12)"


def test_render_entry_groups_and_dedupes() -> None:
    entry = rc.render_entry(
        [
            "- fix(ci): a (#1)",
            "- feat(x): b (#2)",
            "- fix(ci): a (#1)",
            "- docs: c (#3)",
            "- perf: d (#4)",
        ]
    )
    assert entry.count("fix(ci): a (#1)") == 1
    assert (
        entry.index("#### Features")
        < entry.index("#### Fixes")
        < entry.index("#### Performance")
    )
    assert entry.index("#### Performance") < entry.index("#### Other")
    assert "- @" not in entry
    assert rc.render_entry([]).startswith("- No changes")


def test_end_to_end_on_a_temp_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "a").write_text("1")
    git("add", "a")
    git("commit", "-q", "-m", "feat: first (#1)")
    git(
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "chore(release): bump version to v1.0.0 (#2)",
    )
    git("commit", "-q", "--allow-empty", "-m", "@ (#3)")
    git("commit", "-q", "--allow-empty", "-m", "fix(ci): second (#4)")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        rc, "gh_pr_title", lambda _repo, n: {3: "docs: resolved title"}.get(n)
    )
    out = tmp_path / "entry.md"
    gh_out = tmp_path / "gh_output"
    assert (
        rc.main(
            [
                "--current-version",
                "1.0.0",
                "--event",
                "push",
                "--actor",
                "dieter",
                "--out",
                str(out),
                "--github-output",
                str(gh_out),
            ]
        )
        == 0
    )
    entry = out.read_text(encoding="utf-8")
    # Delta only: the pre-bump "feat: first" is NOT re-listed.
    assert "first (#1)" not in entry
    assert "- docs: resolved title (#3)" in entry
    assert "- fix(ci): second (#4)" in entry
    assert "- @" not in entry
    assert "bump=patch" in gh_out.read_text()
    assert "new_version=1.0.1" in gh_out.read_text()
