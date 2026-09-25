"""Fixture corpus for the knowledge-pack engine (#5345)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from shared.python.ai.knowledge import PackManifest, manifest_from_dict


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    git(root, "config", "commit.gpgsign", "false")
    return root


def commit_all(root: Path, message: str = "corpus") -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "--no-verify", "-m", message)
    return git(root, "rev-parse", "HEAD")


ARTICLE = """---
title: Energy Transfer
---

# Energy Transfer

Intro paragraph about the kinematic sequence.

## Timing

Peak pelvis speed precedes peak torso speed by 30 ms in every capture.

```python
# not a heading
x = 1
```

## Shaft Flex

The shaft stores elastic energy late in the downswing.
"""

OLD_RESULT = """---
status: superseded
---

# Bounce Result

Bounce reduces dig depth in bunker shots.
"""

NEW_RESULT = """# Bounce Inversion

More bounce digs deeper in our model across every tested cell.
"""

PAPER = r"""\documentclass{article}
\begin{document}
\section{Methods}
We fit a rigid segment chain to the capture data.
\subsection{Filtering}
A fourth-order Butterworth filter at 12 Hz.
\end{document}
"""


@pytest.fixture
def corpus(tmp_path: Path) -> dict[str, Path]:
    """Two git repositories shaped like AffineDrift and UpstreamDrift."""
    affine = git_repo(tmp_path / "AffineDrift")
    (affine / "articles" / "energy").mkdir(parents=True)
    (affine / "articles" / "energy" / "energy.qmd").write_text(ARTICLE, "utf-8")
    (affine / "articles" / "energy" / "energy-bibliography.md").write_text(
        "# Bibliography\n\nKinematic sequence references.\n", "utf-8"
    )
    commit_all(affine)

    upstream = git_repo(tmp_path / "UpstreamDrift")
    research = upstream / "docs" / "research"
    research.mkdir(parents=True)
    (research / "paper.tex").write_text(PAPER, "utf-8")
    assessments = upstream / "docs" / "assessments"
    assessments.mkdir(parents=True)
    (assessments / "bounce_old.md").write_text(OLD_RESULT, "utf-8")
    (assessments / "bounce_new.md").write_text(NEW_RESULT, "utf-8")
    (assessments / "retracted.md").write_text(
        "# Retracted Claim\n\nBounce has no effect on dig depth.\n", "utf-8"
    )
    commit_all(upstream)
    return {"AffineDrift": affine, "UpstreamDrift": upstream}


MANIFEST: dict[str, object] = {
    "id": "findings",
    "title": "Fixture findings",
    "chunk_chars": 400,
    "sources": [
        {
            "repo": "AffineDrift",
            "authority": "published",
            "include": ["articles/**/*.qmd", "articles/**/*.md"],
            "exclude": ["**/*-bibliography.md"],
        },
        {
            "repo": "UpstreamDrift",
            "authority": "findings",
            "include": ["docs/research/**/*.tex"],
            "exclude": [],
        },
        {
            "repo": "UpstreamDrift",
            "authority": "reviews",
            "include": ["docs/assessments/**/*.md"],
            "exclude": [],
        },
    ],
    "status_overrides": {"docs/assessments/retracted.md": "retracted"},
}


@pytest.fixture
def manifest() -> PackManifest:
    return manifest_from_dict(MANIFEST)
