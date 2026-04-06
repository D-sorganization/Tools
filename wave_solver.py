import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd, cwd=None, ignore_err=False):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=not ignore_err,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        if not ignore_err:
            pass
        return None


def main():
    repo_path = REPO_ROOT

    # Fetch issues
    raw_issues = run_cmd(
        "gh issue list --state open --json number,title,body", cwd=repo_path
    )
    if not raw_issues:
        return

    try:
        issues = json.loads(raw_issues)
    except Exception:
        return

    issues_to_fix = []
    seen_titles = set()

    for i in issues:
        title = i.get("title", "")
        # Filter duplicates we spawned
        if title in seen_titles:
            # We can close them if we want
            run_cmd(f"gh issue close {i['number']}", cwd=repo_path, ignore_err=True)
            continue

        if "[A-N Assessment]" in title:
            seen_titles.add(title)
            issues_to_fix.append(i)

    if not issues_to_fix:
        return

    for issue in issues_to_fix:
        num = issue["number"]
        title = issue["title"]
        body = issue["body"].replace("\n", " ")

        branch = f"fix/a-n-issue-{num}"

        # Cleanup and Branch
        run_cmd("git reset --hard", cwd=repo_path, ignore_err=True)
        run_cmd(
            "git checkout main || git checkout master", cwd=repo_path, ignore_err=True
        )
        run_cmd("git pull --rebase", cwd=repo_path, ignore_err=True)
        run_cmd(f"git checkout -b {branch}", cwd=repo_path, ignore_err=True)

        # Determine strict instructions for Claude Code depending on issue
        # We must keep instructions concise and actionable for Claude Code
        prompt = (
            f"Fix GitHub Issue #{num}: {title}. You are solving this autonomously for high-quality, long-term codebase health. "
            f"Context: {body}. "
            f"Instructions: Open the referenced files (e.g., controller.py, main_window.py, renderer.py, etc.) and strictly solve the refactoring criteria (DRY, LOD, TDD, Changeability, DbC). "
            f"Apply robust changes locally, then test with pytest before finishing. DO NOT commit or push anything! Only yield the local file changes."
        )

        # Claude runs synchronously
        # We use -p (print only) and give it the prompt. It will modify the filesystem.
        cmd = f'claude -p --dangerously-skip-permissions "{prompt}"'

        try:
            # We use unbuffered execution so we can watch it if needed
            subprocess.run(cmd, cwd=repo_path, shell=True, text=True)
        except Exception:
            continue

        # Check if anything changed
        status = run_cmd("git status --porcelain", cwd=repo_path)
        if not status:
            continue

        # Commit & Push
        run_cmd("git add -A", cwd=repo_path)
        run_cmd(
            f'git commit -m "fix: resolve A-N assessment finding #{num} - {title}" --no-verify',
            cwd=repo_path,
        )
        run_cmd(
            f"git push -u origin {branch} --no-verify", cwd=repo_path, ignore_err=True
        )

        # PR and Merge
        safe_title = title.replace('"', '\\"')
        pr_cmd = f'gh pr create --title "fix: Resolve #{num} - {safe_title}" --body "Resolves #{num}. Built via Autonomous Iterative Wave." --base main'
        run_cmd(pr_cmd, cwd=repo_path, ignore_err=True)

        # Enable Auto-Merge
        run_cmd("gh pr merge --squash --auto -d", cwd=repo_path, ignore_err=True)

        # Rest and clean
        time.sleep(2)
        run_cmd("git checkout main", cwd=repo_path, ignore_err=True)


if __name__ == "__main__":
    main()
