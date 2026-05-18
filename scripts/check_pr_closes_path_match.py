import fnmatch
import json
import os
import re
import subprocess
import sys


def run_cmd(cmd: str) -> str:
    """Run a shell command and return its stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running '{cmd}': {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main() -> None:
    """Check if the PR modifies files listed in closed issues."""
    pr_number = os.environ.get("PR_NUMBER")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not pr_number or not repo:
        print("PR_NUMBER and GITHUB_REPOSITORY must be set.", file=sys.stderr)
        sys.exit(1)

    # 1. Get PR body to find closed issues
    pr_json = run_cmd(f"gh pr view {pr_number} --repo {repo} --json body")
    pr_data = json.loads(pr_json)
    pr_body = pr_data.get("body", "")

    issue_numbers = set(
        re.findall(r"(?:[cC]loses|[fF]ixes|[rR]esolves)\s+#(\d+)", pr_body)
    )
    if not issue_numbers:
        print("No linked issues found in PR body. Passing.")
        sys.exit(0)

    # 2. Get PR modified files
    files_json = run_cmd(f"gh pr view {pr_number} --repo {repo} --json files")
    files_data = json.loads(files_json)
    pr_files = {f["path"] for f in files_data.get("files", [])}

    # 3. For each closed issue, check File Paths
    for issue_num in issue_numbers:
        issue_json = run_cmd(f"gh issue view {issue_num} --repo {repo} --json body")
        issue_data = json.loads(issue_json)
        issue_body = issue_data.get("body", "")

        # Find the ## File Paths section
        match = re.search(
            r"## File Paths\s*\n(.*?)(?:\n## |$)", issue_body, re.DOTALL | re.IGNORECASE
        )
        if not match:
            continue

        paths_section = match.group(1)
        # Extract lines starting with '-' or '*'
        required_paths = []
        for line in paths_section.splitlines():
            line = line.strip()
            if (line.startswith("-") or line.startswith("*")) and not line.startswith(
                "<!--"
            ):
                # Extract the path, removing formatting like ` or whitespace
                path = re.sub(r"^[-*]\s*", "", line)
                path = path.strip("`").strip()
                if path and "e.g." not in path:  # Skip example paths
                    required_paths.append(path)

        if not required_paths:
            continue

        print(
            f"Issue #{issue_num} requires at least one of these paths to be modified: {required_paths}"
        )

        matched = False
        for req_path in required_paths:
            for pr_file in pr_files:
                if (
                    pr_file == req_path
                    or fnmatch.fnmatch(pr_file, req_path)
                    or fnmatch.fnmatch(pr_file, f"{req_path}/*")
                ):
                    print(f"Matched: {pr_file} satisfies requirement {req_path}")
                    matched = True
                    break
            if matched:
                break

        if not matched:
            print(
                f"::error::PR does not modify any files listed in Issue #{issue_num}'s 'File Paths' section."
            )
            sys.exit(1)

    print("All issue path constraints satisfied.")


if __name__ == "__main__":
    main()
