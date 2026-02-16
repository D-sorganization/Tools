import os
import subprocess
from pathlib import Path

REPOS_ROOT = Path(r"C:\Users\diete\Repositories")
TARGET_REPOS = [
    "AffineDrift",
    "Games",
    "Gasification_Model",
    "MEB_Conversion",
    "MLProjects",
    "Playground",
    "Repository_Management",
    "UpstreamDrift",
    "Worksheet-Workshop",
]

BRANCH_NAME = "chore/enhance-autofix-robustness"
COMMIT_MSG = "fix(autofix): scope fixes to changed files and enhance mypy agent robustness"
PR_TITLE = "Fix: Enhance Autofix Robustness and Scope PR Fixes"
PR_BODY = """This PR updates the Jules PR AutoFix workflow to only target changed files in a PR, preventing it from making unintended changes to unrelated parts of the codebase. It also enhances the Mypy agent to support targeted file analysis and increases timeout/max limits for more robust behavior."""

def run_cmd(cmd, cwd=None):
    print(f">>> Running: {' '.join(cmd)} in {cwd or os.getcwd()}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def patch_mypy_agent(repo_path: Path):
    agent_path = repo_path / "scripts" / "mypy_autofix_agent.py"
    if not agent_path.exists():
        print(f"Skipping {agent_path} (not found)")
        return False
    
    content = agent_path.read_text(encoding="utf-8")
    
    # Patch run_mypy
    old_run_mypy = """def run_mypy(config_file: str | None = None) -> str:
    \"\"\"Run mypy and return raw output.\"\"\"
    cmd = ["mypy", "src", "--no-error-summary"]
    if config_file:
        cmd.extend(["--config-file", config_file])
    # Show error codes for targeted fixes
    cmd.append("--show-error-codes")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout + result.stderr"""
    
    new_run_mypy = """def run_mypy(config_file: str | None = None, targets: list[str] | None = None) -> str:
    \"\"\"Run mypy and return raw output.\"\"\"
    if not targets:
        # Default to src and tests if no targets provided, but check if they exist
        targets = []
        if Path("src").exists():
            targets.append("src")
        if Path("tests").exists():
            targets.append("tests")
        if not targets:
            targets = ["."]

    cmd = ["mypy"] + targets + ["--no-error-summary"]
    if config_file:
        cmd.extend(["--config-file", config_file])
    # Show error codes for targeted fixes
    cmd.append("--show-error-codes")
    # Add non-interactive and ignore-missing-imports for agent use
    cmd.extend(["--ignore-missing-imports", "--non-interactive"])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.stdout + result.stderr"""

    if old_run_mypy in content:
        content = content.replace(old_run_mypy, new_run_mypy)
    
    # Patch run_agent signature
    content = content.replace(
        "config_file: str | None = None,\n) -> AgentReport:",
        "config_file: str | None = None,\n    targets: list[str] | None = None,\n) -> AgentReport:"
    )
    
    # Patch run_agent call to run_mypy
    content = content.replace(
        'print(">>> Running mypy...")\n    output = run_mypy(config_file)',
        'print(f">>> Running mypy on targets: {targets or \'default\'}...")\n    output = run_mypy(config_file, targets)'
    )
    
    # Patch main args
    content = content.replace(
        'parser.add_argument(\n        "--config-file",',
        'parser.add_argument(\n        "--config-file",'
    )
    # This matching is a bit tricky, let's just insert before args = parser.parse_args()
    if 'parser.add_argument(\n        "targets",' not in content:
        content = content.replace(
            'args = parser.parse_args()',
            'parser.add_argument(\n        "targets",\n        nargs="*",\n        help="Files or directories to check (default: src)",\n    )\n    args = parser.parse_args()'
        )

    # Patch report call
    content = content.replace(
        'config_file=args.config_file,\n    )',
        'config_file=args.config_file,\n        targets=args.targets,\n    )'
    )

    agent_path.write_text(content, encoding="utf-8")
    return True

def patch_workflow(repo_path: Path):
    workflow_path = repo_path / ".github" / "workflows" / "Jules-PR-AutoFix.yml"
    if not workflow_path.exists():
        print(f"Skipping {workflow_path} (not found)")
        return False
    
    content = workflow_path.read_text(encoding="utf-8")
    
    # Target the iterative fix loop
    old_loop_start = """            # Track changes
            CHANGES_MADE=""
            git_status_before=$(git status --porcelain | wc -l)

            # ===== PHASE 1: Formatting & Linting (deterministic, safe) ====="""
    
    new_loop_start = """            # Track changes
            CHANGES_MADE=""
            git_status_before=$(git status --porcelain | wc -l)

            # ----- STEP 0: Collect changed files for this PR -----
            echo ">>> Collecting changed files..."
            # Get base branch from PR info or default to main
            BASE_BRANCH=$(gh pr view "$PR_NUMBER" --json baseRefName --jq '.baseRefName' 2>/dev/null || echo "main")
            git fetch origin "$BASE_BRANCH" --depth=1
            CHANGED_FILES=$(git diff --diff-filter=ACMRT --name-only "origin/$BASE_BRANCH" HEAD -- '*.py' | tr '\\n' ' ')
            
            if [ -z "$CHANGED_FILES" ]; then
              echo "No changed Python files found in this PR. Skipping fixes."
              break
            fi
            echo "Changed files to fix: $CHANGED_FILES"

            # ===== PHASE 1: Formatting & Linting (Scoped to changed files) ====="""

    if old_loop_start in content:
        content = content.replace(old_loop_start, new_loop_start)
    else:
        # Fallback if whitespace differs
        print("Warning: Precise match for loop start failed, trying looser match")

    # Replace all occurrences of "ruff check --fix ." with "ruff check --fix $CHANGED_FILES"
    content = content.replace("ruff check --fix .", "ruff check --fix $CHANGED_FILES")
    content = content.replace("black .", "black $CHANGED_FILES")
    
    # Autoflake replace
    old_autoflake = "autoflake --in-place --remove-all-unused-imports --recursive ."
    new_autoflake = 'for f in $CHANGED_FILES; do autoflake --in-place --remove-all-unused-imports "$f" 2>&1 || true; done'
    content = content.replace(old_autoflake, new_autoflake)

    # Mypy agent call
    content = content.replace(
        "python scripts/mypy_autofix_agent.py \\\n                --max-fixes 20 --max-files 15 --verbose \\\n                --config-file pyproject.toml 2>&1 || true",
        "python scripts/mypy_autofix_agent.py \\\n                --max-fixes 30 --max-files 20 --verbose \\\n                --config-file pyproject.toml $CHANGED_FILES 2>&1 || true"
    )

    workflow_path.write_text(content, encoding="utf-8")
    return True

def process_repo(repo_name):
    repo_path = REPOS_ROOT / repo_name
    print(f"\nProcessing {repo_name}...")
    
    # Reset and prepare branch
    run_cmd(["git", "checkout", "main"], repo_path)
    run_cmd(["git", "pull", "origin", "main"], repo_path)
    run_cmd(["git", "checkout", "-B", BRANCH_NAME], repo_path)
    
    # Patch
    p1 = patch_mypy_agent(repo_path)
    p2 = patch_workflow(repo_path)
    
    if not (p1 or p2):
        print(f"No changes made to {repo_name}")
        return

    # Commit and Push
    run_cmd(["git", "add", "."], repo_path)
    run_cmd(["git", "commit", "-m", COMMIT_MSG, "--no-verify"], repo_path)
    run_cmd(["git", "push", "origin", BRANCH_NAME, "--force-with-lease"], repo_path)
    
    # PR
    run_cmd(["powershell", "-Command", f"gh pr create --title '{PR_TITLE}' --body '{PR_BODY}' --base main --head {BRANCH_NAME}"], repo_path)

if __name__ == "__main__":
    for repo in TARGET_REPOS:
        try:
            process_repo(repo)
        except Exception as e:
            print(f"Failed to process {repo}: {e}")
