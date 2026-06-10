#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# GAAI Daemon Launcher for Golf_GAAI_Sandbox
#
# Run this script from WSL to start the autonomous overnight delivery daemon.
#
# Prerequisites (run once in WSL):
#   sudo apt install -y tmux nodejs npm
#   npm install -g @anthropic-ai/claude-code
#   claude --version   # verify claude is in PATH
#
# Usage:
#   From Windows Git Bash or WSL:
#     bash start-gaai-daemon.sh            # interactive (status output)
#     bash start-gaai-daemon.sh --dry-run  # show what would launch, no action
#     bash start-gaai-daemon.sh --status   # check active sessions
#     bash start-gaai-daemon.sh --yes      # also set the GLOBAL settings key
#
# Agent-safety note:
#   This launcher sets "skipDangerousModePermissionPrompt": true so the daemon
#   can run unattended. By default it writes this ONLY to a project-scoped file
#   (.claude/settings.local.json); your global ~/.claude/settings.json is left
#   untouched unless you pass --yes. Existing keys are always preserved (merge,
#   not overwrite) and a timestamped backup is made.
#   To revert: delete the "skipDangerousModePermissionPrompt" key from
#   .claude/settings.local.json (and ~/.claude/settings.json if you used --yes),
#   restoring from the *.bak.<timestamp> file if needed.
#
# Monitor overnight:
#   tmux attach -t gaai-daemon
#   (Ctrl+B then D to detach without stopping)
#
# View individual story logs:
#   tail -f .gaai/project/contexts/backlog/.delivery-logs/E01S01.log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_SCRIPT="$REPO_ROOT/.gaai/core/scripts/delivery-daemon.sh"

# ── Preflight checks ──────────────────────────────────────────────────────────
check_dependency() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: '$1' not found. Install with: $2"
    exit 1
  fi
}

check_dependency "tmux"   "sudo apt install tmux -y"
check_dependency "claude" "npm install -g @anthropic-ai/claude-code"
check_dependency "git"    "sudo apt install git -y"

# ── Parse our own flags up front (so --dry-run truly changes nothing) ─────────
# Recognized: --dry-run (no filesystem/git changes), --status (query only),
#             --yes (allow modifying the GLOBAL ~/.claude/settings.json).
ARGS="${*}"
DRY_RUN=false
ASSUME_YES=false
case " $ARGS " in
  *" --dry-run "*) DRY_RUN=true ;;
esac
case " $ARGS " in
  *" --yes "*) ASSUME_YES=true ;;
esac

# ── Suppress dangerous-mode permission prompt (NON-DESTRUCTIVE) ────────────────
# This used to clobber the user's entire global ~/.claude/settings.json with a
# single key, destroying their permissions/hooks/env, and silently weakened the
# agent sandbox for every future session (issue #3291).
#
# Behaviour now:
#   * Prefer a PROJECT-scoped setting (.claude/settings.local.json in this repo),
#     which takes precedence over the global file and is the supported mechanism.
#   * Never overwrite an existing file: merge the single key with jq and write a
#     timestamped backup first.
#   * Honour --dry-run (print intended change, touch nothing).
#   * Modifying the GLOBAL settings file requires explicit --yes consent.
SETTINGS_KEY="skipDangerousModePermissionPrompt"

merge_setting() {
  # merge_setting <path>  — idempotently set {SETTINGS_KEY: true} without
  # discarding any existing content. Creates a timestamped backup first.
  local target="$1"
  if [[ -f "$target" ]] && grep -q "$SETTINGS_KEY" "$target" 2>/dev/null; then
    echo "  -> $SETTINGS_KEY already configured in $target (no change)."
    return 0
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "  [dry-run] would set $SETTINGS_KEY=true in $target (existing keys preserved)."
    return 0
  fi
  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" ]]; then
    cp -p "$target" "$target.bak.$(date +%Y%m%d%H%M%S)"
    if command -v jq &>/dev/null; then
      local tmp
      tmp="$(mktemp)"
      jq ". + {\"$SETTINGS_KEY\": true}" "$target" > "$tmp" && mv "$tmp" "$target"
    else
      echo "  -> WARNING: jq not found; leaving $target untouched to avoid data loss."
      echo "     Install jq, or manually add \"$SETTINGS_KEY\": true to $target."
      return 0
    fi
  else
    printf '{\n  "%s": true\n}\n' "$SETTINGS_KEY" > "$target"
  fi
  echo "  -> Set $SETTINGS_KEY=true in $target (existing keys preserved)."
}

PROJECT_SETTINGS="$REPO_ROOT/.claude/settings.local.json"
GLOBAL_SETTINGS="$HOME/.claude/settings.json"

echo "Configuring claude to skip dangerous-mode permission prompt (project-scoped)..."
merge_setting "$PROJECT_SETTINGS"

if [[ "$ASSUME_YES" == true ]]; then
  echo "Applying $SETTINGS_KEY to the GLOBAL settings file (--yes given)..."
  merge_setting "$GLOBAL_SETTINGS"
else
  echo "  Note: global $GLOBAL_SETTINGS left untouched."
  echo "        Pass --yes to also set it globally (this weakens the sandbox for"
  echo "        ALL future Claude Code sessions on this machine)."
fi

# ── Ensure we're on staging branch (skipped on --dry-run) ─────────────────────
cd "$REPO_ROOT"
if [[ "$DRY_RUN" == true ]]; then
  echo "[dry-run] skipping branch switch and git pull."
else
  CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$CURRENT_BRANCH" != "staging" ]]; then
    echo "WARNING: Not on staging branch (currently on '$CURRENT_BRANCH')."
    echo "Switching to staging..."
    git checkout staging
    git pull --ff-only origin staging
  fi

  # ── Pull latest before starting ────────────────────────────────────────────
  echo "Fetching latest from origin/staging..."
  git fetch origin
  git pull --ff-only origin staging || echo "Note: fast-forward failed (may have local changes)"
fi

# ── Pass through arguments to daemon ─────────────────────────────────────────
if [[ "$ARGS" == *"--status"* ]]; then
  bash "$DAEMON_SCRIPT" --status
  exit 0
fi

if [[ "$ARGS" == *"--dry-run"* ]]; then
  echo "DRY RUN — stories that would be launched:"
  bash "$DAEMON_SCRIPT" --dry-run
  exit 0
fi

# ── Launch daemon in tmux session ─────────────────────────────────────────────
SESSION_NAME="gaai-daemon"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "GAAI daemon session '$SESSION_NAME' is already running."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  tmux attach -t "$SESSION_NAME"
  exit 0
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║       GAAI Autonomous Delivery Daemon            ║"
echo "║       Golf_GAAI_Sandbox — Overnight Run          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Starting daemon in tmux session '$SESSION_NAME'..."
echo "Concurrent stories: 2 (conservative for overnight run)"
echo "Poll interval: 60 seconds"
echo ""

# Launch daemon: 2 concurrent max for overnight safety, 60s poll
tmux new-session -d -s "$SESSION_NAME" \
  "bash $DAEMON_SCRIPT --max-concurrent 2 --interval 60; read -p 'Daemon exited. Press Enter to close.'"

echo "Daemon started in tmux session '$SESSION_NAME'."
echo ""
echo "Useful commands:"
echo "  tmux attach -t $SESSION_NAME           # watch live"
echo "  Ctrl+B then D                           # detach without stopping"
echo "  bash start-gaai-daemon.sh --status      # check what's running"
echo "  tail -f .gaai/project/contexts/backlog/.delivery-logs/*.log  # story logs"
echo ""
echo "Attaching now... (Ctrl+B D to detach)"
sleep 2
tmux attach -t "$SESSION_NAME"
