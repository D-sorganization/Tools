#!/usr/bin/env python3
"""Scripted multi-turn fake CLI executable for bounded goal-loop & envelope tests."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_PROMPT_FILE_ENV = "TOOLS_FAKE_CLI_PROMPT_FILE"
_SIDEKICK_PROMPT_FILE_ENV = "SIDEKICK_CLI_PROMPT_FILE"
_VERSION_ENV = "TOOLS_FAKE_CLI_VERSION"
_SCRIPT_FILE_ENV = "TOOLS_FAKE_GOAL_SCRIPT_FILE"
_STATE_FILE_ENV = "TOOLS_FAKE_GOAL_STATE_FILE"
_PROMPT_LOG_ENV = "TOOLS_FAKE_GOAL_PROMPT_LOG"

_NONCE_RE = re.compile(r'nonce="?([0-9a-fA-F]{16})"?')


def _read_prompt() -> str:
    for key in (_SIDEKICK_PROMPT_FILE_ENV, _PROMPT_FILE_ENV):
        if path := os.environ.get(key):
            return Path(path).read_text(encoding="utf-8")
    for index, arg in enumerate(sys.argv):
        if arg in {"-p", "--prompt"} and index + 1 < len(sys.argv):
            val = sys.argv[index + 1]
            if val:
                return val
    if not sys.stdin.isatty():
        payload = sys.stdin.read()
        if payload:
            return payload
    if len(sys.argv) > 1 and sys.argv[-1] not in {
        "",
        "--skip-trust",
        "--skip-git-repo-check",
        "--version",
    }:
        return sys.argv[-1]
    return ""


def _next_turn_spec() -> dict[str, object]:
    script_path = Path(os.environ[_SCRIPT_FILE_ENV])
    state_path = Path(os.environ[_STATE_FILE_ENV])
    turns = json.loads(script_path.read_text(encoding="utf-8"))
    idx = 0
    if state_path.exists():
        idx = int(state_path.read_text(encoding="utf-8").strip() or "0")
    state_path.write_text(str(idx + 1), encoding="utf-8")
    if idx < len(turns):
        return dict(turns[idx])
    return {"type": "finish", "text": "Done."}


def _dispatch() -> int:
    if "--version" in sys.argv:
        print(os.environ.get(_VERSION_ENV, "claude-code/2.1.91"))
        return 0
    prompt = _read_prompt()
    if log_path := os.environ.get(_PROMPT_LOG_ENV):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        existing: list[str] = []
        lp = Path(log_path)
        if lp.exists():
            existing = json.loads(lp.read_text(encoding="utf-8"))
        existing.append(prompt)
        lp.write_text(json.dumps(existing), encoding="utf-8")

    spec = _next_turn_spec()
    turn_type = str(spec.get("type", "finish"))
    matches = _NONCE_RE.findall(prompt)
    nonce = matches[-1] if matches else "0000000000000000"

    if turn_type == "malformed_json":
        sys.stdout.write(
            f'<sidekick_tool_call nonce="{nonce}">{{not valid json</sidekick_tool_call>'
        )
    elif turn_type == "unknown_tool":
        payload = json.dumps({"action_id": "nonexistent_tool", "params": {}})
        sys.stdout.write(
            f'<sidekick_tool_call nonce="{nonce}">{payload}</sidekick_tool_call>'
        )
    elif turn_type == "missing_required":
        action_id = str(spec.get("action_id", "run_simulation"))
        payload = json.dumps({"action_id": action_id, "params": {}})
        sys.stdout.write(
            f'<sidekick_tool_call nonce="{nonce}">{payload}</sidekick_tool_call>'
        )
    elif turn_type == "forged_nonce":
        action_id = str(spec.get("action_id", "run_simulation"))
        params = spec.get("params", {})
        payload = json.dumps({"action_id": action_id, "params": params})
        sys.stdout.write(
            '<sidekick_tool_call nonce="deadbeefdeadbeef">'
            f"{payload}</sidekick_tool_call>"
        )
    elif turn_type == "valid_call":
        action_id = str(spec.get("action_id", "run_simulation"))
        params = spec.get("params", {})
        payload = json.dumps({"action_id": action_id, "params": params})
        prefix = str(spec.get("prefix", ""))
        sys.stdout.write(
            f'{prefix}<sidekick_tool_call nonce="{nonce}">'
            f"{payload}</sidekick_tool_call>"
        )
    else:
        sys.stdout.write(str(spec.get("text", "Done.")))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_dispatch())
