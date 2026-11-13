---
issue: 5727
summary: "Wire CLI adapters to sidekick cli_bridge and enforce ENABLED_BY_DEFAULT opt-in (Wave 2.3)"
---

- Wire production CLI adapters (`ClaudeCodeAdapter`, `CodexCliAdapter`, `GeminiCliAdapter`, `AntigravityCliAdapter`) to `shared.python.sidekick.agent.cli_bridge` via `src/shared/python/ai/adapters/cli_tool_bridge.py` (`build_cli_tool_prompt_section`, `parse_cli_tool_envelopes_with_diagnostics`, `run_goal_loop`), surfacing malformed/unknown/missing-param envelope diagnostics instead of silently dropping them.
- Enforce `ENABLED_BY_DEFAULT = False` before class instantiation and cache lookup in `AdapterFactory.create`, `get_best_available`, and `get_available_providers`.
- Add deterministic Tier-1 process and opt-in oracle tests (`tests/unit/ai/adapters/test_cli_bridge_and_optin_oracles.py`).
