# CLI Adapter Discovery & Subscription Qualification Runbook (`Tools #5727` Slice 9)

This runbook documents Tier-2 (no-model-call discovery) and Tier-3 (single-turn subscription qualification) verification for the subscription CLI adapters (`claude_code`, `codex_cli`, `agy`).

## Policy

1. **Subscription login only** — qualification runs MUST authenticate via interactive CLI subscription login (`claude login`, `codex login`, `agy`). Any run with `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY` set in the environment is **invalid by definition** (`validate_qualification_record` rejects it).
2. **Never run live model turns in PR CI** — Tier-1 fake-executable process tests and Tier-2 `--version`/`--help` probes run deterministically; Tier-3 single-turn qualification runs only manually or on an explicitly credentialed host.

## Tier-2: Discovery Probe (`--version` / `--help` Only)

Run against `DEFAULT_CLI_DISCOVERY_SPECS`:

- `claude_code` (`claude`): checks `-p`, `--model`
- `codex_cli` (`codex`): checks `exec`, `--skip-git-repo-check`, `--model`
- `agy` (`agy`): checks `--skip-trust`, `-p`, `--model`

```python
from shared.python.ai.adapters.cli_qualification import (
    DEFAULT_CLI_DISCOVERY_SPECS,
    probe_cli_provider_discovery,
)

for spec in DEFAULT_CLI_DISCOVERY_SPECS:
    result = probe_cli_provider_discovery(
        spec.provider, spec.binary_name, spec.required_help_flags
    )
```

## Tier-3: Single-Turn Subscription Qualification Record

Verify that no forbidden API-key environment variables are set, execute one short single-turn prompt, compute the SHA-256 of the captured transcript, and validate the resulting `CliQualificationRecord` with `validate_qualification_record(record, env=os.environ)`.
