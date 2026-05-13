# Shared Chat Contract

Tools owns the reusable chat and AI package surface for downstream products.
Product repositories such as UpstreamDrift and Gasification_Model should import
the public facade from `chat` after adding Tools to `PYTHONPATH`, installing the
editable package, or vendoring a pinned Tools checkout.

## Public Facade

Consumers may import these symbols from `chat`:

- `ChatDockWidget`
- `ChatMessageBubble`
- `ChatMessageRequest`
- `ChatChunkResponse`
- `ChatSessionInfo`
- `ChatHistoryResponse`
- `ChatModelInfo`
- `ChatModelListResponse`
- `ChatIndexStatusResponse`
- `TerminalShellInfo`
- `TerminalAgentProviderInfo`
- `TerminalAgentSessionRequest`
- `TerminalAgentSessionInfo`
- `TerminalAgentEvent`
- `TerminalProviderRegistry`
- `TerminalRegistryError`
- `ResponseStyle`
- `DEFAULT_RESPONSE_STYLE`
- `RESPONSE_STYLE_PROMPTS`
- `style_prompt`
- `ChatServiceBase`
- `ChatSession`
- `ChatMessage`
- `create_chat_router`

Modules under `chat._*` and implementation modules such as
`chat._chat_dock_widget_qt` are private. Consumer repositories should use local
adapters only for product-specific routing, settings, and window integration.

## Install And Vendor Matrix

| Consumer                | Supported source                                              | Required install surface                                    | Notes                                                                                                 |
| ----------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| UpstreamDrift           | `vendor/ud-tools` pinned to a Tools commit                    | `src/shared/python` on `PYTHONPATH` plus chat dependencies  | Product code should import `chat`, not copy chat implementation files.                                |
| Gasification_Model      | Tools checkout or package dependency pinned to a Tools commit | `pip install -e .[chat]` or equivalent runtime dependencies | Adapters may provide domain prompts and route wiring, but the widget/session contract stays in Tools. |
| Tools local development | Repository root checkout                                      | `pip install -e .[chat,dev]`                                | Focused contract tests live under `tests/shared/python/chat/`.                                        |

The `chat` extra installs the contract models, FastAPI router dependency, and
Qt dock widget dependency. Headless services that only need data models can
depend on `pydantic>=2.0.0` and import model symbols from `chat` when the
facade is importable.

## Compatibility Rules

- Keep the canonical shared implementation in `src/shared/python/chat`.
- Add facade exports before downstream repositories consume new chat payloads.
- Preserve package-relative imports so vendored and editable installs both work.
- Add or update contract tests whenever the public facade changes.
- Do not add a second shared chat implementation under another Tools package.

## Terminal-Agent Contract

Terminal-agent mode is an optional shared-chat extension for users who prefer
Claude Code, Codex, Cline CLI, Gemini CLI, or future terminal-native agents. It
keeps shell runtime selection separate from agent provider selection:

- Shell descriptors describe where the command runs, such as PowerShell, Bash,
  or WSL.
- Provider descriptors describe which agent CLI is launched inside the selected
  shell.
- Session requests carry the app context, resolved project root, shell id, and
  provider id.
- Session events normalize terminal output into stdout, stderr, status, exit,
  error, and auth-required payloads.

Downstream applications should populate dropdowns from `TerminalProviderRegistry`
instead of copying provider lists into product UI code. Runtime implementations
must launch providers with a validated project root as the process working
directory and may pass explicit Tools context through environment variables or a
generated context file. Secrets must stay out of command strings, diagnostics,
and persisted UI selections.
