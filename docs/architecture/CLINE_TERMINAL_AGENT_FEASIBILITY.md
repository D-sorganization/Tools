# Cline Terminal-Agent Feasibility Spike

Issue: [Tools #2619](https://github.com/D-sorganization/Tools/issues/2619)

## Purpose

Tools is standardizing shared chat around reusable contracts and terminal-agent
providers. This spike evaluates three ways to make Cline available from that
shared chat surface:

1. Wrap the Cline CLI as a terminal provider.
2. Build a richer native adapter on top of the Cline SDK.
3. Embed the full VS Code-style Cline user interface inside Tools.

The decision in this document is intentionally scoped to architecture. It does
not add runtime code, tests, provider wiring, or UI changes.

## Current Inputs

- The shared chat contract is owned by Tools and must remain the canonical
  reusable surface for downstream products. Consumers should import the public
  `chat` facade instead of copying a second implementation.
- Cline's current CLI documentation describes interactive terminal sessions,
  headless workflows, `cline auth`, JSON output, and global flags such as
  `--cwd`, `--config`, and `--data-dir`.
- Cline's current SDK documentation describes `@cline/sdk` as an open source
  TypeScript package for embedding the same agent runtime used by the CLI and
  IDE extensions. It requires a Node runtime and exposes packages such as
  `@cline/core`, `@cline/agents`, `@cline/llms`, and `@cline/shared`.
- The upstream Cline repository is published as Apache-2.0 and contains
  separate surfaces for CLI, SDK, VS Code extension, webview UI, and runtime
  files.
- Tools currently has older Cline adapter references that assume a local
  IDE/server style integration. Those assumptions do not satisfy the terminal
  provider roadmap on their own.

## Option 1: Cline CLI Provider

### Shape

Tools launches `cline` through the shared terminal runtime. The provider
descriptor owns command construction and lightweight probes, while Cline owns
its agent execution, provider authentication, task history, and local state.

Likely command shape:

```text
cline --cwd <project-root> --json --config <settings-dir> --data-dir <state-dir> <prompt>
```

The first implementation should probe capability without mutating the user's
workspace:

- `cline --version` or equivalent availability check.
- `cline --help` / command help check for supported flags.
- Auth/setup diagnostic that reports "auth required" without reading or logging
  provider secrets.
- JSON-mode smoke check only when it can be run against a harmless prompt and
  bounded timeout.

### Benefits

- Lowest integration risk because Tools can reuse its terminal provider
  boundary instead of embedding Cline internals.
- Cross-platform fit is better than editor embedding because the CLI is already
  a terminal product.
- `--cwd` maps directly to Tools' resolved project root contract.
- `--json` gives a path for structured output once the terminal runtime can
  parse provider events.
- `--config` and `--data-dir` make isolated app/project state possible without
  requiring Tools to parse Cline's private config files.
- Failure modes are easy to present as typed unavailable states: missing
  Node/npm, missing `cline`, unauthenticated provider, unsupported flags, or
  timeout.

### Risks

- CLI behavior and JSON schema are external contracts that may drift with Cline
  releases. Tools should probe capabilities at runtime and keep tests focused
  on command construction and diagnostic handling, not on an assumed full
  upstream schema.
- Cline can edit files and run commands. Tools must make the launch mode and
  approval posture visible before enabling autonomous actions.
- Using global Cline state may surprise users; using isolated state may require
  extra setup. The MVP should default to user-global state only when the user
  explicitly selects that behavior, and otherwise prefer a Tools-managed
  `--data-dir` for project/app isolation.

### Fit

Recommended for MVP.

## Option 2: Cline SDK Adapter

### Shape

Tools hosts a Node sidecar or integration package that imports `@cline/sdk` and
bridges SDK events into the shared chat contract. The Python/Qt application
would talk to that sidecar over a narrow local protocol rather than invoking the
CLI directly.

### Benefits

- Better long-term access to structured events, tool permissions, model
  configuration, session state, and custom integration points.
- More natural fit for a native shared chat experience once Tools has a stable
  terminal-agent abstraction and enough test coverage around provider events.
- Avoids parsing terminal output when the SDK event stream can be modeled
  directly.

### Risks

- Adds a Node runtime boundary to a primarily Python/Qt shared chat flow.
- Requires explicit protocol design between Python and Node: lifecycle,
  cancellation, streaming, permissions, errors, state paths, logging, and
  version compatibility.
- Requires a dedicated security review for SDK tools that can read/write files,
  execute commands, fetch web content, and use MCP servers.
- The SDK may be materially better than CLI wrapping, but that needs evidence
  from a follow-up spike with a small sidecar prototype.

### Fit

Good follow-up spike after the CLI provider exists. Not the MVP path.

## Option 3: Full VS Code-Style UI Embedding

### Shape

Tools attempts to embed the Cline IDE experience, including the sidebar/webview
UI and editor-grade interaction model, inside a PyQt chat dock.

### Benefits

- Highest potential fidelity to the existing Cline extension experience.
- Could expose rich approvals, diffs, checkpoints, and settings if all
  supporting editor surfaces were recreated cleanly.

### Risks

This option is deferred because the VS Code-style experience depends on more
than a chat panel. A faithful embedding needs a substantial extension-host and
editor surface, including:

- VS Code extension activation and command routing.
- Webview lifecycle, message passing, persistence, and asset loading.
- Workspace filesystem APIs and project trust boundaries.
- Integrated terminal APIs, process lifecycle, environment handling, and shell
  selection.
- Diff editors, diagnostics, checkpoints, and file-revert affordances.
- Settings storage, secrets handling, provider configuration, and account/auth
  flows.
- Human approval flows for file edits, commands, browser actions, MCP tools,
  and potentially destructive operations.

Rehosting those dependencies in Tools would likely mean embedding a
VS Code-compatible extension host or forking/adapting large UI/runtime pieces.
That is not proportional to the shared chat terminal-agent MVP.

### Fit

Deferred. Revisit only if a later product requirement demands full Cline UI
fidelity and a separate spike proves the extension-host surface can be isolated
without forking the upstream UI/runtime.

## Recommendation

Use the Cline CLI provider for the MVP.

The MVP should treat Cline like a terminal agent selected from shared chat. Tools
should build a provider descriptor that constructs `cline` commands, probes
availability/auth without leaking secrets, resolves `--cwd` from the selected
project root, and supports optional `--config`, `--data-dir`, provider/model,
timeout, and JSON-mode flags when the installed CLI advertises them.

Open a follow-up SDK spike only after the CLI provider is stable. The SDK spike
should answer whether a Node sidecar gives enough event fidelity, permission
control, and structured state handling to justify the extra runtime boundary.

Do not pursue full UI embedding for the MVP. The VS Code extension-host,
webview, workspace, terminal, diff, settings, and approval dependencies make it
too large and too brittle for the terminal-agent roadmap slice.

## Implementation Implications

### TDD

Future implementation should start with tests for provider behavior before UI
work:

- Command descriptor construction includes `--cwd`, optional provider/model,
  optional `--config`, optional `--data-dir`, optional `--json`, and timeout.
- Missing Node/npm/Cline returns a typed unavailable state.
- Auth-needed diagnostics do not include provider API keys, config file
  contents, tokens, or raw secret-bearing command output.
- Unsupported CLI flags degrade gracefully through capability probes.
- Terminal launch uses the generic terminal runtime and can be canceled.

### DbC

Provider contracts should be explicit:

- Precondition: launch requires a resolved project root and selected shell.
- Precondition: provider/model/config/data-dir values must be normalized before
  command construction.
- Invariant: Tools never assumes Cline global config exists; auth status must be
  probed.
- Invariant: provider diagnostics never expose secrets.
- Postcondition: unavailable states are typed and user-actionable.
- Postcondition: generated command arguments are represented as an argv list,
  not a shell-concatenated string.

### DRY

- Reuse the generic provider probe and terminal runtime from the shared
  terminal-agent architecture.
- Keep Cline-specific flag logic inside one provider adapter/descriptor.
- Do not duplicate chat models, message bubbles, or session storage outside the
  existing `chat` facade.
- Do not fork Cline UI or SDK code into Tools for the MVP.

### LOD

- The shared chat dock should talk to a provider abstraction, not directly to
  Cline process details.
- The Cline provider should own Cline-specific flags and state paths, while the
  terminal runtime owns process lifecycle.
- Downstream applications should continue to depend on the Tools shared chat
  facade and product-specific routing only.

## License And Compliance Notes

- The upstream Cline repository is Apache-2.0. Invoking the installed CLI as an
  external user tool does not require vendoring its source, but Tools should
  document the external dependency and avoid bundling credentials or upstream
  binaries without an explicit packaging review.
- Using `@cline/sdk` introduces an npm dependency tree and a Node runtime
  boundary. A follow-up spike must record package versions, transitive license
  posture, update cadence, and security scanning expectations before shipping.
- Vendoring or forking Cline UI/runtime files would require preserving upstream
  notices and auditing all copied dependencies. This is another reason to defer
  full UI embedding.
- Provider credentials remain Cline-owned. Tools should not read, persist, or
  log Cline API keys; setup guidance should point users to `cline auth` or to
  the selected isolated Cline config/data directory.
- Logs and diagnostics must redact command output that may contain provider
  tokens, config paths with sensitive user names only if policy requires it, or
  model/provider secrets.

## Decision

Accepted for this spike: pursue Cline CLI provider first, spike Cline SDK later,
and defer full VS Code-style UI embedding.

## Sources Checked

- [Cline CLI overview](https://docs.cline.bot/usage/cli-overview)
- [Cline CLI reference](https://docs.cline.bot/cli/cli-reference)
- [Cline SDK overview](https://docs.cline.bot/sdk/overview)
- [cline/cline GitHub repository](https://github.com/cline/cline)
