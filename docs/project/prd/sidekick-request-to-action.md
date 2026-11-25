# PRD — Sidekick: from a user request to an executed, audited action

| Field        | Value                                                                    |
| ------------ | ------------------------------------------------------------------------ |
| Repository   | `D-sorganization/Tools`                                                  |
| Intent       | Epic #5727 (owner-approved panel `th_8820201930a3`); backlog plan #6023  |
| Owner answer | pending                                                                  |
| Author       | `product-owner` on provider `claude`                                     |
| Date         | 2026-11-16                                                               |

## Problem

Epic #5727 promises that a Sidekick user can ask a subscription CLI agent
(`claude`, `codex`, `agy`) to do something and see it done safely. Every Tools
slice the epic lists is closed (#5728–#5732), and the process-runner,
cancellation and opt-in work from plan #6023 waves 1.2, 2.2 and 2.3 has merged
(#6030, #6039, #6042, #6043). The user still cannot get from a request to an
action. The independent analysis on #5727 (comment 6761096943) found the chain
broken in four places, and on `main` at `099f897a8` all four still hold:

- The chat stream emits only text. Tool calls the model returns are dropped
  before they reach the UI (stage 6).
- `SidekickActionService` is built only in tests. No window composes it with a
  policy and an audit sink (stage 10).
- An attached file is sent as a `file_upload` action that the websocket router
  rejects. `tests/shared/python/chat/test_router_action_coverage.py` lists it as
  `KNOWN_UNHANDLED` (stage 3).
- `ai.types.Message` has no attachment field, so the model never sees the file
  (stage 4).

Each half is unit-tested against a stand-in for the other, so CI is green while
no end-to-end path exists. The epic body lists only finished slices, so nobody
has written down what "done" means for the rest.

## Users

- **Tools desktop users** working in the standalone Sidekick window. They want
  to ask the agent to inspect or drive a tool (list sub-tabs, run a calculator)
  and see what it did.
- **Host maintainers** in UpstreamDrift and Gasification_Model. They want one
  working reference composition in Tools to copy, not a library with no caller.
- **The owner.** They need to see that every action the agent takes was allowed
  by policy and recorded, under subscription logins only.

## Scope

1. A model's tool call reaches the Sidekick UI and becomes a planned action.
2. One reference composition in the standalone Sidekick window. It joins the
   stream, planner, policy, action service and audit sink, and registers the
   `subtab.*` read actions.
3. A policy and an audit sink must be supplied on purpose. Leaving them out is
   an error, never a silent "allow everything, record nothing".
4. Attachments are honest. Until the model can actually receive a file, the UI
   does not show the file as accepted.
5. Attached files reach the model turn for CLI and API providers.

## Non-Goals

- Host wiring in UpstreamDrift and Gasification_Model. Their own issues own the
  host end-to-end claim (UpstreamDrift #13660, Gasification_Model #5475/#5476).
- Packaging, PyInstaller or installer changes. The Sidekick packaging hold
  stands.
- Turning subscription CLI driving on by default. It stays opt-in until the
  owner signs off on consumer-subscription terms and rate limits.
- Write, shell or file-edit actions. This PRD covers read actions and the
  existing `subtab.*` / `canonical.*` set only.
- Paid API keys of any kind.
- Choosing the attachment storage format, transport or prompt layout. That is a
  design decision for the chief architect.

## Acceptance Criteria

Each item runs in ordinary CI with a fake CLI executable on `PATH` and no real
model call, unless it says otherwise.

1. **Text-only end to end.** A fake CLI returns one valid tool-call envelope for
   `subtab.list`. Driven through the real Sidekick panel, the run executes
   exactly one action, renders exactly one action chip and writes exactly one
   audit line.
2. **No dropped tool calls.** For a fake turn that returns N tool calls
   (N = 1, 2, 3), the UI receives N tool-call events. A test fails if any is
   dropped.
3. **Tool declarations don't crash.** With a CLI provider selected and at least
   one tool enabled, sending a message raises no exception. Today it raises
   `AttributeError`.
4. **Policy is required.** Building `SidekickActionService` without a policy
   argument raises `TypeError`. Leaving out the audit sink also raises
   `TypeError`. Tests use an explicit, named allow-all object.
5. **Refusals are audited.** A destructive action requested without user
   confirmation is not executed, the chip shows it as refused, and the audit log
   records the refusal.
6. **The reference composition exists.** `SidekickActionService` is constructed
   in at least one non-test module under `src/` (the standalone Sidekick
   window), and that module passes criterion 1.
7. **Honest attachment chip.** Until criterion 8 passes, attaching a file either
   is not offered or shows the backend's rejection on the chip. No state
   displays a file as accepted when the backend rejected it.
8. **Attachments reach the turn.** `file_upload` is removed from `KNOWN_UNHANDLED`
   in `test_router_action_coverage.py`. A fake CLI turn proves that a 1 KB text
   attachment's content (or a path the CLI is allowed to read) is present in the
   turn it receives. An API-provider test proves the attachment is in the
   request.
9. **Opt-in still holds.** With CLI driving not enabled, criteria 1 and 8 make
   no CLI process. A test proves that zero processes are spawned.
10. **Subscription only.** The manual Tier-3 qualification run (one real turn per
    provider on a logged-in machine) records CLI version, OS, exit code and a
    transcript hash. It fails if any provider API key is present in the
    environment. *Person-run; recorded on #5727.*

## Success Metric

The number of broken stages in the #5727 attachment-to-action flow (comment
6761096943, stages 3, 4, 6 and 10). It is **4** on `main` at `099f897a8`. The
target is **0**, shown by criteria 1 and 8 passing in CI.

## Open Questions

1. **Order and gating.** Analysis comment 6761105197 says the tool-call fix
   (criteria 2–3) should land only together with, or after, the approval-gate
   prerequisites handled privately under #6003 and the required-policy change
   (criterion 4). Are those prerequisites resolved, so criteria 1–6 can start?
2. **Tools' end-to-end claim.** Is "library plus one reference composition in
   the standalone Sidekick window" the right boundary for Tools, with the host
   repos owning their own end-to-end claim? Or must Tools also prove host
   integration?
3. **Attachments now or later.** Should criteria 7–8 (attachments) be in this
   PRD's definition of done, or split out after text-only tool use ships?
