# WorkflowEngine — Drafts

This directory holds the unwired `WorkflowEngine` and `workflow_definitions`
that previously lived in `src/shared/python/ai/`. They were moved here per
[issue #2760](https://github.com/D-sorganization/Tools/issues/2760) (Option C).

## Why this is in `drafts/`

The engine is fully implemented (~1,300 lines across the two files) but
**no production code path instantiates it**. `ConversationContext.active_workflow_id`
is declared but never assigned. Shipping ~1.3k lines of dead code in production
wheels is a maintenance liability and a source of confusion for new contributors,
so the work has been parked here pending an explicit wiring decision.

The code itself is solid — it has step-level validation, multiple recovery
strategies, and a small library of canned workflows. The reason it is not in
`src/` is purely a wiring/UX gap, not a code-quality gap.

## What full wiring would require (Option A)

If/when the team decides to wire this into the AI assistant panel, the work
breaks down roughly as follows:

1. **Lifecycle owner.** Pick a controller in `src/shared/python/ai/gui/` to
   own a `WorkflowEngine` instance. The recently-decomposed `AIAssistantPanel`
   already has an `IndexingController`-style split — a sibling
   `WorkflowController` is the natural home.

2. **Workflow picker UI.** A user-visible affordance to start a workflow
   (toolbar button, slash-command, or context-menu entry on the chat input)
   that lists workflows from `workflow_definitions.py`.

3. **Context binding.** When a workflow starts, set
   `ConversationContext.active_workflow_id = execution.id`, and clear it on
   `COMPLETED` / `FAILED` / `ABORTED`. Persist it across session reloads via
   the existing `to_dict` / `from_dict` round-trip in `types.py`.

4. **`ASK_USER` callback.** `RecoveryStrategy.ASK_USER` requires a UI
   prompt. Wire it to a Qt input dialog (modal `QInputDialog` or a custom
   side-panel widget), and route the user's choice back into the engine on
   the GUI thread.

5. **Progress surface.** Stream step status (`PENDING` → `RUNNING` →
   `COMPLETED`/`FAILED`) into the chat transcript so the user can see what
   the engine is doing. Educational hints from each step's `learning_notes`
   should render inline.

6. **Tests.** At minimum:
   - One end-to-end test that runs a 3-step workflow against a fake
     `ToolRegistry` and asserts terminal state.
   - One test that exercises `ASK_USER` with a mocked dialog.
   - A contract test that the picker UI lists every workflow exported from
     `workflow_definitions.py`.

Realistic estimate: a focused day of UX + a focused day of test plumbing.
Better tackled as its own PR than tangled with the chat-hardening epic that
the assistant panel decomposition just landed.

## Files

| File                      | Purpose                                                                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `workflow_engine.py`      | `WorkflowEngine`, `Workflow`, `WorkflowStep`, `WorkflowExecution`, `StepStatus`, `RecoveryStrategy`, `ValidationResult`, `StepResult`. |
| `workflow_definitions.py` | Canned workflows: C3D import, cross-engine validation, drift-control decomposition, first analysis, inverse dynamics.                  |

## If you want to bring this back

1. `git mv drafts/workflow_engine/workflow_engine.py src/shared/python/ai/workflow_engine.py`
2. `git mv drafts/workflow_engine/workflow_definitions.py src/shared/python/ai/workflow_definitions.py`
3. Restore the lazy entries and `__all__` lines in `src/shared/python/ai/__init__.py`
   (see git history for the exact block).
4. Add the wiring described above and the accompanying tests.
5. Delete this directory.
