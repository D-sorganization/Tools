# AGENT_HANDOFF_TEMPLATE — copy to src/&lt;your_tool&gt;/AGENT_HANDOFF.md

Copy this file to `src/<your_tool>/AGENT_HANDOFF.md` and fill in every
section from the actual current state of the tool — do not leave
placeholders, and do not write a changelog (history lives in git; this file
is current-state only). Keep it under 150 lines. Update it as part of every
PR that touches this tool and every push that lands on `main`.

Delete this instructional preamble once you copy the file.

---

# AGENT_HANDOFF — &lt;tool_name&gt;

> **Update this file with every PR and every push to main.**
> Last updated: &lt;YYYY-MM-DD&gt;

## Where This Tool Is Headed

One or two paragraphs: what is this tool, what is its current development
direction, and what epic(s)/issue(s) — if any — are actively driving it
right now. Link the epic numbers. If no epic is currently open against this
tool, say so explicitly and describe its role (e.g. "stable, maintenance
only" or "shared dependency for &lt;other tool&gt;").

## Recent Activity (grounding)

Summarize `git log --oneline -15 -- src/<your_tool>` in a sentence or two —
what kind of work has actually been landing (features vs. maintenance vs.
consolidation). List any open PRs that touch this tool
(`gh pr list --search "<tool> in:title,body"`).

## Must-Read Architecture Pointers

List 3-5 actual file paths a new agent should read before making changes,
with one clause each on why:

1. `src/<your_tool>/README.md` — ...
2. `src/<your_tool>/<key_module>.py` — ...
3. ...
4. ...
5. ...

## Gate Commands (this tool)

The exact commands to run before opening a PR — copy from CI config, don't
guess:

```bash
python3 -m pytest src/<your_tool> -n auto --timeout=60
python3 -m ruff check src/<your_tool>
python3 -m mypy src/<your_tool>
# web mirror, if it has one:
cd src/<your_tool>/web && npm run test && npx tsc --noEmit
```

## Do-Not List

Concrete traps found while working in this tool — dependency arrows that
must not be crossed, files with 500-LOC budgets, deprecated modules,
contract-frozen functions, anything a past agent got burned by. Not generic
advice — specific to this tool.

- Do not ...
- Do not ...

## Roadmap (ordered)

Numbered, ordered near-term steps — not an aspirational wishlist, the actual
next things to do:

1. ...
2. ...
3. ...
