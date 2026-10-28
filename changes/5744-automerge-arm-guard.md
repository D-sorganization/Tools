---
issue: 5744
summary: "drop raw gh pr merge --auto instructions from fleet-managed blocks and docs"
---

Syncs updated fleet-managed blocks from Repository_Management (PR #2486) removing raw `gh pr merge --auto` instructions and tightens `tests/ops/test_agent_docs_automerge_arm.py` to assert no raw instructions across all blocks including fleet-managed blocks.
