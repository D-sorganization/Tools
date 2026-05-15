"""Draft implementations not yet wired to production UI.

These modules contain scaffolding for features that are complete at the
code level but have not been connected to any GUI entry point.  They are
preserved here so the work is not lost, but they are intentionally
**excluded from the public ``ai`` package API**.

Do **not** import from this sub-package in production code.  When a draft
is ready to graduate, open a follow-up issue, wire it to the GUI, add
contract tests, and move it back to the parent package.

Issue #2760: WorkflowEngine and workflow_definitions moved here (Option C).
Status: preserved, unwired.  Candidate for future GUI integration.
"""
