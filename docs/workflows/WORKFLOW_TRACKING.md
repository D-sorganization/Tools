# Workflow Tracking Document

This document lists all active GitHub Workflows in this repository.

| Workflow Name | Filename | Status | Purpose |
| :--- | :--- | :--- | :--- |
| **Control Tower** | `Jules-Control-Tower.yml` | Active | Orchestrates agentic workers. |
| **PR Compiler** | `Jules-PR-Compiler.yml` | Active | Compiles PR info for fleet management. |
| **CI Standard** | `ci-standard.yml` | Active | Linting, Formatting, and Type checking. |
| **Python Tests** | `ci-fast-tests.yml` | Active | Runs unit and integration tests. |
| **Assessment Generator** | `Jules-Assessment-Generator.yml` | Active | Automated architecture & quality audits. |

---

## Maintenance
Update this document whenever a new workflow is added or the status of an existing workflow changes. For global standards, see `Repository_Management/docs/architecture/WORKFLOW_GOVERNANCE.md`.
