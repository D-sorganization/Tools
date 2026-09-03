# Documentation Index

This directory contains detailed documentation for the AffineDrift Tools repository.

## 🏗️ Architecture

- **[Jules Architecture](architecture/JULES_ARCHITECTURE.md)**: Describes the "Control Tower" CI/CD architecture, agent roles, and automated workflows.
- **[Modification Guidance](architecture/Modification_Guidance.md)**: Guide for upgrading and modifying the architecture and tools.
- **[Shared Chat Contract](architecture/SHARED_CHAT_CONTRACT.md)**: Public facade and compatibility matrix for Tools-owned chat/AI integrations.

## 💻 Development

- **[Guardrails & Guidelines](development/GUARDRAILS_GUIDELINES.md)**: Essential safety rules, linting configurations (Ruff, MyPy), and CI integration standards.
- **[Branching Workflow](development/BRANCHING_WORKFLOW_RULE.md)**: The mandatory feature-branch workflow and naming conventions.
- **[Launcher Improvements](development/LAUNCHER_IMPROVEMENTS_PR.md)**: Specific documentation related to the launcher enhancement initiative.

## 🛠️ Tools

- **[Tools Inventory & Platform Parity](tools/TOOLS_INVENTORY.md)**: Complete inventory of all applications with platform support matrix (PyQt6, Web, Tauri) and identified parity gaps.
- **[Enhanced Tools](tools/ENHANCED_TOOLS.md)**: Detailed overview and comparison of the "Pro" tools (Folder Fix Pro, Folder Packer Pro) versus their legacy counterparts.

## 🧪 Physics & Models

- **[Impact-Interval Club Dynamics](physics/IMPACT_INTERVAL_DYNAMICS.md)**: Formulation, publication boundary, and binding validation program for the six-DOF club/ball contact-interval solver (`src/shared/python/swing_sim/impact_interval/`).

## 📅 Release & History

- **[Changelog](release/CHANGELOG.md)**: Record of all notable changes to the project.
- **[Safe State](release/SAFE_STATE.md)**: Notes on the repository's safety state and recovery points.
- **[Closed-Stack Gap Audit](release/closed_stack_gap_audit.md)**: Evidence for Tools #4921 — which files from the folded golf-app PR stack (#4466, #4449 and their parent PRs) never reached `main`; regenerate with `python scripts/audit_closed_stack_branches.py` (JSON: `release/closed_stack_gap_audit.v1.json`).
