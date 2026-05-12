# Epic: Chat Conversation Management & UI/UX Parity

## Overview
This epic tracks the stabilization, integration, and UI/UX modernization of the AI Chat interface across the D-sorganization fleet (Tools, UpstreamDrift, Gasification_Model). With the AI backend successfully migrated to Rust for enhanced performance, the frontend PyQt6 GUI must maintain strict parity, ensuring beautiful glassmorphic themes, functional settings, and proper categorization of tools and models in the launcher.

## Tasks
- [x] **Restore Chat UI Styling:** Restore the modernized, theme-aware chat panel that was inadvertently overwritten during the Rust backend squash merge (PR #2589).
- [x] **Fix Theme Inheritance:** Update `AIAssistantPanel` and `MessageWidget` to correctly extract theme colors from the updated `ThemeManager` dictionary output.
- [x] **Correct Initialization Order:** Fix crashes in `history_sidebar.py` caused by premature `_theme_colors` access.
- [x] **Relabel Expertise Levels:** Update the bottom selector / settings dialog to reflect verbosity-based labels (Verbose, Normal, Concise, Minimal) instead of generic levels (Beginner, Intermediate, Advanced, Expert).
- [x] **Fix Ollama UI Bug:** Ensure Ollama host and refresh inputs are strictly hidden for cloud models (Codex, Claude) in the settings dialog.
- [x] **Launcher Category Mapping (Biomechanics):** Move `mujoco`, `drake`, `opensim`, `pinocchio`, and `myosim` from "Physics Engines" to "Biomechanics".
- [x] **Launcher Category Mapping (Simulation):** Ensure `putting_green`, `golf_simulator`, and `aerodynamics_model` render under the "Simulation" section.
- [x] **Launcher Category Mapping (Tools & Data):** Ensure `video_processor` and `video_analyzer` render alongside `data_processor` in "Tools & Data".
- [ ] **Ollama Connectivity:** Investigate and resolve the underlying connection drop between the new `RustAgentAdapter` and local Ollama host.
- [ ] **Cross-Repo Re-Audit:** Review and reopen any prematurely closed chat-related issues in Gasification_Model and UpstreamDrift to ensure complete fleet parity.

## Associated Repositories
- Tools
- UpstreamDrift
- Gasification_Model
