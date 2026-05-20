# UI Refactoring Review: Layouts, Sidekick, and Sidebar

**Date**: 2026-05-17
**Repositories**: `Tools`, `UpstreamDrift`

This document details the architectural review of the UI enhancements applied to the Sidekick Chat Interface, Sidekick docking behavior, and Launcher Global Sidebar.

## Test-Driven Development (TDD)

- **Signal/Slot Contracts**: The chat buttons (Send, Steer, Stop) maintain strict signal connections to their corresponding handler methods (`_on_send`, `_on_steer`, `_on_stop_agent`). These paths can be unit-tested seamlessly without reaching into the layout structure.
- **Visual Boundaries**: Refactoring the UI layout (e.g., QSplitter bounds and QTabWidget elision) was treated as satisfying a visual contract. `UnifiedToolsSidebar.minimumSizeHint` acts as a programmable layout seam that tests can assert against (`assert sidebar.minimumSizeHint().width() == 100`).

## Design by Contract (DbC)

- **Minimum Width Enforcement**: `QSplitter` enforces a hard contract on its children: it will not resize a child smaller than `minimumSizeHint()`. The previous `ChatDockWidget` layout broke the expected contract by yielding a highly constrained width (~369px) due to its horizontal input row containing rigid elements.
- **Contract Fulfillment**: By explicitly overriding `minimumSizeHint()` to `QSize(100, 0)` in `UnifiedToolsSidebar`, we fulfill the `QSplitter` contract allowing dynamic compression, shifting the responsibility of clipping or reflowing to the internal tab contents.

## Law of Demeter (LoD)

- **Layout Encapsulation**: The changes isolate logic appropriately. `launcher_ui_setup.py` does not manipulate the internal layout policies of `UnifiedToolsSidebar`. Instead, `UnifiedToolsSidebar` encapsulates its own `minimumSizeHint`.
- **Global Sidebar**: The menu actions directly invoke bounded methods on `self` (`self.open_layout_manager()`) rather than reaching deeply into `LayoutManagerDialog` configurations from the menu setup.

## Don't Repeat Yourself (DRY)

- **Chat Layout Comboboxes**: Redundant `setMinimumWidth(50)` calls were consolidated to `0` and generalized to avoid boilerplate sizing rules. The `_build_header_combobox` method in `ChatDockWidget` sets sizes centrally.
- **Stylesheet Consistency**: The hover state for the global sidebar leverages `colors.border_default` and `colors.bg_highlight` fetched dynamically from the `theme_manager`, avoiding hardcoded hex strings and ensuring DRY compliance with the broader fleet-wide theming system.

## Summary of Accomplishments

1. Refactored the Chat Sidekick input row layout to align standard AI interaction patterns.
2. Rectified the `QSplitter` sidekick pane 40% width limitation by establishing a formal `minimumSizeHint` boundary.
3. Repaired the disabled state of `Select Visible Tiles...` by decoupling it from the layout editing mode toggle.
4. Added modernized CSS hover styles to the Global Sidebar tabs.
