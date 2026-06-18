/**
 * Single source of truth for the dashboard tab bar (#3546).
 *
 * The tab list was previously restated three times (the button bar, the
 * visibility-state object, and the settings checkboxes) with eight
 * near-identical inline-styled buttons. Defining the tabs once here lets the
 * tab bar, the body switch, and the settings panel iterate the same array.
 *
 * The matching `.tab-btn` CSS rule lives in `index.css`.
 */

export type TabId =
  | "trends"
  | "controllers"
  | "routing"
  | "tuning"
  | "events"
  | "ladder"
  | "hierarchy"
  | "powerSupply";

export interface TabDef {
  /** Stable identifier used as the `activeTab` value and visibility key. */
  id: TabId;
  /** Label rendered on the tab button. */
  label: string;
  /** Label rendered in the settings visibility panel. */
  settingsLabel: string;
  /** CSS variable used for the active accent color. */
  accentVar: string;
}

export const TABS: readonly TabDef[] = [
  {
    id: "trends",
    label: "Trends & Monitors",
    settingsLabel: "Live Trends & Monitors",
    accentVar: "var(--accent-cyan)",
  },
  {
    id: "controllers",
    label: "PID & Mass Flow",
    settingsLabel: "PID Loops & MFCs",
    accentVar: "var(--accent-purple)",
  },
  {
    id: "routing",
    label: "Signal Routing",
    settingsLabel: "Signal Routing Matrix",
    accentVar: "var(--accent-magenta)",
  },
  {
    id: "tuning",
    label: "Tuning & MPC",
    settingsLabel: "Tuning & MPC Groundwork",
    accentVar: "var(--accent-cyan)",
  },
  {
    id: "events",
    label: "Events & Alarms",
    settingsLabel: "Alarms & Event Log",
    accentVar: "var(--color-warning)",
  },
  {
    id: "ladder",
    label: "Ladder Explorer",
    settingsLabel: "Ladder Explorer",
    accentVar: "var(--accent-magenta)",
  },
  {
    id: "hierarchy",
    label: "Plant Hierarchy",
    settingsLabel: "Plant Hierarchy",
    accentVar: "var(--accent-cyan)",
  },
  {
    id: "powerSupply",
    label: "Power Supply",
    settingsLabel: "Power Supply Controller",
    accentVar: "var(--accent-purple, #a78bfa)",
  },
] as const;

/** Default visibility map (all tabs visible) derived from {@link TABS}. */
export function defaultTabVisibility(): Record<TabId, boolean> {
  return TABS.reduce(
    (acc, tab) => {
      acc[tab.id] = true;
      return acc;
    },
    {} as Record<TabId, boolean>,
  );
}
