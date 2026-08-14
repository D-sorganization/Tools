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
  | "operator"
  | "trends"
  | "explorer"
  | "controllers"
  | "routing"
  | "tuning"
  | "events"
  | "ladder"
  | "hierarchy"
  | "powerSupply"
  | "temperature"
  | "diagnostics";

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
    id: "operator",
    label: "Operator Overview",
    settingsLabel: "Representative Operator Workspace",
    accentVar: "var(--accent-cyan)",
  },
  {
    id: "trends",
    label: "Trends & Monitors",
    settingsLabel: "Live Trends & Monitors",
    accentVar: "var(--accent-cyan)",
  },
  {
    id: "explorer",
    label: "Data Explorer",
    settingsLabel: "Data Explorer & Analysis",
    accentVar: "var(--accent-magenta)",
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
  {
    id: "temperature",
    label: "Heater Controls",
    settingsLabel: "Heater Controls",
    accentVar: "var(--accent-orange, #fb923c)",
  },
  {
    id: "diagnostics",
    label: "Signal Diagnostics",
    settingsLabel: "Signal Diagnostics (raw 0–5 V)",
    accentVar: "var(--accent-cyan)",
  },
] as const;

/** Default visibility map (all tabs visible) derived from {@link TABS}. */
export function defaultTabVisibility(): Record<TabId, boolean> {
  // ⚡ Bolt Optimization: Replace chained array operations (.reduce, .map) with single-pass loops to avoid callback and closure overhead.
  const acc = {} as Record<TabId, boolean>;
  for (let i = 0; i < TABS.length; i++) {
    acc[TABS[i].id] = true;
  }
  return acc;
}

/** The canonical tab id order as declared in {@link TABS}. */
export function defaultTabOrder(): TabId[] {
  // ⚡ Bolt Optimization: Replace array map with a single-pass loop to avoid callback and closure overhead.
  const order = new Array<TabId>(TABS.length);
  for (let i = 0; i < TABS.length; i++) {
    order[i] = TABS[i].id;
  }
  return order;
}

const ORDER_KEY = "p1am.tabOrder.v1";
const VISIBILITY_KEY = "p1am.tabVisibility.v1";

/**
 * Reconcile a saved id list with the current {@link TABS}: keep the saved order
 * for ids that still exist, drop unknown ids, and append any tabs added since
 * the order was saved (so a new release's tab is never silently hidden).
 */
function reconcileOrder(saved: readonly unknown[]): TabId[] {
  // ⚡ Bolt Optimization: Replace chained .filter() array passes with single-pass for-loops and O(1) Set lookups
  const known = new Set<TabId>(defaultTabOrder());
  const kept: TabId[] = [];
  const keptSet = new Set<TabId>();
  for (let i = 0; i < saved.length; i++) {
    const id = saved[i];
    if (typeof id === "string" && known.has(id as TabId)) {
      kept.push(id as TabId);
      keptSet.add(id as TabId);
    }
  }

  const missing: TabId[] = [];
  const defaults = defaultTabOrder();
  for (let i = 0; i < defaults.length; i++) {
    const id = defaults[i];
    if (!keptSet.has(id)) {
      missing.push(id);
    }
  }
  return [...kept, ...missing];
}

/** Load the persisted tab order, falling back to the declared order. */
export function loadTabOrder(): TabId[] {
  try {
    const raw = localStorage.getItem(ORDER_KEY);
    if (raw) return reconcileOrder(JSON.parse(raw));
  } catch {
    /* corrupt/unavailable storage — use defaults */
  }
  return defaultTabOrder();
}

export function saveTabOrder(order: readonly TabId[]): void {
  try {
    localStorage.setItem(ORDER_KEY, JSON.stringify(order));
  } catch {
    /* storage unavailable — non-fatal */
  }
}

/** Load persisted visibility, defaulting any unknown/new tab to visible. */
export function loadTabVisibility(): Record<TabId, boolean> {
  const base = defaultTabVisibility();
  try {
    const raw = localStorage.getItem(VISIBILITY_KEY);
    if (raw) {
      const saved = JSON.parse(raw) as Partial<Record<TabId, boolean>>;
      for (const tab of TABS) {
        if (typeof saved[tab.id] === "boolean") base[tab.id] = saved[tab.id]!;
      }
    }
  } catch {
    /* corrupt/unavailable storage — use defaults */
  }
  return base;
}

export function saveTabVisibility(visibility: Record<TabId, boolean>): void {
  try {
    localStorage.setItem(VISIBILITY_KEY, JSON.stringify(visibility));
  } catch {
    /* storage unavailable — non-fatal */
  }
}
