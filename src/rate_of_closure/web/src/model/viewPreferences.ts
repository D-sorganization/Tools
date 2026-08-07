/** Versioned, defensive persistence for the primary workbench navigation. */

export const PRIMARY_VIEWS = [
  { id: "explorer", label: "Explorer" },
  { id: "calculation", label: "Calculation Description" },
  { id: "simulation", label: "Simulation" },
  { id: "plots", label: "Plots" },
  { id: "flight", label: "Flight Explorer" },
  { id: "launch-monitor-analytics", label: "Launch Monitor Analytics" },
  { id: "variation", label: "Variation" },
  { id: "putting", label: "Putting" },
  { id: "glossary", label: "Glossary" },
] as const;

export type PrimaryViewId = (typeof PRIMARY_VIEWS)[number]["id"];

export const PRIMARY_VIEW_IDS: readonly PrimaryViewId[] = PRIMARY_VIEWS.map(
  ({ id }) => id,
);
export const PRIMARY_VIEW_STORAGE_KEY =
  "rate-of-closure.web.primary-views.v1";

export interface PrimaryViewState {
  version: 1;
  order: PrimaryViewId[];
  active: PrimaryViewId;
}

export const DEFAULT_PRIMARY_VIEW_STATE: PrimaryViewState = {
  version: 1,
  order: [...PRIMARY_VIEW_IDS],
  active: "explorer",
};

interface StorageReader {
  getItem(key: string): string | null;
}

interface StorageWriter {
  setItem(key: string, value: string): void;
}

function isPrimaryViewId(value: unknown): value is PrimaryViewId {
  return typeof value === "string" && PRIMARY_VIEW_IDS.includes(value as PrimaryViewId);
}

function sanitizeOrder(value: unknown): PrimaryViewId[] {
  const supplied = Array.isArray(value) ? value.filter(isPrimaryViewId) : [];
  const unique = supplied.filter((id, index) => supplied.indexOf(id) === index);
  return [...unique, ...PRIMARY_VIEW_IDS.filter((id) => !unique.includes(id))];
}

export function loadPrimaryViewState(
  storage?: StorageReader | null,
): PrimaryViewState {
  try {
    const target = storage === undefined
      ? (typeof window === "undefined" ? null : window.localStorage)
      : storage;
    if (target === null) return { ...DEFAULT_PRIMARY_VIEW_STATE, order: [...PRIMARY_VIEW_IDS] };
    const raw = target.getItem(PRIMARY_VIEW_STORAGE_KEY);
    if (raw === null) return { ...DEFAULT_PRIMARY_VIEW_STATE, order: [...PRIMARY_VIEW_IDS] };
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (parsed.version !== 1) return { ...DEFAULT_PRIMARY_VIEW_STATE, order: [...PRIMARY_VIEW_IDS] };
    const order = sanitizeOrder(parsed.order);
    const active = isPrimaryViewId(parsed.active) ? parsed.active : order[0];
    return { version: 1, order, active };
  } catch {
    return { ...DEFAULT_PRIMARY_VIEW_STATE, order: [...PRIMARY_VIEW_IDS] };
  }
}

export function savePrimaryViewState(
  state: PrimaryViewState,
  storage?: StorageWriter | null,
): boolean {
  try {
    const target = storage === undefined
      ? (typeof window === "undefined" ? null : window.localStorage)
      : storage;
    if (target === null) return false;
    target.setItem(PRIMARY_VIEW_STORAGE_KEY, JSON.stringify(state));
    return true;
  } catch {
    return false;
  }
}

export function movePrimaryView(
  order: readonly PrimaryViewId[],
  moving: string,
  target: string,
  placeAfter = false,
): PrimaryViewId[] {
  if (!isPrimaryViewId(moving) || !isPrimaryViewId(target) || moving === target) {
    return [...order];
  }
  const next = order.filter((id) => id !== moving);
  const targetIndex = next.indexOf(target);
  const destination = placeAfter ? targetIndex + 1 : targetIndex;
  if (targetIndex < 0) return [...order];
  next.splice(destination, 0, moving);
  return next;
}

export function shiftPrimaryView(
  order: readonly PrimaryViewId[],
  moving: PrimaryViewId,
  offset: -1 | 1,
): PrimaryViewId[] {
  const source = order.indexOf(moving);
  const destination = source + offset;
  if (source < 0 || destination < 0 || destination >= order.length) return [...order];
  const next = [...order];
  [next[source], next[destination]] = [next[destination], next[source]];
  return next;
}

export function primaryViewLabel(id: PrimaryViewId): string {
  return PRIMARY_VIEWS.find((view) => view.id === id)?.label ?? id;
}
