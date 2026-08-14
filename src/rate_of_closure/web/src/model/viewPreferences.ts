/** Versioned, defensive persistence for workbench module navigation. */

export const PRIMARY_VIEWS = [
  { id: "explorer", label: "Explorer", required: true },
  { id: "calculation", label: "Calculation Description", required: false },
  { id: "simulation", label: "Simulation", required: false },
  { id: "plots", label: "Plots", required: false },
  { id: "flight", label: "Flight Explorer", required: false },
  { id: "regional-surfaces", label: "Ground Surfaces", required: false },
  { id: "ground-playback", label: "Ground Playback", required: false },
  { id: "launch-monitor-analytics", label: "Launch Monitor Analytics", required: false },
  { id: "capability-optimization", label: "Shot Optimizer", required: false },
  { id: "variation", label: "Variation", required: false },
  { id: "putting", label: "Putting", required: false },
  { id: "glossary", label: "Glossary", required: false },
] as const;

export type PrimaryViewId = (typeof PRIMARY_VIEWS)[number]["id"];

export const PRIMARY_VIEW_IDS: readonly PrimaryViewId[] = PRIMARY_VIEWS.map(
  ({ id }) => id,
);
export const REQUIRED_PRIMARY_VIEW_IDS: readonly PrimaryViewId[] = PRIMARY_VIEWS
  .filter(({ required }) => required)
  .map(({ id }) => id);
const NEWLY_REGISTERED_VISIBLE_IDS: readonly PrimaryViewId[] = ["ground-playback"];
export const PRIMARY_VIEW_STORAGE_KEY =
  "rate-of-closure.web.workspace-modules.v2";
export const LEGACY_PRIMARY_VIEW_STORAGE_KEY =
  "rate-of-closure.web.primary-views.v1";

export interface PrimaryViewState {
  version: 2;
  order: PrimaryViewId[];
  visible: PrimaryViewId[];
  active: PrimaryViewId;
}

const defaultState = (): PrimaryViewState => ({
  version: 2,
  order: [...PRIMARY_VIEW_IDS],
  visible: [...PRIMARY_VIEW_IDS],
  active: "explorer",
});

export const DEFAULT_PRIMARY_VIEW_STATE: PrimaryViewState = defaultState();

interface StorageReader {
  getItem(key: string): string | null;
}

interface StorageWriter {
  setItem(key: string, value: string): void;
}

function isPrimaryViewId(value: unknown): value is PrimaryViewId {
  return typeof value === "string" && PRIMARY_VIEW_IDS.includes(value as PrimaryViewId);
}

function sanitizeIds(value: unknown): PrimaryViewId[] {
  const supplied = Array.isArray(value) ? value.filter(isPrimaryViewId) : [];
  return supplied.filter((id, index) => supplied.indexOf(id) === index);
}

function sanitizeOrder(value: unknown): PrimaryViewId[] {
  const unique = sanitizeIds(value);
  return [...unique, ...PRIMARY_VIEW_IDS.filter((id) => !unique.includes(id))];
}

function sanitizeVisible(value: unknown, savedOrder: unknown): PrimaryViewId[] {
  if (!Array.isArray(value)) return [...PRIMARY_VIEW_IDS];
  const supplied = sanitizeIds(value);
  const known = new Set(sanitizeIds(savedOrder));
  const introduced = NEWLY_REGISTERED_VISIBLE_IDS.filter((id) => !known.has(id));
  const visible = [...supplied, ...introduced.filter((id) => !supplied.includes(id))];
  return [...visible, ...REQUIRED_PRIMARY_VIEW_IDS.filter((id) => !visible.includes(id))];
}

function normalizedState(
  orderValue: unknown,
  visibleValue: unknown,
  activeValue: unknown,
): PrimaryViewState {
  const order = sanitizeOrder(orderValue);
  const visible = sanitizeVisible(visibleValue, orderValue);
  const fallback = order.find((id) => visible.includes(id)) ?? "explorer";
  const active = isPrimaryViewId(activeValue) && visible.includes(activeValue)
    ? activeValue
    : fallback;
  return { version: 2, order, visible, active };
}

function targetStorage(storage?: StorageReader | null): StorageReader | null {
  if (storage !== undefined) return storage;
  return typeof window === "undefined" ? null : window.localStorage;
}

function parseDocument(raw: string): Record<string, unknown> {
  const value: unknown = JSON.parse(raw);
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("Workspace module preferences must be an object.");
  }
  return value as Record<string, unknown>;
}

export function loadPrimaryViewState(
  storage?: StorageReader | null,
): PrimaryViewState {
  try {
    const target = targetStorage(storage);
    if (target === null) return defaultState();
    const current = target.getItem(PRIMARY_VIEW_STORAGE_KEY);
    if (current !== null) {
      const parsed = parseDocument(current);
      if (parsed.version !== 2) return defaultState();
      return normalizedState(parsed.order, parsed.visible, parsed.active);
    }
    const legacy = target.getItem(LEGACY_PRIMARY_VIEW_STORAGE_KEY);
    if (legacy === null) return defaultState();
    const parsed = parseDocument(legacy);
    if (parsed.version !== 1) return defaultState();
    return normalizedState(parsed.order, PRIMARY_VIEW_IDS, parsed.active);
  } catch {
    return defaultState();
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
    const normalized = normalizedState(state.order, state.visible, state.active);
    target.setItem(PRIMARY_VIEW_STORAGE_KEY, JSON.stringify(normalized));
    return true;
  } catch {
    return false;
  }
}

export function visiblePrimaryViewIds(state: PrimaryViewState): PrimaryViewId[] {
  return state.order.filter((id) => state.visible.includes(id));
}

export function setPrimaryViewVisibility(
  state: PrimaryViewState,
  view: string,
  visible: boolean,
): PrimaryViewState {
  if (!isPrimaryViewId(view)) return state;
  if (!visible && REQUIRED_PRIMARY_VIEW_IDS.includes(view)) return state;
  const visibleSet = new Set(state.visible);
  if (visible) visibleSet.add(view);
  else visibleSet.delete(view);
  REQUIRED_PRIMARY_VIEW_IDS.forEach((id) => visibleSet.add(id));
  const nextVisible = state.order.filter((id) => visibleSet.has(id));
  const active = nextVisible.includes(state.active) ? state.active : nextVisible[0];
  return { ...state, visible: nextVisible, active };
}

export function restorePrimaryViewDefaults(): PrimaryViewState {
  return defaultState();
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

export function shiftVisiblePrimaryView(
  state: PrimaryViewState,
  moving: PrimaryViewId,
  offset: -1 | 1,
): PrimaryViewId[] {
  const visible = visiblePrimaryViewIds(state);
  const source = visible.indexOf(moving);
  const target = visible[source + offset];
  if (source < 0 || target === undefined) return [...state.order];
  return movePrimaryView(state.order, moving, target, offset > 0);
}

export function primaryViewLabel(id: PrimaryViewId): string {
  return PRIMARY_VIEWS.find((view) => view.id === id)?.label ?? id;
}

export function primaryViewRequired(id: PrimaryViewId): boolean {
  return PRIMARY_VIEWS.find((view) => view.id === id)?.required ?? false;
}
