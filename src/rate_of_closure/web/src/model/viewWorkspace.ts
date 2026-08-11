/** Versioned multi-viewport layout and resilient local persistence (#4225). */

export const VIEW_KINDS = ["impact", "swing", "flight"] as const;
export type ViewKind = (typeof VIEW_KINDS)[number];
export const VIEW_LAYOUTS = [
  "single",
  "split_horizontal",
  "split_vertical",
  "grid",
] as const;
export type ViewLayout = (typeof VIEW_LAYOUTS)[number];

export interface ViewSlot {
  readonly id: ViewKind;
  readonly kind: ViewKind;
  readonly legend: LegendPlacement;
}

export type LegendPlacement = "hidden" | "outside_right";

export interface ViewPlayback {
  readonly timeS: number;
  readonly playing: boolean;
  readonly loop: boolean;
  readonly rate: number;
}

export interface ViewWorkspace {
  readonly layout: ViewLayout;
  readonly slots: readonly ViewSlot[];
  readonly activeSlotId: ViewKind;
  readonly playback: ViewPlayback;
}

export const VIEW_WORKSPACE_STORAGE_KEY = "rate-of-closure.web.view-workspace.v1";
const FORMAT = "rate_of_closure.view_workspace/1";
const DEFAULT_PLAYBACK: ViewPlayback = {
  timeS: 0,
  playing: false,
  loop: false,
  rate: 1,
};
const slot = (kind: ViewKind): ViewSlot => ({ id: kind, kind, legend: "outside_right" });

export const defaultViewWorkspace: ViewWorkspace = {
  layout: "single",
  slots: [slot("swing")],
  activeSlotId: "swing",
  playback: DEFAULT_PLAYBACK,
};

interface StorageReader { getItem(key: string): string | null }
interface StorageWriter { setItem(key: string, value: string): void }

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const isViewKind = (value: unknown): value is ViewKind =>
  typeof value === "string" && VIEW_KINDS.includes(value as ViewKind);
function normalizedLayout(value: unknown, count: number): ViewLayout {
  if (count === 1) return "single";
  if (count >= VIEW_KINDS.length) return "grid";
  return value === "split_vertical" || value === "split_horizontal"
    ? value
    : "split_horizontal";
}

function playbackCandidate(value: Record<string, unknown>): ViewPlayback | null {
  const candidate = {
    timeS: value.time_s,
    playing: value.playing,
    loop: value.loop,
    rate: value.rate,
  };
  if (
    typeof candidate.timeS !== "number" || !Number.isFinite(candidate.timeS) || candidate.timeS < 0 ||
    typeof candidate.rate !== "number" || !Number.isFinite(candidate.rate) || candidate.rate <= 0 ||
    typeof candidate.playing !== "boolean" || typeof candidate.loop !== "boolean"
  ) return null;
  return candidate as ViewPlayback;
}

function recoveredPlayback(value: unknown): ViewPlayback {
  return isRecord(value) ? playbackCandidate(value) ?? DEFAULT_PLAYBACK : DEFAULT_PLAYBACK;
}

function recoveredSlots(value: unknown): ViewSlot[] {
  if (!Array.isArray(value)) return [];
  const slots = value.flatMap((item): ViewSlot[] => {
    if (isViewKind(item)) return [slot(item)];
    if (!isRecord(item) || item.id !== item.kind || !isViewKind(item.id)) return [];
    const legend: LegendPlacement = item.legend === "hidden"
      ? "hidden"
      : "outside_right";
    return [{ id: item.id, kind: item.id, legend }];
  });
  return slots.filter((candidate, index) =>
    slots.findIndex(({ id }) => id === candidate.id) === index);
}

function slotsForKinds(current: ViewWorkspace, kinds: readonly ViewKind[]): ViewSlot[] {
  return kinds.map((kind) =>
    current.slots.find(({ id }) => id === kind) ?? slot(kind));
}

/** Recover canonical or legacy layout data with deterministic known-view fallback. */
export function migrateViewWorkspace(value: unknown): ViewWorkspace {
  if (!isRecord(value)) return defaultViewWorkspace;
  const slots = recoveredSlots(Array.isArray(value.slots) ? value.slots : value.views);
  if (slots.length === 0) return defaultViewWorkspace;
  const kinds = slots.map(({ kind }) => kind);
  const activeRaw = value.active_slot_id ?? value.active;
  const activeSlotId = isViewKind(activeRaw) && kinds.includes(activeRaw)
    ? activeRaw
    : kinds[0];
  return {
    layout: normalizedLayout(value.layout, kinds.length),
    slots,
    activeSlotId,
    playback: recoveredPlayback(value.playback),
  };
}

export function workspaceForSingleView(kind: ViewKind, current: ViewWorkspace): ViewWorkspace {
  return {
    ...current,
    layout: "single",
    slots: slotsForKinds(current, [kind]),
    activeSlotId: kind,
  };
}

export function workspaceWithLayout(layout: ViewLayout, current: ViewWorkspace): ViewWorkspace {
  if (layout === "single") return workspaceForSingleView(current.activeSlotId, current);
  const minimum = layout === "grid" ? VIEW_KINDS.length : 2;
  const present = current.slots.map(({ kind }) => kind);
  const kinds = [...present, ...VIEW_KINDS.filter((kind) => !present.includes(kind))]
    .slice(0, minimum);
  return {
    ...current,
    layout: normalizedLayout(layout, kinds.length),
    slots: slotsForKinds(current, kinds),
    activeSlotId: kinds[0],
  };
}

export function toggleWorkspaceView(current: ViewWorkspace, kind: ViewKind): ViewWorkspace {
  const present = current.slots.map(({ kind: candidate }) => candidate);
  const kinds = present.includes(kind)
    ? present.filter((candidate) => candidate !== kind)
    : [...present, kind];
  if (kinds.length === 0) return current;
  const activeSlotId = kinds.includes(current.activeSlotId) ? current.activeSlotId : kinds[0];
  return {
    ...current,
    layout: normalizedLayout(current.layout, kinds.length),
    slots: slotsForKinds(current, kinds),
    activeSlotId,
  };
}

export function viewWorkspaceDocument(workspace: ViewWorkspace): Record<string, unknown> {
  return {
    format: FORMAT,
    layout: workspace.layout,
    slots: workspace.slots.map(({ id, kind, legend }) => ({
      id, kind, plot_id: null, legend,
    })),
    active_slot_id: workspace.activeSlotId,
    playback: {
      time_s: workspace.playback.timeS,
      playing: workspace.playback.playing,
      loop: workspace.playback.loop,
      rate: workspace.playback.rate,
    },
  };
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  context: string,
): void {
  const actual = Object.keys(value);
  if (actual.length !== expected.length || expected.some((key) => !(key in value))) {
    throw new TypeError(`${context} has invalid fields`);
  }
}

function strictWorkspaceDocument(value: unknown): ViewWorkspace {
  if (!isRecord(value)) throw new TypeError("workspace document must be an object");
  exactKeys(value, ["format", "layout", "slots", "active_slot_id", "playback"], "workspace");
  if (value.format !== FORMAT) {
    throw new TypeError(`unsupported workspace format: ${String(value.format)}`);
  }
  if (!VIEW_LAYOUTS.includes(value.layout as ViewLayout) || !Array.isArray(value.slots)) {
    throw new TypeError("workspace layout and slots must be valid");
  }
  const slots = value.slots.map((item): ViewSlot => {
    if (!isRecord(item)) throw new TypeError("workspace slot must be an object");
    exactKeys(item, ["id", "kind", "plot_id", "legend"], "workspace slot");
    if (item.id !== item.kind || !isViewKind(item.id) || item.plot_id !== null) {
      throw new TypeError("workspace slot identity is invalid");
    }
    if (item.legend !== "hidden" && item.legend !== "outside_right") {
      throw new TypeError("workspace slot legend is invalid");
    }
    return { id: item.id, kind: item.id, legend: item.legend };
  });
  if (slots.length === 0 || slots.length > VIEW_KINDS.length ||
      new Set(slots.map(({ id }) => id)).size !== slots.length) {
    throw new TypeError("workspace slots must be non-empty and unique");
  }
  if (!isViewKind(value.active_slot_id) ||
      !slots.some(({ id }) => id === value.active_slot_id)) {
    throw new TypeError("workspace active slot is invalid");
  }
  if (normalizedLayout(value.layout, slots.length) !== value.layout) {
    throw new TypeError("workspace layout cardinality is invalid");
  }
  if (!isRecord(value.playback)) throw new TypeError("workspace playback is invalid");
  exactKeys(value.playback, ["time_s", "playing", "loop", "rate"], "workspace playback");
  const playback = playbackCandidate(value.playback);
  if (playback === null) throw new TypeError("workspace playback is invalid");
  return {
    layout: value.layout as ViewLayout,
    slots,
    activeSlotId: value.active_slot_id,
    playback,
  };
}

/** Serialize a strict, versioned compositor document for file/workspace adapters. */
export function exportViewWorkspace(workspace: ViewWorkspace): string {
  const document = viewWorkspaceDocument(workspace);
  strictWorkspaceDocument(document);
  return `${JSON.stringify(document, null, 2)}\n`;
}

/** Parse a strict compositor export without partial mutation or version recovery. */
export function importViewWorkspace(text: string): ViewWorkspace {
  if (typeof text !== "string" || text.trim().length === 0) {
    throw new TypeError("workspace import must be non-empty JSON text");
  }
  return strictWorkspaceDocument(JSON.parse(text));
}

/** Validate one already-parsed compositor document without recovery. */
export function viewWorkspaceFromDocument(document: unknown): ViewWorkspace {
  return strictWorkspaceDocument(document);
}

export function loadViewWorkspace(storage?: StorageReader | null): ViewWorkspace {
  try {
    const target = storage === undefined
      ? (typeof window === "undefined" ? null : window.localStorage)
      : storage;
    const raw = target?.getItem(VIEW_WORKSPACE_STORAGE_KEY);
    return raw === null || raw === undefined
      ? defaultViewWorkspace
      : migrateViewWorkspace(JSON.parse(raw));
  } catch {
    return defaultViewWorkspace;
  }
}

export function saveViewWorkspace(
  workspace: ViewWorkspace,
  storage?: StorageWriter | null,
): boolean {
  try {
    const target = storage === undefined
      ? (typeof window === "undefined" ? null : window.localStorage)
      : storage;
    if (target === null) return false;
    target.setItem(VIEW_WORKSPACE_STORAGE_KEY, JSON.stringify(viewWorkspaceDocument(workspace)));
    return true;
  } catch {
    return false;
  }
}
