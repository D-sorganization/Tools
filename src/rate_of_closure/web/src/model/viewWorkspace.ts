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

function recoveredPlayback(value: unknown): ViewPlayback {
  if (!isRecord(value)) return DEFAULT_PLAYBACK;
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
  ) return DEFAULT_PLAYBACK;
  return candidate as ViewPlayback;
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

function documentFor(workspace: ViewWorkspace): Record<string, unknown> {
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
    target.setItem(VIEW_WORKSPACE_STORAGE_KEY, JSON.stringify(documentFor(workspace)));
    return true;
  } catch {
    return false;
  }
}
