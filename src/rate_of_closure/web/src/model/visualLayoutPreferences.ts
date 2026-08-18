import {
  canonicalClubCamera,
  DEFAULT_CLUB_CAMERA,
  type ClubCamera,
} from "./clubCamera";

export const VISUAL_LAYOUT_STORAGE_KEY = "rate-of-closure.web.visual-layout.v1";
export const MIN_SIDEBAR_FRACTION = 0.2;
export const MAX_SIDEBAR_FRACTION = 0.38;
export const DEFAULT_SIDEBAR_FRACTION = 0.27;

export interface VisualLayoutPreferences {
  readonly clubCamera: ClubCamera;
  readonly moduleHelpOpen: boolean;
  readonly shellSidebarFraction: number;
}

export const DEFAULT_VISUAL_LAYOUT: VisualLayoutPreferences = Object.freeze({
  clubCamera: DEFAULT_CLUB_CAMERA,
  moduleHelpOpen: false,
  shellSidebarFraction: DEFAULT_SIDEBAR_FRACTION,
});

interface StorageReader { getItem(key: string): string | null }
interface StorageWriter { setItem(key: string, value: string): void }

const record = (value: unknown, name: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
};

const number = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
  return value;
};

export function parseVisualLayout(value: unknown): VisualLayoutPreferences {
  const document = record(value, "visual layout");
  if (document.version !== 1) throw new TypeError("visual layout must be version 1");
  const rawCamera = record(document.clubCamera, "visual layout clubCamera");
  const azimuthDeg = number(rawCamera.azimuthDeg, "camera azimuth");
  const elevationDeg = number(rawCamera.elevationDeg, "camera elevation");
  const zoom = number(rawCamera.zoom, "camera zoom");
  if (azimuthDeg < -180 || azimuthDeg >= 180) throw new RangeError("camera azimuth range");
  if (elevationDeg < -80 || elevationDeg > 80) throw new RangeError("camera elevation range");
  if (zoom < 0.3 || zoom > 4) throw new RangeError("camera zoom range");
  if (typeof document.moduleHelpOpen !== "boolean") {
    throw new TypeError("moduleHelpOpen must be boolean");
  }
  const shellSidebarFraction = number(document.shellSidebarFraction, "sidebar fraction");
  if (shellSidebarFraction < MIN_SIDEBAR_FRACTION
    || shellSidebarFraction > MAX_SIDEBAR_FRACTION) {
    throw new RangeError("sidebar fraction range");
  }
  return Object.freeze({
    clubCamera: canonicalClubCamera({ azimuthDeg, elevationDeg, zoom }),
    moduleHelpOpen: document.moduleHelpOpen,
    shellSidebarFraction,
  });
}

export function visualLayoutDocument(preferences: VisualLayoutPreferences) {
  const validated = parseVisualLayout({ version: 1, ...preferences });
  return { version: 1, ...validated };
}

const targetStorage = (storage?: StorageReader | null): StorageReader | null =>
  storage === undefined
    ? (typeof window === "undefined" ? null : window.localStorage)
    : storage;

export function loadVisualLayout(
  storage?: StorageReader | null,
): VisualLayoutPreferences {
  try {
    const target = targetStorage(storage);
    if (target === null) return DEFAULT_VISUAL_LAYOUT;
    const raw = target.getItem(VISUAL_LAYOUT_STORAGE_KEY);
    return raw === null ? DEFAULT_VISUAL_LAYOUT : parseVisualLayout(JSON.parse(raw));
  } catch {
    return DEFAULT_VISUAL_LAYOUT;
  }
}

export function saveVisualLayout(
  preferences: VisualLayoutPreferences,
  storage?: StorageWriter | null,
): boolean {
  try {
    const target = storage === undefined
      ? (typeof window === "undefined" ? null : window.localStorage)
      : storage;
    if (target === null) return false;
    target.setItem(VISUAL_LAYOUT_STORAGE_KEY, JSON.stringify(visualLayoutDocument(preferences)));
    return true;
  } catch {
    return false;
  }
}
