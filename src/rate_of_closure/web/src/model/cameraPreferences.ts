/** Strict durable camera preferences shared by all 3D viewport adapters. */

import {
  CAMERA_COMMAND_IDS,
  applyCameraPreset,
  type CameraState,
  type CameraViewId,
  type FaceOnSide,
} from "./cameraCommands";

export const CAMERA_PREFERENCES_FORMAT = "camera-preferences/v1";
export const CAMERA_VIEWPORT_IDS = ["impact", "swing", "flight"] as const;
export type CameraViewportId = (typeof CAMERA_VIEWPORT_IDS)[number];

export interface CameraPreference {
  readonly presetId: CameraViewId;
  readonly faceOnSide: FaceOnSide;
  readonly zoom: number;
  readonly trackingEnabled: boolean;
  readonly autoFitEnabled: boolean;
}

export interface CameraPreferences {
  readonly viewports: Readonly<Record<CameraViewportId, CameraPreference>>;
}

const VIEW_COMMANDS = CAMERA_COMMAND_IDS.filter(
  (command): command is CameraViewId => command.startsWith("camera.view."),
);
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function exactRecord(
  value: unknown,
  expected: readonly string[],
  context: string,
): Record<string, unknown> {
  if (!isRecord(value)) throw new TypeError(`${context} must be an object`);
  const actual = Object.keys(value);
  if (actual.length !== expected.length || expected.some((key) => !(key in value))) {
    throw new TypeError(`${context} has invalid fields`);
  }
  return value;
}

const stationaryPreference = (): CameraPreference => ({
  presetId: "camera.view.isometric",
  faceOnSide: "right",
  zoom: 1,
  trackingEnabled: false,
  autoFitEnabled: false,
});
const movingPreference = (): CameraPreference => ({
  ...stationaryPreference(),
  zoom: 2,
  trackingEnabled: true,
  autoFitEnabled: true,
});

/** Deterministic #4303 migration defaults. */
export function defaultCameraPreferences(): CameraPreferences {
  return {
    viewports: {
      impact: stationaryPreference(),
      swing: movingPreference(),
      flight: movingPreference(),
    },
  };
}

function preferenceFromDocument(value: unknown): CameraPreference {
  const data = exactRecord(
    value,
    ["preset_id", "face_on_side", "zoom", "tracking_enabled", "auto_fit_enabled"],
    "camera preference",
  );
  if (!VIEW_COMMANDS.includes(data.preset_id as CameraViewId)) {
    throw new TypeError("camera preference preset_id is invalid");
  }
  if (data.face_on_side !== "right" && data.face_on_side !== "left") {
    throw new TypeError("camera preference face_on_side is invalid");
  }
  if (
    typeof data.zoom !== "number" ||
    !Number.isFinite(data.zoom) ||
    data.zoom < 0.25 ||
    data.zoom > 8
  ) {
    throw new RangeError("camera preference zoom must be within [0.25, 8]");
  }
  if (
    typeof data.tracking_enabled !== "boolean" ||
    typeof data.auto_fit_enabled !== "boolean"
  ) {
    throw new TypeError("camera preference tracking and Auto Fit must be booleans");
  }
  return {
    presetId: data.preset_id as CameraViewId,
    faceOnSide: data.face_on_side,
    zoom: data.zoom,
    trackingEnabled: data.tracking_enabled,
    autoFitEnabled: data.auto_fit_enabled,
  };
}

export function cameraPreferencesDocument(
  preferences: CameraPreferences,
): Record<string, unknown> {
  return {
    format: CAMERA_PREFERENCES_FORMAT,
    viewports: Object.fromEntries(CAMERA_VIEWPORT_IDS.map((viewportId) => {
      const preference = preferences.viewports[viewportId];
      return [viewportId, {
        preset_id: preference.presetId,
        face_on_side: preference.faceOnSide,
        zoom: preference.zoom,
        tracking_enabled: preference.trackingEnabled,
        auto_fit_enabled: preference.autoFitEnabled,
      }];
    })),
  };
}

export function cameraPreferencesFromDocument(value: unknown): CameraPreferences {
  const data = exactRecord(value, ["format", "viewports"], "camera preferences");
  if (data.format !== CAMERA_PREFERENCES_FORMAT) {
    throw new TypeError(`unsupported camera preferences format: ${String(data.format)}`);
  }
  const viewports = exactRecord(
    data.viewports,
    CAMERA_VIEWPORT_IDS,
    "camera preference viewports",
  );
  return {
    viewports: Object.fromEntries(CAMERA_VIEWPORT_IDS.map((viewportId) => [
      viewportId,
      preferenceFromDocument(viewports[viewportId]),
    ])) as unknown as Record<CameraViewportId, CameraPreference>,
  };
}

/** Capture durable fields only; manual orbit retains the last durable preset. */
export function preferenceFromCameraState(
  state: CameraState,
  fallback: CameraPreference,
): CameraPreference {
  return {
    presetId: state.presetId ?? fallback.presetId,
    faceOnSide: state.faceOnSide,
    zoom: state.zoom,
    trackingEnabled: state.trackingEnabled,
    autoFitEnabled: state.autoFitEnabled,
  };
}

/** Apply preferences without restoring a moving target or manual suspension. */
export function applyCameraPreference(
  state: CameraState,
  preference: CameraPreference,
): CameraState {
  return {
    ...applyCameraPreset(
      { ...state, faceOnSide: preference.faceOnSide },
      preference.presetId,
    ),
    zoom: preference.zoom,
    trackingEnabled: preference.trackingEnabled,
    trackingSuspended: false,
    autoFitEnabled: preference.autoFitEnabled,
  };
}

export function withCameraPreference(
  preferences: CameraPreferences,
  viewportId: CameraViewportId,
  preference: CameraPreference,
): CameraPreferences {
  const current = preferences.viewports[viewportId];
  if (
    current.presetId === preference.presetId &&
    current.faceOnSide === preference.faceOnSide &&
    current.zoom === preference.zoom &&
    current.trackingEnabled === preference.trackingEnabled &&
    current.autoFitEnabled === preference.autoFitEnabled
  ) return preferences;
  return {
    viewports: { ...preferences.viewports, [viewportId]: preference },
  };
}
