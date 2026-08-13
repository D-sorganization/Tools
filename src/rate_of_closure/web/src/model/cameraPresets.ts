/** Canonical camera presets in app frame x=downrange, y=up, z=right. */

import type { Vec3 } from "./simulation";

export const CAMERA_VIEW_IDS = [
  "camera.view.isometric",
  "camera.view.face_on",
  "camera.view.down_the_line",
  "camera.view.overhead",
] as const;
export const CAMERA_ACTION_IDS = ["camera.reset_view", "camera.auto_fit"] as const;
export const CAMERA_TRACKING_COMMAND_IDS = [
  "camera.track_clubhead", "camera.recenter",
] as const;
export const CAMERA_CONTROL_IDS = ["camera.auto_fit_fallback"] as const;
export const CAMERA_TRACKING_STATE_IDS = [
  "camera.tracking.off", "camera.tracking.active", "camera.tracking.suspended",
] as const;
export const CAMERA_PRESET_COMMAND_IDS = [
  ...CAMERA_VIEW_IDS, ...CAMERA_ACTION_IDS,
] as const;
export const CAMERA_COMMAND_IDS = [
  ...CAMERA_PRESET_COMMAND_IDS, ...CAMERA_TRACKING_COMMAND_IDS,
] as const;

export type CameraViewId = (typeof CAMERA_VIEW_IDS)[number];
export type CameraActionId = (typeof CAMERA_ACTION_IDS)[number];
export type CameraCommandId = (typeof CAMERA_COMMAND_IDS)[number];
export type CameraTrackingStateId = (typeof CAMERA_TRACKING_STATE_IDS)[number];
export type FaceOnSide = "right" | "left";

export interface CameraPreset {
  commandId: CameraViewId;
  viewDirection: Vec3;
  screenUp: Vec3;
}

export interface CameraState {
  presetId: CameraViewId;
  faceOnSide: FaceOnSide;
  targetM: Vec3;
  zoom: number;
  yawRad: number;
  pitchRad: number;
  trackingEnabled: boolean;
  trackingSuspended: boolean;
  autoFitFallbackEnabled: boolean;
}

export const MIN_CAMERA_ZOOM = 0.3;
export const MAX_CAMERA_ZOOM = 4;
export const AUTO_FIT_CLEARANCE_FRACTION = 0.16;
export const TRACKING_MAX_TARGET_STEP_M = 0.05;

const ISOMETRIC_DIRECTION: Vec3 = [
  0.7071067811865476, -0.4082482904638631, -0.5773502691896258,
];
const ISOMETRIC_SCREEN_UP: Vec3 = [
  0.316227766016838, 0.9128709291752768, -0.2581988897471612,
];
const VERTICAL_UP: Vec3 = [0, 1, 0];

export function cameraPreset(commandId: CameraViewId | string, side: FaceOnSide | string): CameraPreset {
  const viewId = parseCameraViewId(commandId);
  const faceSide = parseFaceOnSide(side);
  if (viewId === "camera.view.isometric") {
    return validatedPreset(viewId, ISOMETRIC_DIRECTION, ISOMETRIC_SCREEN_UP);
  }
  if (viewId === "camera.view.face_on") {
    return validatedPreset(viewId, [0, 0, faceSide === "right" ? -1 : 1], VERTICAL_UP);
  }
  if (viewId === "camera.view.down_the_line") {
    return validatedPreset(viewId, [1, 0, 0], VERTICAL_UP);
  }
  return validatedPreset(viewId, [0, -1, 0], [1, 0, 0]);
}

export function canvasAngles(preset: CameraPreset): { yawRad: number; pitchRad: number } {
  const [downrange, up, right] = preset.viewDirection;
  return {
    yawRad: Math.atan2(right, downrange),
    pitchRad: Math.asin(Math.max(-1, Math.min(1, up))),
  };
}

export function defaultCameraState(): CameraState {
  const preset = cameraPreset("camera.view.isometric", "right");
  return validateCameraState({
    presetId: preset.commandId,
    faceOnSide: "right",
    targetM: [0, 0, 0],
    zoom: 1,
    trackingEnabled: false,
    trackingSuspended: false,
    autoFitFallbackEnabled: false,
    ...canvasAngles(preset),
  });
}

export function applyCameraView(state: CameraState, commandId: CameraViewId | string): CameraState {
  validateCameraState(state);
  const preset = cameraPreset(commandId, state.faceOnSide);
  return { ...state, presetId: preset.commandId, ...canvasAngles(preset) };
}

export function setFaceOnSide(state: CameraState, side: FaceOnSide | string): CameraState {
  validateCameraState(state);
  const faceOnSide = parseFaceOnSide(side);
  const next = { ...state, faceOnSide };
  return state.presetId === "camera.view.face_on"
    ? applyCameraView(next, "camera.view.face_on") : next;
}

export function withCameraZoom(state: CameraState, zoom: number): CameraState {
  validateCameraState(state);
  if (!Number.isFinite(zoom)) throw new RangeError("zoom must be finite");
  return { ...state, zoom: Math.max(MIN_CAMERA_ZOOM, Math.min(MAX_CAMERA_ZOOM, zoom)) };
}

export function autoFitCamera(
  state: CameraState,
  subjectRadiusM: number,
  baseHalfExtentM: number,
  clearanceFraction = AUTO_FIT_CLEARANCE_FRACTION,
): CameraState {
  if (![subjectRadiusM, baseHalfExtentM].every((value) => Number.isFinite(value) && value > 0)) {
    throw new RangeError("subject radius and base half extent must be finite and positive");
  }
  if (!Number.isFinite(clearanceFraction) || clearanceFraction < 0 || clearanceFraction >= 1) {
    throw new RangeError("clearanceFraction must be finite and within [0, 1)");
  }
  return withCameraZoom(
    state,
    baseHalfExtentM * (1 - clearanceFraction) / subjectRadiusM,
  );
}

export function setCameraTracking(
  state: CameraState, enabled: boolean, subjectM: Vec3,
): CameraState {
  validateCameraState(state);
  if (typeof enabled !== "boolean") throw new TypeError("enabled must be a Boolean");
  validateVector(subjectM, "subjectM");
  return {
    ...state,
    targetM: enabled ? [...subjectM] : state.targetM,
    trackingEnabled: enabled,
    trackingSuspended: false,
  };
}

export function setAutoFitFallback(state: CameraState, enabled: boolean): CameraState {
  validateCameraState(state);
  if (typeof enabled !== "boolean") throw new TypeError("enabled must be a Boolean");
  return { ...state, autoFitFallbackEnabled: enabled };
}

export function applyManualCameraOverride(
  state: CameraState, targetM: Vec3 = state.targetM,
): CameraState {
  validateCameraState(state);
  validateVector(targetM, "targetM");
  return {
    ...state,
    targetM: [...targetM],
    trackingSuspended: state.trackingEnabled,
  };
}

export function recenterCamera(state: CameraState, subjectM: Vec3): CameraState {
  validateCameraState(state);
  validateVector(subjectM, "subjectM");
  return { ...state, targetM: [...subjectM], trackingSuspended: false };
}

export function updateTrackingTarget(
  state: CameraState,
  subjectM: Vec3,
  maxStepM = TRACKING_MAX_TARGET_STEP_M,
): CameraState {
  validateCameraState(state);
  validateVector(subjectM, "subjectM");
  if (!Number.isFinite(maxStepM) || maxStepM <= 0) {
    throw new RangeError("maxStepM must be finite and positive");
  }
  if (!state.trackingEnabled || state.trackingSuspended) return state;
  const delta: Vec3 = [
    subjectM[0] - state.targetM[0],
    subjectM[1] - state.targetM[1],
    subjectM[2] - state.targetM[2],
  ];
  const distance = Math.hypot(...delta);
  if (distance <= 1e-12) return state;
  const fraction = Math.min(1, maxStepM / distance);
  return {
    ...state,
    targetM: [
      state.targetM[0] + fraction * delta[0],
      state.targetM[1] + fraction * delta[1],
      state.targetM[2] + fraction * delta[2],
    ],
  };
}

export function enforceTrackingClearance(
  state: CameraState,
  subjectRadiusM: number,
  baseHalfExtentM: number,
  clearanceFraction = AUTO_FIT_CLEARANCE_FRACTION,
): CameraState {
  const fitted = autoFitCamera(
    state, subjectRadiusM, baseHalfExtentM, clearanceFraction,
  );
  return !state.autoFitFallbackEnabled || state.zoom <= fitted.zoom ? state : fitted;
}

export function trackingStateId(state: CameraState): CameraTrackingStateId {
  validateCameraState(state);
  if (!state.trackingEnabled) return "camera.tracking.off";
  return state.trackingSuspended
    ? "camera.tracking.suspended" : "camera.tracking.active";
}

function parseCameraViewId(value: string): CameraViewId {
  if (!(CAMERA_VIEW_IDS as readonly string[]).includes(value)) {
    throw new RangeError(`unknown camera view: ${JSON.stringify(value)}`);
  }
  return value as CameraViewId;
}

function parseFaceOnSide(value: string): FaceOnSide {
  if (value !== "right" && value !== "left") {
    throw new RangeError(`unknown face-on side: ${JSON.stringify(value)}`);
  }
  return value;
}

function validatedPreset(commandId: CameraViewId, viewDirection: Vec3, screenUp: Vec3): CameraPreset {
  return validateCameraPreset({ commandId, viewDirection, screenUp });
}

export function validateCameraPreset(preset: CameraPreset): CameraPreset {
  const { commandId, viewDirection, screenUp } = preset;
  parseCameraViewId(commandId);
  validateVector(viewDirection, "viewDirection");
  validateVector(screenUp, "screenUp");
  const viewNorm = Math.hypot(...viewDirection);
  const upNorm = Math.hypot(...screenUp);
  const dot = viewDirection.reduce((sum, value, axis) => sum + value * screenUp[axis], 0);
  if (Math.abs(viewNorm - 1) > 1e-12 || Math.abs(upNorm - 1) > 1e-12) {
    throw new RangeError("camera directions must be unit vectors");
  }
  if (Math.abs(dot) > 1e-12) {
    throw new RangeError("screenUp must be perpendicular to viewDirection");
  }
  return { commandId, viewDirection: [...viewDirection], screenUp: [...screenUp] };
}

export function validateCameraState(state: CameraState): CameraState {
  const presetId = parseCameraViewId(state.presetId);
  const faceOnSide = parseFaceOnSide(state.faceOnSide);
  validateVector(state.targetM, "targetM");
  if (!Number.isFinite(state.zoom)
    || state.zoom < MIN_CAMERA_ZOOM || state.zoom > MAX_CAMERA_ZOOM) {
    throw new RangeError(
      `zoom must be finite and within [${MIN_CAMERA_ZOOM}, ${MAX_CAMERA_ZOOM}]`,
    );
  }
  if (!Number.isFinite(state.yawRad) || !Number.isFinite(state.pitchRad)) {
    throw new RangeError("camera angles must be finite");
  }
  for (const [value, name] of [
    [state.trackingEnabled, "trackingEnabled"],
    [state.trackingSuspended, "trackingSuspended"],
    [state.autoFitFallbackEnabled, "autoFitFallbackEnabled"],
  ] as const) {
    if (typeof value !== "boolean") throw new TypeError(`${name} must be a Boolean`);
  }
  if (state.trackingSuspended && !state.trackingEnabled) {
    throw new RangeError("tracking cannot be suspended while disabled");
  }
  return {
    ...state,
    presetId,
    faceOnSide,
    targetM: [...state.targetM],
  };
}

function validateVector(vector: Vec3, name: string): void {
  if (vector.length !== 3 || !vector.every(Number.isFinite)) {
    throw new RangeError(`${name} must contain three finite values`);
  }
}
