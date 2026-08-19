/** UI-neutral camera commands in app frame x=downrange, y=up, z=right. */

import type { Vec3 } from "./simulation";

export const CAMERA_COMMAND_IDS = [
  "camera.view.isometric",
  "camera.view.face_on",
  "camera.view.down_the_line",
  "camera.view.overhead",
  "camera.auto_fit",
  "camera.recenter",
  "camera.track_subject",
] as const;

export type CameraCommandId = (typeof CAMERA_COMMAND_IDS)[number];
export type CameraViewId = Extract<CameraCommandId, `camera.view.${string}`>;
export type FaceOnSide = "right" | "left";

export interface CameraPreset {
  commandId: CameraViewId;
  viewDirection: Vec3;
  screenUp: Vec3;
}

export interface CameraState {
  presetId: CameraViewId | null;
  faceOnSide: FaceOnSide;
  targetM: Vec3;
  zoom: number;
  yawRad: number;
  pitchRad: number;
  trackingEnabled: boolean;
  trackingSuspended: boolean;
  autoFitEnabled: boolean;
}

const ISOMETRIC_DIRECTION: Vec3 = [
  0.7071067811865476, -0.4082482904638631, -0.5773502691896258,
];
const VERTICAL_UP: Vec3 = [0, 1, 0];
const ISOMETRIC_SCREEN_UP: Vec3 = [
  0.316227766016838, 0.9128709291752768, -0.2581988897471612,
];
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 8;
const TRACKING_CLEARANCE_FRACTION = 0.16;

export function cameraPreset(commandId: CameraViewId, side: FaceOnSide): CameraPreset {
  if (commandId === "camera.view.isometric") {
    return validatedPreset(commandId, ISOMETRIC_DIRECTION, ISOMETRIC_SCREEN_UP);
  }
  if (commandId === "camera.view.face_on") {
    return validatedPreset(commandId, [0, 0, side === "right" ? -1 : 1], VERTICAL_UP);
  }
  if (commandId === "camera.view.down_the_line") {
    return validatedPreset(commandId, [1, 0, 0], VERTICAL_UP);
  }
  return validatedPreset(commandId, [0, -1, 0], [1, 0, 0]);
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
  return {
    presetId: preset.commandId,
    faceOnSide: "right",
    targetM: [0, 0, 0],
    zoom: 1,
    ...canvasAngles(preset),
    trackingEnabled: false,
    trackingSuspended: false,
    autoFitEnabled: false,
  };
}

export function movingSubjectCameraState(): CameraState {
  return {
    ...defaultCameraState(),
    zoom: 2,
    trackingEnabled: true,
    autoFitEnabled: true,
  };
}

export function applyCameraPreset(state: CameraState, commandId: CameraViewId): CameraState {
  const preset = cameraPreset(commandId, state.faceOnSide);
  return { ...state, presetId: commandId, ...canvasAngles(preset) };
}

export function setFaceOnSide(state: CameraState, side: FaceOnSide): CameraState {
  const next = { ...state, faceOnSide: side };
  return state.presetId === "camera.view.face_on"
    ? applyCameraPreset(next, "camera.view.face_on") : next;
}

export function setTrackingEnabled(
  state: CameraState, enabled: boolean, subjectM: Vec3,
): CameraState {
  validateVector(subjectM, "subjectM");
  return {
    ...state,
    targetM: enabled ? [...subjectM] : state.targetM,
    trackingEnabled: enabled,
    trackingSuspended: false,
  };
}

export function applyManualOverride(state: CameraState): CameraState {
  return {
    ...state,
    presetId: null,
    trackingSuspended: state.trackingEnabled,
  };
}

export function recenterCamera(state: CameraState, subjectM: Vec3): CameraState {
  validateVector(subjectM, "subjectM");
  return { ...state, targetM: [...subjectM], trackingSuspended: false };
}

export function updateTrackingTarget(
  state: CameraState, subjectM: Vec3, maxStepM: number,
): CameraState {
  validateVector(subjectM, "subjectM");
  if (!Number.isFinite(maxStepM) || maxStepM <= 0) {
    throw new RangeError("maxStepM must be finite and positive");
  }
  if (!state.trackingEnabled || state.trackingSuspended) return state;
  const delta: Vec3 = subjectM.map((value, axis) => value - state.targetM[axis]) as Vec3;
  const distance = Math.hypot(...delta);
  if (distance <= 1e-12) return state;
  const fraction = Math.min(1, maxStepM / distance);
  const targetM = state.targetM.map((value, axis) => value + fraction * delta[axis]) as Vec3;
  return { ...state, targetM };
}

export function safeTrackingZoom(
  requestedZoom: number, subjectRadiusM: number, baseHalfExtentM: number,
): number {
  if (![requestedZoom, subjectRadiusM, baseHalfExtentM]
    .every((value) => Number.isFinite(value) && value > 0)) {
    throw new RangeError("zoom, subject radius, and base half extent must be positive");
  }
  const maximum = baseHalfExtentM * (1 - TRACKING_CLEARANCE_FRACTION) / subjectRadiusM;
  return Math.max(MIN_ZOOM, Math.min(requestedZoom, maximum, MAX_ZOOM));
}

export function withManualOrbit(
  state: CameraState, yawRad: number, pitchRad: number,
): CameraState {
  if (!Number.isFinite(yawRad) || !Number.isFinite(pitchRad)) {
    throw new RangeError("manual camera angles must be finite");
  }
  return { ...applyManualOverride(state), yawRad, pitchRad };
}

export function withCameraZoom(state: CameraState, zoom: number): CameraState {
  if (!Number.isFinite(zoom)) throw new RangeError("zoom must be finite");
  return { ...state, zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom)) };
}

function validatedPreset(
  commandId: CameraViewId, viewDirection: Vec3, screenUp: Vec3,
): CameraPreset {
  validateVector(viewDirection, "viewDirection");
  validateVector(screenUp, "screenUp");
  const viewNorm = Math.hypot(...viewDirection);
  const upNorm = Math.hypot(...screenUp);
  const dot = viewDirection.reduce((sum, value, axis) => sum + value * screenUp[axis], 0);
  if (Math.abs(viewNorm - 1) > 1e-12 || Math.abs(upNorm - 1) > 1e-12 || Math.abs(dot) > 1e-12) {
    throw new RangeError("camera directions must be orthonormal unit vectors");
  }
  return { commandId, viewDirection: [...viewDirection], screenUp: [...screenUp] };
}

function validateVector(vector: Vec3, name: string): void {
  if (vector.length !== 3 || !vector.every(Number.isFinite)) {
    throw new RangeError(`${name} must contain three finite values`);
  }
}
