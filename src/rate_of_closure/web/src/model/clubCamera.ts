export interface ClubCamera {
  readonly azimuthDeg: number;
  readonly elevationDeg: number;
  readonly zoom: number;
}

export type ClubCameraAction =
  | "left" | "right" | "up" | "down"
  | "zoom_in" | "zoom_out" | "home";

export const DEFAULT_CLUB_CAMERA: ClubCamera = Object.freeze({
  azimuthDeg: 150,
  elevationDeg: 30,
  zoom: 1,
});

const ORBIT_STEP_DEG = 5;
const ZOOM_STEP = 1.1;

export function canonicalClubCamera(camera: ClubCamera): ClubCamera {
  const values = [camera.azimuthDeg, camera.elevationDeg, camera.zoom];
  if (!values.every(Number.isFinite)) throw new Error("club camera values must be finite");
  const rawAzimuth = ((camera.azimuthDeg + 180) % 360 + 360) % 360 - 180;
  return Object.freeze({
    azimuthDeg: Object.is(rawAzimuth, -0) ? 0 : rawAzimuth,
    elevationDeg: Math.max(-80, Math.min(80, camera.elevationDeg)),
    zoom: Math.max(0.3, Math.min(4, camera.zoom)),
  });
}

export function applyClubCameraAction(
  camera: ClubCamera,
  action: ClubCameraAction,
): ClubCamera {
  camera = canonicalClubCamera(camera);
  if (action === "home") return DEFAULT_CLUB_CAMERA;
  let { azimuthDeg, elevationDeg, zoom } = camera;
  if (action === "left") azimuthDeg -= ORBIT_STEP_DEG;
  if (action === "right") azimuthDeg += ORBIT_STEP_DEG;
  if (action === "up") elevationDeg = Math.min(80, elevationDeg + ORBIT_STEP_DEG);
  if (action === "down") elevationDeg = Math.max(-80, elevationDeg - ORBIT_STEP_DEG);
  if (action === "zoom_in") zoom = Math.min(4, zoom * ZOOM_STEP);
  if (action === "zoom_out") zoom = Math.max(0.3, zoom / ZOOM_STEP);
  return canonicalClubCamera({ azimuthDeg, elevationDeg, zoom });
}

export function cameraStatus(camera: ClubCamera, source: string): string {
  camera = canonicalClubCamera(camera);
  return `${source}; camera azimuth ${camera.azimuthDeg.toFixed(0)}°, `
    + `elevation ${camera.elevationDeg.toFixed(0)}°, zoom ${camera.zoom.toFixed(2)}×.`;
}

export function applyClubCameraDrag(
  camera: ClubCamera,
  deltaX: number,
  deltaY: number,
): ClubCamera {
  if (!Number.isFinite(deltaX) || !Number.isFinite(deltaY)) {
    throw new Error("club camera drag deltas must be finite");
  }
  camera = canonicalClubCamera(camera);
  return canonicalClubCamera({
    azimuthDeg: camera.azimuthDeg - deltaX * 0.45,
    elevationDeg: camera.elevationDeg + deltaY * 0.45,
    zoom: camera.zoom,
  });
}
