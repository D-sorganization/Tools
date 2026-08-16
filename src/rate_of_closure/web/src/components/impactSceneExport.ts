/** Versioned, reproducible data export for the interactive impact still. */

import type { ImpactKinematicsTs } from "../model/impactKinematics";

export interface ImpactCameraTs { yaw: number; pitch: number; zoom: number }

export function impactSceneExportPayload(
  scene: ImpactKinematicsTs,
  visibleLayers: ReadonlySet<string>,
  camera: ImpactCameraTs,
) {
  return {
    format: "rate-of-closure.impact-scene/v2",
    ...scene,
    renderPreferences: {
      visibleLayers: [...visibleLayers].sort(),
      camera: { ...camera },
    },
  };
}
