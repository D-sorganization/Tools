import { describe, expect, it } from "vitest";

import { getClub } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import { impactKinematics } from "../model/impactKinematics";
import { runSimulation } from "../model/simulation";
import { impactSceneExportPayload } from "./impactSceneExport";

describe("impact scene data export", () => {
  it("pins the schema and records the exact active layers and camera", () => {
    const club = getClub("Driver 10.5°");
    const run = runSimulation({
      sourceKind: "manual",
      clubheadSpeedMph: DEFAULT_SCENARIO.clubheadSpeedMph,
      omegaDps: [0, 0, 0],
      loftDeg: club.loftDeg,
      impactOffsetToeMm: 0,
      impactOffsetHighMm: 0,
      planeYawDeg: 0,
      planeSideTiltDeg: -45,
      planeForwardTiltDeg: 0,
      impactTimeS: 0.03,
      swingDurationS: 1.5,
    });
    const scene = impactKinematics(run, DEFAULT_SCENARIO, club);
    const payload = impactSceneExportPayload(
      scene,
      new Set(["spinLoftSector", "faceNormal"]),
      { yaw: 2.62, pitch: 0.52, zoom: 2.2 },
    );

    expect(payload.format).toBe("rate-of-closure.impact-scene/v3");
    expect(payload.sashoFaceCenterRotation.methodId).toBe(
      "sasho_nearest_shaft_face_center_rotation_only_aoa_v1",
    );
    expect(payload.sashoFaceCenterRotation.nearestShaftPointM).toHaveLength(3);
    expect(payload.angularVelocityRadS).toHaveLength(3);
    expect(payload.renderPreferences.visibleLayers).toEqual([
      "faceNormal", "spinLoftSector",
    ]);
    expect(payload.renderPreferences.camera.zoom).toBe(2.2);
    expect(() => JSON.stringify(payload)).not.toThrow();
  });
});
