import { expect, it } from "vitest";

import { getClub } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import { impactKinematics } from "../model/impactKinematics";
import { runSimulation } from "../model/simulation";
import { impactSceneGeometry } from "./impactSceneGeometry";
import { impactSceneSvg } from "./impactSceneSvg";

it("exports labeled, locked-scale impact geometry as true SVG primitives", () => {
  const scenario = { ...DEFAULT_SCENARIO, clubheadSpeedMph: 30 };
  const run = runSimulation({
    sourceKind: "manual", clubheadSpeedMph: 30, omegaDps: [0, 0, 0],
    loftDeg: 46, impactOffsetToeMm: 0, impactOffsetHighMm: 0,
    planeYawDeg: 0, planeSideTiltDeg: -45, planeForwardTiltDeg: 0,
    impactTimeS: 0.03, swingDurationS: 1.5,
  });
  const scene = impactKinematics(run, scenario, getClub("Pitching Wedge"));
  const geometry = impactSceneGeometry(
    scene, new Set([
      ...scene.vectors.map((vector) => vector.key),
      "faceNormal", "faceCenterTravel", "dplaneNormal", "projectedPath",
      "spinLoftSector",
    ]),
  );

  const svg = impactSceneSvg(geometry, { yaw: 2.62, pitch: 0.52, zoom: 2.2 });

  expect(svg).toContain("<svg");
  expect(svg).toContain("<polyline");
  expect(svg).toContain("Physical Shaft Axis");
  expect(svg).toContain("Declared Contact Point");
  expect(svg).toContain("Face-Center Normal");
  expect(svg).toContain("Face-Center Travel");
  expect(svg).toContain("3D Spin Loft");
  expect(svg).toContain("<polygon");
  expect(svg).not.toContain("<image");
});
