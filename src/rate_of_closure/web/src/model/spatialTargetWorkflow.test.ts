import { describe, expect, it } from "vitest";

import {
  createSpatialTarget,
  sphereTolerance,
  targetPointFromFrame,
} from "./spatialTarget";
import {
  spatialTargetForGroundWorkflow,
  SpatialTargetWorkflowError,
} from "./spatialTargetWorkflow";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "./targets";

describe("spatial target legacy-workflow adapter", () => {
  it("projects a validated landing target without changing its geometry", () => {
    const target = spatialTargetFromRegion({ ...DEFAULT_TARGET, distanceM: 205 });
    expect(spatialTargetForGroundWorkflow(target, "solver")).toEqual({
      targetRegion: { ...DEFAULT_TARGET, distanceM: 205 },
      diagnostic: null,
    });
  });

  it("fails closed with a typed diagnostic for an elevated target", () => {
    const target = createSpatialTarget({
      label: "Apex gate",
      kind: "aerial_waypoint",
      point: targetPointFromFrame([140, 24, -3], "app"),
      tolerance: sphereTolerance(4),
      elevationSource: "absolute",
    });
    const result = spatialTargetForGroundWorkflow(target, "variation");
    expect(result.targetRegion).toBeNull();
    expect(result.diagnostic).toBeInstanceOf(SpatialTargetWorkflowError);
    expect(result.diagnostic).toMatchObject({
      code: "AERIAL_TARGET_UNSUPPORTED",
      workflow: "variation",
    });
    expect(result.diagnostic?.message).toMatch(/elevation.*not.*zero/i);
  });
});
