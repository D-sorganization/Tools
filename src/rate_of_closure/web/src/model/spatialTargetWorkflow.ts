/** Explicit compatibility boundary for workflows that only support surface targets. */

import { createSpatialTarget, type SpatialTargetTs } from "./spatialTarget";
import { spatialTargetToRegion, type TargetRegionTs } from "./targets";

export type GroundTargetWorkflow = "solver" | "variation";
export type GroundTargetWorkflowErrorCode = "AERIAL_TARGET_UNSUPPORTED";

/** Typed diagnostic used instead of flattening a 3D target onto the course surface. */
export class SpatialTargetWorkflowError extends Error {
  readonly name = "SpatialTargetWorkflowError";

  constructor(
    readonly code: GroundTargetWorkflowErrorCode,
    readonly workflow: GroundTargetWorkflow,
  ) {
    super(
      `${workflow} currently supports course-surface landing targets only; ` +
      "the aerial target is unavailable and its elevation was not coerced to zero.",
    );
  }
}

export interface GroundTargetWorkflowResult {
  readonly targetRegion: TargetRegionTs | null;
  readonly diagnostic: SpatialTargetWorkflowError | null;
}

/** Validate once, then project only targets that meet the ground-workflow contract. */
export function spatialTargetForGroundWorkflow(
  input: SpatialTargetTs,
  workflow: GroundTargetWorkflow,
): GroundTargetWorkflowResult {
  const target = createSpatialTarget(input);
  if (target.kind === "aerial_waypoint") {
    return {
      targetRegion: null,
      diagnostic: new SpatialTargetWorkflowError("AERIAL_TARGET_UNSUPPORTED", workflow),
    };
  }
  return { targetRegion: spatialTargetToRegion(target), diagnostic: null };
}
