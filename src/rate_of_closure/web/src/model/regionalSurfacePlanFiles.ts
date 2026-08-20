/** Browser-only strict import/download boundary for regional plan requests. */

import {
  MAX_GROUND_REGIONAL_PLAN_WIRE_BYTES,
  groundRegionalMaterialPlanRequestFromJson,
  stableGroundRegionalMaterialPlanJson,
  type GroundRegionalMaterialPlanRequest,
} from "./groundRegionalPlan";

export interface RegionalPlanFile {
  readonly name: string;
  readonly size: number;
  text(): Promise<string>;
}

export const readRegionalSurfacePlanFile = async (
  file: RegionalPlanFile,
): Promise<GroundRegionalMaterialPlanRequest> => {
  if (file.size > MAX_GROUND_REGIONAL_PLAN_WIRE_BYTES) {
    throw new RangeError("regional material plan exceeds maximum wire size");
  }
  return groundRegionalMaterialPlanRequestFromJson(await file.text());
};

export const downloadRegionalSurfacePlanRequest = (
  request: GroundRegionalMaterialPlanRequest,
): void => {
  const text = stableGroundRegionalMaterialPlanJson(request);
  const url = URL.createObjectURL(
    new Blob([text], { type: "application/json;charset=utf-8" }),
  );
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "regional-surface-plan.json";
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
};
