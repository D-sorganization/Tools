/** Canonical UI defaults for a new, unsaved variation study. */

import type { BallSetup } from "./ballSetup";
import {
  keysForMode,
  variableDef,
  type NoiseSpecTs,
  type VariationMode,
  type VariationPlanTs,
} from "./variation";

/** Build the first available independent noise specification for one mode. */
export const defaultVariationSpec = (
  mode: VariationMode,
  excluded: ReadonlySet<string> = new Set(),
  ballSetup?: BallSetup,
): NoiseSpecTs => {
  const keys = keysForMode(mode, ballSetup);
  const key = keys.find((candidate) => !excluded.has(candidate)) ?? keys[0];
  return {
    variableKey: key,
    distribution: "normal",
    scale: variableDef(key)?.typicalScale ?? 1,
    lower: null,
    upper: null,
    specId: key,
    timeWindowS: null,
    pointIds: [],
  };
};

export const defaultVariationPlan = (
  ballSetup?: BallSetup,
): VariationPlanTs => ({
  mode: "delivery",
  baseVariables: {},
  noise: [defaultVariationSpec("delivery", new Set(), ballSetup)],
  nRuns: 200,
  seed: 0,
  flightModel: "waterloo_penner",
  groups: [],
  ...(ballSetup === undefined ? {} : { ballSetup }),
});
