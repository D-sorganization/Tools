import { describe, expect, it } from "vitest";

import {
  AUTHORITY_FLIGHT_MODELS,
  type AuthorityFlightModel,
} from "./morrisAuthorityRequest";
import {
  BROWSER_VARIATION_FLIGHT_MODELS,
  SUPPORTED_VARIATION_FLIGHT_MODEL,
  validatePlan,
  type VariationPlanTs,
} from "./variationSchema";
import { CATEGORY_LAUNCH } from "./variation";

const BALL = `${CATEGORY_LAUNCH}.ball_speed_mph`;

const validBasePlan = (flightModel: string): VariationPlanTs => ({
  mode: "launch",
  baseVariables: { [BALL]: 160 },
  noise: [
    {
      variableKey: BALL,
      distribution: "normal",
      scale: 2,
      lower: null,
      upper: null,
    },
  ],
  nRuns: 10,
  seed: 42,
  flightModel,
});

describe("variation flight model policy pinning (#4457)", () => {
  it("derives browser variation supported model as an element of authority flight models", () => {
    expect(SUPPORTED_VARIATION_FLIGHT_MODEL).toBe("waterloo_penner");
    expect(AUTHORITY_FLIGHT_MODELS).toContain(SUPPORTED_VARIATION_FLIGHT_MODEL);
    expect(BROWSER_VARIATION_FLIGHT_MODELS).toEqual([SUPPORTED_VARIATION_FLIGHT_MODEL]);
    expect(AUTHORITY_FLIGHT_MODELS).toHaveLength(7);
  });

  it("accepts waterloo_penner in browser variation plan validation", () => {
    const plan = validBasePlan(SUPPORTED_VARIATION_FLIGHT_MODEL);
    expect(() => validatePlan(plan)).not.toThrow();
  });

  it.each(
    AUTHORITY_FLIGHT_MODELS.filter(
      (model): model is Exclude<AuthorityFlightModel, "waterloo_penner"> =>
        model !== SUPPORTED_VARIATION_FLIGHT_MODEL,
    ),
  )(
    "honestly rejects authority flight model %s with descriptive guidance for browser variation",
    (flightModel) => {
      const plan = validBasePlan(flightModel);
      expect(() => validatePlan(plan)).toThrow(
        `Flight model "${flightModel}" is unsupported for browser variation; ` +
          `supported model: waterloo_penner. Recreate or edit the stored plan.`,
      );
    },
  );

  it.each(["custom-flight-model", "unknown_aerodynamics", "", "waterloo-penner"])(
    "rejects unregistered flight model %s with descriptive guidance",
    (flightModel) => {
      const plan = validBasePlan(flightModel);
      expect(() => validatePlan(plan)).toThrow(
        `Flight model "${flightModel}" is unsupported for browser variation; ` +
          `supported model: waterloo_penner. Recreate or edit the stored plan.`,
      );
    },
  );

  it("documents asymmetry: browser variation is a strict subset of Morris authority models", () => {
    // Every model accepted by browser variation is valid in the Morris authority
    for (const model of BROWSER_VARIATION_FLIGHT_MODELS) {
      expect(AUTHORITY_FLIGHT_MODELS).toContain(model);
    }
    // But Morris authority supports the full set of 7 models supported by Python
    expect(AUTHORITY_FLIGHT_MODELS.length).toBeGreaterThan(
      BROWSER_VARIATION_FLIGHT_MODELS.length,
    );
  });
});
