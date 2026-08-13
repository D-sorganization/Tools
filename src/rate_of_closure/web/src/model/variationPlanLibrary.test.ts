import { describe, expect, it } from "vitest";

import { CATEGORY_LAUNCH, planFromJson, planToJson, type VariationPlanTs } from "./variation";
import {
  VARIATION_PLAN_LIBRARY_KEY,
  VARIATION_PLAN_LIBRARY_VERSION,
  deleteVariationPlan,
  duplicateVariationPlan,
  loadVariationPlanLibrary,
  saveVariationPlanLibrary,
  type NamedVariationPlan,
} from "./variationPlanLibrary";

class MemoryStorage implements Storage {
  private values = new Map<string, string>();
  get length() { return this.values.size; }
  clear() { this.values.clear(); }
  getItem(key: string) { return this.values.get(key) ?? null; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string) { this.values.delete(key); }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

const BALL = `${CATEGORY_LAUNCH}.ball_speed_mph`;
const ANGLE = `${CATEGORY_LAUNCH}.launch_angle_deg`;

const fullPlan = (): VariationPlanTs => ({
  mode: "launch",
  baseVariables: { [BALL]: 161.25, [ANGLE]: 13.5 },
  noise: [
    {
      variableKey: BALL,
      distribution: "normal",
      scale: 2,
      lower: 150,
      upper: 170,
      specId: "speed-at-impact",
      timeWindowS: [0.72, 0.78],
      pointIds: ["swing.clubhead"],
    },
    {
      variableKey: ANGLE,
      distribution: "normal",
      scale: 1,
      lower: null,
      upper: null,
      specId: "launch-angle",
      timeWindowS: null,
      pointIds: [],
    },
  ],
  groups: [
    {
      groupId: "impact-correlation",
      specIds: ["speed-at-impact", "launch-angle"],
      matrixKind: "correlation",
      matrix: [
        [1, -0.35],
        [-0.35, 1],
      ],
    },
  ],
  nRuns: 24,
  seed: 17,
  flightModel: "waterloo_penner",
});

const entry = (id = "plan-1", name = "Impact Window"): NamedVariationPlan => ({
  id,
  name,
  plan: fullPlan(),
});

describe("variation plan library", () => {
  it("round-trips a named plan with complete v2 fidelity", () => {
    const storage = new MemoryStorage();
    saveVariationPlanLibrary([entry()], storage);

    const loaded = loadVariationPlanLibrary(storage);

    expect(loaded.warnings).toEqual([]);
    expect(loaded.plans).toHaveLength(1);
    expect(loaded.plans[0]).toEqual({
      ...entry(),
      plan: planFromJson(planToJson(fullPlan())),
    });
    const raw = JSON.parse(storage.getItem(VARIATION_PLAN_LIBRARY_KEY)!) as {
      schema_version: number;
    };
    expect(raw.schema_version).toBe(VARIATION_PLAN_LIBRARY_VERSION);
  });

  it("recovers safely from corrupt JSON and unsupported library versions", () => {
    const storage = new MemoryStorage();
    storage.setItem(VARIATION_PLAN_LIBRARY_KEY, "{broken");
    expect(loadVariationPlanLibrary(storage)).toMatchObject({ plans: [] });
    expect(loadVariationPlanLibrary(storage).warnings[0]).toMatch(/corrupt/i);

    storage.setItem(
      VARIATION_PLAN_LIBRARY_KEY,
      JSON.stringify({ schema_version: 99, plans: [] }),
    );
    expect(loadVariationPlanLibrary(storage)).toMatchObject({ plans: [] });
    expect(loadVariationPlanLibrary(storage).warnings[0]).toMatch(/unsupported/i);
  });

  it("keeps valid entries when another stored entry is partial or invalid", () => {
    const storage = new MemoryStorage();
    const validPlanObject = JSON.parse(planToJson(fullPlan())) as unknown;
    storage.setItem(
      VARIATION_PLAN_LIBRARY_KEY,
      JSON.stringify({
        schema_version: VARIATION_PLAN_LIBRARY_VERSION,
        plans: [
          { id: "valid", name: "Valid Plan", plan: validPlanObject },
          { id: "partial", name: "Missing Plan" },
          { id: "", name: "Bad ID", plan: validPlanObject },
        ],
      }),
    );

    const loaded = loadVariationPlanLibrary(storage);
    expect(loaded.plans.map((plan) => plan.id)).toEqual(["valid"]);
    expect(loaded.warnings).toHaveLength(2);
  });

  it("ignores an unsupported persisted model with an explicit migration warning", () => {
    const storage = new MemoryStorage();
    const unsupported = JSON.parse(planToJson(fullPlan())) as Record<string, unknown>;
    unsupported.flight_model = "custom-flight-model";
    storage.setItem(
      VARIATION_PLAN_LIBRARY_KEY,
      JSON.stringify({
        schema_version: VARIATION_PLAN_LIBRARY_VERSION,
        plans: [{ id: "legacy", name: "Legacy Plan", plan: unsupported }],
      }),
    );

    const loaded = loadVariationPlanLibrary(storage);

    expect(loaded.plans).toEqual([]);
    expect(loaded.warnings).toEqual([
      expect.stringMatching(/ignored.*flight model.*custom-flight-model.*waterloo_penner/i),
    ]);
  });

  it("reports unreadable storage without crashing the workbench", () => {
    const storage = {
      getItem: () => { throw new Error("access denied"); },
      setItem: () => undefined,
    } as unknown as Storage;

    expect(loadVariationPlanLibrary(storage)).toEqual({
      plans: [],
      warnings: ["Stored plan library could not be read: access denied"],
    });
  });

  it("duplicates and deletes without mutating the source collection", () => {
    const source = [entry()];
    const duplicated = duplicateVariationPlan(source, "plan-1", "plan-2");
    expect(source).toHaveLength(1);
    expect(duplicated).toHaveLength(2);
    expect(duplicated[1]).toMatchObject({ id: "plan-2", name: "Impact Window Copy" });
    expect(duplicated[1].plan).toEqual(duplicated[0].plan);
    expect(duplicated[1].plan).not.toBe(duplicated[0].plan);
    expect(deleteVariationPlan(duplicated, "plan-1").map((plan) => plan.id)).toEqual([
      "plan-2",
    ]);
  });
});
