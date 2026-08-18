import { describe, expect, it } from "vitest";

import matrixDocument from "./__fixtures__/variation_visual_state_matrix_v1.json";
import {
  parseVariationVisualStateMatrix,
  variationExecutionIdentity,
  variationVisualState,
  type VariationVisualEvent,
} from "./variationVisualState";
import { defaultVariationPlan } from "../components/variationUi";

describe("variation visual state contract", () => {
  it("consumes every strict shared transition", () => {
    const matrix = parseVariationVisualStateMatrix(matrixDocument);
    expect(matrix).toHaveLength(8);
    for (const row of matrixDocument.states) {
      expect(variationVisualState(row.event as VariationVisualEvent)).toEqual({
        phase: row.phase,
        visualOrigin: row.visual_origin,
        announcementRole: row.announcement_role,
      });
    }
  });

  it.each([
    { ...matrixDocument, extra: true },
    { ...matrixDocument, schema_version: true },
    { ...matrixDocument, states: matrixDocument.states.map((row, index) => (
      index === 0 ? { ...row, phase: "busy" } : row
    )) },
    { ...matrixDocument, states: [...matrixDocument.states, matrixDocument.states[0]] },
  ])("rejects unknown, coercive, and duplicate data", (document) => {
    expect(() => parseVariationVisualStateMatrix(document)).toThrow();
  });

  it("keys every plan and analysis policy field without presentation-only inputs", () => {
    const plan = defaultVariationPlan();
    const baseline = variationExecutionIdentity(plan, "all_together");
    expect(baseline).toBe(variationExecutionIdentity(plan, "all_together"));
    expect(baseline).not.toBe(variationExecutionIdentity({ ...plan, seed: plan.seed + 1 }, "all_together"));
    expect(baseline).not.toBe(variationExecutionIdentity(plan, "both"));
  });

  it("returns a deeply immutable parsed authority", () => {
    const matrix = parseVariationVisualStateMatrix(matrixDocument);
    expect(Object.isFrozen(matrix)).toBe(true);
    expect(Object.isFrozen(matrix[0])).toBe(true);
    expect(Object.isFrozen(matrix[0]?.[1])).toBe(true);
    expect(() => (matrix as unknown[]).push([])).toThrow();
  });

  it("returns a non-accepted sentinel for transient invalid editor states", () => {
    const plan = defaultVariationPlan();
    const invalid = {
      ...plan,
      noise: [{ ...plan.noise[0], scale: 0 }],
    };
    expect(() => variationExecutionIdentity(invalid, "all_together")).not.toThrow();
    expect(variationExecutionIdentity(invalid, "all_together")).toMatch(/^invalid:/);
    expect(variationExecutionIdentity(plan, "all_together")).toMatch(/^valid:/);
  });
});
