import { describe, expect, it } from "vitest";

import { illustrativeRegionalSurfacePlanDraft } from "./regionalSurfacePlan";
import {
  GROUND_NORMAL_RESTITUTION_KEY,
  GROUND_ROLLING_RESISTANCE_KEY,
  applyRegionalGroundVariationRequest,
  composeRegionalGroundVariationRequest,
  createRegionalGroundVariationWorkspaceState,
  regionalGroundVariationWorkspaceReducer,
  type RegionalGroundVariationRequestTs,
} from "./regionalGroundVariationWorkspace";
import type { VariationPlanTs } from "./variation";

const groundPlan = (runs = 12): VariationPlanTs => ({
  mode: "launch",
  baseVariables: {
    [GROUND_NORMAL_RESTITUTION_KEY]: 0.42,
    [GROUND_ROLLING_RESISTANCE_KEY]: 0.04,
  },
  noise: [
    {
      variableKey: GROUND_NORMAL_RESTITUTION_KEY,
      distribution: "normal",
      scale: 0.02,
      lower: 0.1,
      upper: 0.8,
      specId: GROUND_NORMAL_RESTITUTION_KEY,
      timeWindowS: null,
      pointIds: [],
    },
  ],
  nRuns: runs,
  seed: 91,
  flightModel: "waterloo_penner",
  groups: [],
});

const request = (): RegionalGroundVariationRequestTs => {
  const state = createRegionalGroundVariationWorkspaceState();
  let prepared = regionalGroundVariationWorkspaceReducer(state, {
    type: "replace_variation_plan",
    plan: groundPlan(),
  });
  prepared = regionalGroundVariationWorkspaceReducer(prepared, {
    type: "replace_regional_draft",
    draft: prepared.regionalDraft,
  });
  return composeRegionalGroundVariationRequest(prepared);
};

describe("regional ground variation workspace", () => {
  it("never composes the disclosed illustrative draft as user evidence", () => {
    const state = regionalGroundVariationWorkspaceReducer(
      createRegionalGroundVariationWorkspaceState(),
      { type: "replace_variation_plan", plan: groundPlan() },
    );

    expect(() => composeRegionalGroundVariationRequest(state))
      .toThrow(/explicitly edited or imported/i);
  });

  it("composes the current variation and regional editors without substituting state", () => {
    let state = createRegionalGroundVariationWorkspaceState();
    const editedDraft = {
      ...illustrativeRegionalSurfacePlanDraft(),
      request_id: "user-edited-course",
    };
    state = regionalGroundVariationWorkspaceReducer(state, {
      type: "replace_variation_plan",
      plan: groundPlan(),
    });
    state = regionalGroundVariationWorkspaceReducer(state, {
      type: "replace_regional_draft",
      draft: editedDraft,
    });

    const snapshot = composeRegionalGroundVariationRequest(state);

    expect(snapshot.plan).toBe(state.variationPlan);
    expect(snapshot.regionalPlan.request_id).toBe("user-edited-course");
    expect(snapshot.regionalPlan.base_surface.normal_restitution).toBe(0.42);
    expect(snapshot.resultId).toBe(state.requestIdentity.resultId);
  });

  it("applies a complete request transactionally and retains exact import evidence", () => {
    const imported = request();
    const previous = createRegionalGroundVariationWorkspaceState();

    const next = applyRegionalGroundVariationRequest(previous, imported);

    expect(next.variationPlan).toBe(imported.plan);
    expect(next.importedRegionalRequest).toBe(imported.regionalPlan);
    expect(next.regionalDraftOrigin).toBe("imported");
    expect(next.regionalDraft.request_id).toBe(imported.regionalPlan.request_id);
    expect(next.requestIdentity).toEqual({
      resultId: imported.resultId,
      sourceProvenance: imported.sourceProvenance,
      maxRows: imported.maxRows,
      seriesId: imported.seriesId,
    });
  });

  it("retains an exact regional import while deriving its controlled draft", () => {
    const imported = request().regionalPlan;
    const state = createRegionalGroundVariationWorkspaceState();

    const next = regionalGroundVariationWorkspaceReducer(state, {
      type: "apply_regional_import",
      request: imported,
    });

    expect(next.importedRegionalRequest).toBe(imported);
    expect(next.regionalDraft.request_id).toBe(imported.request_id);
  });

  it("rejects an invalid apply before changing any workspace field", () => {
    const previous = createRegionalGroundVariationWorkspaceState();
    const invalid = { ...request(), maxRows: 1 };

    expect(() => applyRegionalGroundVariationRequest(previous, invalid))
      .toThrow(/nRuns exceeds maxRows/i);
    expect(previous).toEqual(createRegionalGroundVariationWorkspaceState());
  });

  it("rejects a mismatched material base when composing", () => {
    let state = createRegionalGroundVariationWorkspaceState();
    state = regionalGroundVariationWorkspaceReducer(state, {
      type: "replace_regional_draft",
      draft: state.regionalDraft,
    });
    state = regionalGroundVariationWorkspaceReducer(state, {
      type: "replace_variation_plan",
      plan: {
        ...groundPlan(),
        baseVariables: {
          [GROUND_NORMAL_RESTITUTION_KEY]: 0.7,
          [GROUND_ROLLING_RESISTANCE_KEY]: 0.04,
        },
      },
    });

    expect(() => composeRegionalGroundVariationRequest(state))
      .toThrow(/normal restitution base does not match/i);
  });
});
