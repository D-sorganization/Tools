import type { VariationAnalysisExecution } from "./variationAnalysisPolicy";
import type { GroundRegionalMaterialPlanRequest } from "./groundRegionalPlan";
import {
  editorDraftFromGroundRegionalSurfacePlanRequest,
  illustrativeRegionalSurfacePlanDraft,
  regionalSurfacePlanRequestForDraft,
  type RegionalSurfacePlanDraft,
} from "./regionalSurfacePlan";
import {
  GROUND_NORMAL_RESTITUTION_KEY,
  GROUND_ROLLING_RESISTANCE_KEY,
  validatePlan,
  type VariationPlanTs,
} from "./variation";
import { defaultVariationPlan } from "./variationDefaults";

export {
  GROUND_NORMAL_RESTITUTION_KEY,
  GROUND_ROLLING_RESISTANCE_KEY,
};

export const MAX_REGIONAL_GROUND_STUDY_ROWS = 100_000;

export interface RegionalGroundVariationRequestIdentity {
  readonly resultId: string;
  readonly sourceProvenance: string;
  readonly maxRows: number;
  readonly seriesId: string | null;
}

export interface RegionalGroundVariationRequestTs {
  readonly plan: VariationPlanTs;
  readonly regionalPlan: GroundRegionalMaterialPlanRequest;
  readonly resultId: string;
  readonly sourceProvenance: string;
  readonly maxRows: number;
  readonly seriesId: string | null;
}

export interface RegionalGroundVariationRequestPort {
  readonly snapshot: () => RegionalGroundVariationRequestTs;
  readonly apply: (request: RegionalGroundVariationRequestTs) => void;
}

export interface RegionalGroundVariationWorkspaceState {
  readonly variationPlan: VariationPlanTs;
  readonly analysisExecution: VariationAnalysisExecution;
  readonly regionalDraft: RegionalSurfacePlanDraft;
  readonly importedRegionalRequest: GroundRegionalMaterialPlanRequest | null;
  readonly regionalDraftOrigin: "illustrative" | "user_edited" | "imported";
  readonly requestIdentity: RegionalGroundVariationRequestIdentity;
}

export type RegionalGroundVariationWorkspaceAction =
  | { readonly type: "replace_variation_plan"; readonly plan: VariationPlanTs }
  | {
    readonly type: "replace_analysis_execution";
    readonly analysisExecution: VariationAnalysisExecution;
  }
  | { readonly type: "replace_regional_draft"; readonly draft: RegionalSurfacePlanDraft }
  | {
    readonly type: "apply_regional_import";
    readonly request: GroundRegionalMaterialPlanRequest;
  }
  | { readonly type: "apply_request"; readonly request: RegionalGroundVariationRequestTs };

const DEFAULT_IDENTITY: RegionalGroundVariationRequestIdentity = Object.freeze({
  resultId: "regional-ground-variation-result",
  sourceProvenance: "rate-web-interactive-workspace",
  maxRows: 500,
  seriesId: null,
});

const exactKeys = (value: Record<string, number>, expected: readonly string[]): boolean => {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === expected.length &&
    actual.every((key, index) => key === wanted[index]);
};

const nonblank = (value: string, name: string): void => {
  if (value.trim().length === 0) throw new RangeError(`${name} must be nonempty`);
};

const validateGroundPlan = (
  plan: VariationPlanTs,
  regionalPlan: GroundRegionalMaterialPlanRequest,
): void => {
  validatePlan(plan);
  if (plan.mode !== "launch") {
    throw new RangeError("ground material variation requires launch mode");
  }
  if (plan.ballSetup !== undefined) {
    throw new RangeError("regional-ground request plans cannot include ballSetup");
  }
  const supported = [
    GROUND_NORMAL_RESTITUTION_KEY,
    GROUND_ROLLING_RESISTANCE_KEY,
  ];
  if (!exactKeys(plan.baseVariables, supported)) {
    throw new RangeError("unsupported ground base key");
  }
  for (const spec of plan.noise) {
    if (!supported.includes(spec.variableKey)) {
      throw new RangeError("unsupported ground variation key");
    }
    if ((spec.timeWindowS !== null && spec.timeWindowS !== undefined) ||
        (spec.pointIds?.length ?? 0) > 0) {
      throw new RangeError("ground material variation must be global");
    }
    if (spec.lower === null || spec.upper === null ||
        spec.lower < 0 || spec.lower >= spec.upper || spec.upper > 1) {
      throw new RangeError("ground material bounds must be explicit within [0, 1]");
    }
    const base = plan.baseVariables[spec.variableKey];
    if (base < spec.lower || base > spec.upper) {
      throw new RangeError("ground material base must lie within bounds");
    }
  }
  if (plan.baseVariables[GROUND_NORMAL_RESTITUTION_KEY] !==
      regionalPlan.base_surface.normal_restitution) {
    throw new RangeError("normal restitution base does not match regional plan");
  }
  if (plan.baseVariables[GROUND_ROLLING_RESISTANCE_KEY] !==
      regionalPlan.base_surface.rolling_resistance) {
    throw new RangeError("rolling resistance base does not match regional plan");
  }
};

const validateRequest = (request: RegionalGroundVariationRequestTs): void => {
  nonblank(request.resultId, "resultId");
  nonblank(request.sourceProvenance, "sourceProvenance");
  if (!Number.isInteger(request.maxRows) || request.maxRows < 1 ||
      request.maxRows > MAX_REGIONAL_GROUND_STUDY_ROWS) {
    throw new RangeError("maxRows is outside the supported range");
  }
  if (request.seriesId !== null) nonblank(request.seriesId, "seriesId");
  if (request.plan.nRuns > request.maxRows) {
    throw new RangeError("plan nRuns exceeds maxRows");
  }
  validateGroundPlan(request.plan, request.regionalPlan);
};

/** Build one fresh owner state for both editors. */
export const createRegionalGroundVariationWorkspaceState = (
  variationPlan: VariationPlanTs = defaultVariationPlan(),
): RegionalGroundVariationWorkspaceState => ({
  variationPlan,
  analysisExecution: "both",
  regionalDraft: illustrativeRegionalSurfacePlanDraft(),
  importedRegionalRequest: null,
  regionalDraftOrigin: "illustrative",
  requestIdentity: DEFAULT_IDENTITY,
});

/** Compose and validate the exact current editor state without fallback values. */
export const composeRegionalGroundVariationRequest = (
  state: RegionalGroundVariationWorkspaceState,
): RegionalGroundVariationRequestTs => {
  if (state.regionalDraftOrigin === "illustrative") {
    throw new RangeError(
      "regional plan must be explicitly edited or imported before composition",
    );
  }
  const request = {
    plan: state.variationPlan,
    regionalPlan: regionalSurfacePlanRequestForDraft(
      state.regionalDraft,
      state.importedRegionalRequest,
    ),
    ...state.requestIdentity,
  };
  validateRequest(request);
  return request;
};

/** Validate a complete request before returning the replacement owner state. */
export const applyRegionalGroundVariationRequest = (
  previous: RegionalGroundVariationWorkspaceState,
  request: RegionalGroundVariationRequestTs,
): RegionalGroundVariationWorkspaceState => {
  validateRequest(request);
  const draft = editorDraftFromGroundRegionalSurfacePlanRequest(request.regionalPlan);
  return {
    ...previous,
    variationPlan: request.plan,
    regionalDraft: draft,
    importedRegionalRequest: request.regionalPlan,
    regionalDraftOrigin: "imported",
    requestIdentity: {
      resultId: request.resultId,
      sourceProvenance: request.sourceProvenance,
      maxRows: request.maxRows,
      seriesId: request.seriesId,
    },
  };
};

/** Pure transactional state transition shared by the App-owned hook and tests. */
export const regionalGroundVariationWorkspaceReducer = (
  state: RegionalGroundVariationWorkspaceState,
  action: RegionalGroundVariationWorkspaceAction,
): RegionalGroundVariationWorkspaceState => {
  switch (action.type) {
    case "replace_variation_plan":
      return { ...state, variationPlan: action.plan };
    case "replace_analysis_execution":
      return { ...state, analysisExecution: action.analysisExecution };
    case "replace_regional_draft":
      return {
        ...state,
        regionalDraft: action.draft,
        regionalDraftOrigin: "user_edited",
      };
    case "apply_regional_import":
      {
        const draft = editorDraftFromGroundRegionalSurfacePlanRequest(action.request);
        return {
          ...state,
          regionalDraft: draft,
          importedRegionalRequest: action.request,
          regionalDraftOrigin: "imported",
        };
      }
    case "apply_request":
      return applyRegionalGroundVariationRequest(state, action.request);
  }
};
