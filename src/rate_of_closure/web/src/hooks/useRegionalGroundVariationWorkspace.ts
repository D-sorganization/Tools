import { useCallback, useMemo, useReducer } from "react";

import { DRIVER_TEE_HEIGHT_M } from "../model/ballSetup";
import { loadBallSetupPreference } from "../model/ballSetupPersistence";
import type { GroundRegionalMaterialPlanRequest } from "../model/groundRegionalPlan";
import {
  applyRegionalGroundVariationRequest,
  composeRegionalGroundVariationRequest,
  createRegionalGroundVariationWorkspaceState,
  regionalGroundVariationWorkspaceReducer,
  type RegionalGroundVariationRequestTs,
  type RegionalGroundVariationRequestPort,
  type RegionalGroundVariationWorkspaceState,
} from "../model/regionalGroundVariationWorkspace";
import type { RegionalSurfacePlanDraft } from "../model/regionalSurfacePlan";
import type { VariationPlanTs } from "../model/variation";
import type { VariationAnalysisExecution } from "../model/variationAnalysisPolicy";
import { defaultVariationPlan } from "../model/variationDefaults";

export interface RegionalGroundVariationWorkspaceController {
  readonly state: RegionalGroundVariationWorkspaceState;
  readonly replaceVariationPlan: (plan: VariationPlanTs) => void;
  readonly replaceAnalysisExecution: (value: VariationAnalysisExecution) => void;
  readonly replaceRegionalDraft: (draft: RegionalSurfacePlanDraft) => void;
  readonly applyRegionalImport: (request: GroundRegionalMaterialPlanRequest) => void;
  readonly requestPort: RegionalGroundVariationRequestPort;
}

const initialState = (storage?: Storage): RegionalGroundVariationWorkspaceState => {
  const setup = loadBallSetupPreference(
    storage,
    { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M },
  ).setup;
  return createRegionalGroundVariationWorkspaceState({
    ...defaultVariationPlan(),
    ballSetup: setup,
  });
};

/** Own both request editors above mutually exclusive workspace panels. */
export function useRegionalGroundVariationWorkspace(
  storage?: Storage,
): RegionalGroundVariationWorkspaceController {
  const [state, dispatch] = useReducer(
    regionalGroundVariationWorkspaceReducer,
    storage,
    initialState,
  );
  const replaceVariationPlan = useCallback((plan: VariationPlanTs) => {
    dispatch({ type: "replace_variation_plan", plan });
  }, []);
  const replaceAnalysisExecution = useCallback((value: VariationAnalysisExecution) => {
    dispatch({ type: "replace_analysis_execution", analysisExecution: value });
  }, []);
  const replaceRegionalDraft = useCallback((draft: RegionalSurfacePlanDraft) => {
    dispatch({ type: "replace_regional_draft", draft });
  }, []);
  const applyRegionalImport = useCallback((request: GroundRegionalMaterialPlanRequest) => {
    dispatch({ type: "apply_regional_import", request });
  }, []);
  const snapshot = useCallback(
    () => composeRegionalGroundVariationRequest(state),
    [state],
  );
  const apply = useCallback((request: RegionalGroundVariationRequestTs) => {
    applyRegionalGroundVariationRequest(state, request);
    dispatch({ type: "apply_request", request });
  }, [state]);
  const requestPort = useMemo(() => ({ snapshot, apply }), [snapshot, apply]);
  return {
    state,
    replaceVariationPlan,
    replaceAnalysisExecution,
    replaceRegionalDraft,
    applyRegionalImport,
    requestPort,
  };
}
