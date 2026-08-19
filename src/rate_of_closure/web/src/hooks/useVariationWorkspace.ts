import { useState, type Dispatch, type SetStateAction } from "react";

import type { BallSetup } from "../model/ballSetup";
import { outputsForMode, type VariationPlanTs } from "../model/variation";
import { defaultVariationPlan } from "../model/variationDefaults";
import {
  validatedVariationWorkspace,
  type VariationAnalysisExecution,
  type VariationWorkspaceSnapshot,
} from "../model/workspaceVariationSession";

export interface ControlledVariationWorkspaceProps {
  readonly variationWorkspace?: VariationWorkspaceSnapshot;
  readonly onVariationWorkspaceChange?: Dispatch<
    SetStateAction<VariationWorkspaceSnapshot>
  >;
}

export function initialVariationWorkspace(
  ballSetup?: BallSetup,
): VariationWorkspaceSnapshot {
  const plan = defaultVariationPlan(ballSetup);
  return validatedVariationWorkspace(
    {
      plan,
      analysisExecution: "both",
      selectedOutputMetrics: outputsForMode(plan.mode),
    },
    ballSetup,
  );
}

/** Own variation authoring state locally or bridge the app-level authority. */
export function useVariationWorkspace(
  controlled: ControlledVariationWorkspaceProps,
  ballSetup?: BallSetup,
) {
  const [internal, setInternal] = useState(() =>
    initialVariationWorkspace(ballSetup),
  );
  const state = controlled.variationWorkspace ?? internal;
  const setState = controlled.onVariationWorkspaceChange ?? setInternal;
  const setPlan = (plan: VariationPlanTs) => {
    setState((current) => {
      const available = outputsForMode(plan.mode);
      const retained = current.selectedOutputMetrics.filter((metric) =>
        available.includes(metric),
      );
      return validatedVariationWorkspace(
        {
          ...current,
          plan,
          selectedOutputMetrics: retained.length > 0 ? retained : available,
        },
        ballSetup,
      );
    });
  };
  const setAnalysisExecution = (
    analysisExecution: VariationAnalysisExecution,
  ) => {
    setState((current) =>
      validatedVariationWorkspace(
        {
          ...current,
          analysisExecution,
        },
        ballSetup,
      ),
    );
  };
  const setSelectedOutputMetrics = (
    selectedOutputMetrics: readonly string[],
  ) => {
    setState((current) =>
      validatedVariationWorkspace(
        {
          ...current,
          selectedOutputMetrics,
        },
        ballSetup,
      ),
    );
  };
  return {
    state,
    setPlan,
    setAnalysisExecution,
    setSelectedOutputMetrics,
  };
}
