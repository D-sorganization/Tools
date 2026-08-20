/** Complete synchronous capability run used only inside worker/test boundaries. */

import {
  CapabilityObservationEnsembleBuilder,
  type CapabilityCohort,
} from "./capabilityObservationEnsemble";
import { makeCapabilityFlightEvaluator } from "./capabilityFlightEvaluator";
import { optimizeCapability } from "./capabilityOptimizer";
import type { OptimizationResult } from "./capabilityContract";
import type { ScalarEnsembleResult } from "./scalarEnsembleContract";
import type { CapabilityWorkflowDocument } from "./capabilityWorkflow";

export interface CapabilityProgress {
  readonly completed: number; readonly total: number;
}

export interface CapabilityRunOutput {
  readonly result: OptimizationResult;
  readonly ensemble: ScalarEnsembleResult<CapabilityCohort>;
}

/** Run one immutable workflow; callers must place this CPU work off the UI thread. */
export function runCapabilityOptimization(
  document: CapabilityWorkflowDocument,
  onProgress: (progress: CapabilityProgress) => void = () => undefined,
): CapabilityRunOutput {
  const total = document.request.candidateBudget * document.request.ensembleSize;
  const builder = new CapabilityObservationEnsembleBuilder({
    target: document.request.target, maxRows: total,
    sourceProvenance: document.profile.provenance,
  });
  const evaluator = makeCapabilityFlightEvaluator(
    document.profile, document.request, document.evaluatorConfig,
  );
  let lastReported = 0;
  const result = optimizeCapability(document.profile, document.request, evaluator, {
    observationSink: (observation) => {
      builder.accept(observation);
      const completed = observation.attemptedCount;
      const interval = Math.max(1, Math.floor(total / 100));
      if (completed === total || completed === 1 || completed - lastReported >= interval) {
        lastReported = completed;
        onProgress({ completed, total });
      }
    },
  });
  return Object.freeze({ result, ensemble: builder.build() });
}
