/** Canonical-target boundary around the solver's legacy ground-region API. */

import type { ImpactScenario } from "../model/impact";
import type { SpatialTargetTs } from "../model/spatialTarget";
import { spatialTargetForGroundWorkflow } from "../model/spatialTargetWorkflow";
import { SolverPanel } from "./SolverPanel";

interface Props {
  readonly onApply: (updates: Partial<ImpactScenario>) => void;
  readonly spatialTarget: SpatialTargetTs;
}

export function SpatialTargetSolverPanel({
  onApply,
  spatialTarget,
}: Props): JSX.Element {
  const targetUse = spatialTargetForGroundWorkflow(spatialTarget, "solver");
  return (
    <>
      <SolverPanel onApply={onApply} target={targetUse.targetRegion ?? undefined} />
      {targetUse.diagnostic && (
        <p role="status" aria-label="Solver spatial target compatibility"
          className="rounded-lg border border-amber-400/40 bg-amber-950/20 p-3 text-xs text-amber-200">
          {targetUse.diagnostic.message}
        </p>
      )}
    </>
  );
}
