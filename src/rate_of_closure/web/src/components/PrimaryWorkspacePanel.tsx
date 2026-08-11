import { lazy, Suspense } from "react";

import type { ImpactAppModel } from "../hooks/useImpactAppModel";
import type { PrimaryViewId } from "../model/viewPreferences";
import type { ViewWorkspace } from "../model/viewWorkspace";
import { Derivation } from "./Derivation";
import { FlightExplorerPanel } from "./FlightExplorerPanel";
import { GlossaryPanel } from "./GlossaryPanel";
import { ImpactExplorerPanel } from "./ImpactExplorerPanel";
import { LaunchMonitorAnalyticsPanel } from "./LaunchMonitorAnalyticsPanel";
import { PlotsPanel } from "./PlotsPanel";
import { PuttingPanel } from "./PuttingPanel";
import { SimulationPanel } from "./SimulationPanel";
import { VariationPanel } from "./VariationPanel";

const LazyCapabilityOptimizationPanel = lazy(() =>
  import("./CapabilityOptimizationPanel").then((module) => ({
    default: module.CapabilityOptimizationPanel,
  })),
);

interface WorkspacePanelProps {
  readonly active: PrimaryViewId;
  readonly model: ImpactAppModel;
  readonly onOpenGlossary: (term: string | undefined) => void;
  readonly viewWorkspace: ViewWorkspace;
  readonly viewCommandRevision: number;
  readonly onViewWorkspaceChange: (workspace: ViewWorkspace) => void;
}

function SimulationWorkspace(props: {
  readonly model: ImpactAppModel;
  readonly viewWorkspace: ViewWorkspace;
  readonly viewCommandRevision: number;
  readonly onViewWorkspaceChange: (workspace: ViewWorkspace) => void;
}) {
  const { model } = props;
  return (
    <SimulationPanel
      scenario={model.scenario}
      loftDeg={10.5}
      clubSpec={model.clubSpec}
      onScenarioChange={(updates) =>
        model.setScenario((state) => ({ ...state, ...updates }))
      }
      spatialTarget={model.spatialTarget}
      onSpatialTargetChange={model.setSpatialTarget}
      ballSetup={model.ballSetup}
      ballSetupUserOverridden={model.ballSetupUserOverridden}
      ballSetupMessage={model.ballSetupMessage}
      onBallSetupChange={model.setBallSetup}
      onBallSetupUserOverriddenChange={model.setBallSetupUserOverridden}
      onBallSetupMessageChange={model.setBallSetupMessage}
      torqueWorkspace={model.torqueWorkspace}
      onTorqueWorkspaceChange={model.setTorqueWorkspace}
      distanceUnit={model.units.distance}
      viewWorkspace={props.viewWorkspace}
      viewCommandRevision={props.viewCommandRevision}
      onViewWorkspaceChange={props.onViewWorkspaceChange}
    />
  );
}

function ExplorerWorkspace(
  props: Pick<WorkspacePanelProps, "model" | "onOpenGlossary">,
) {
  const { model } = props;
  return (
    <ImpactExplorerPanel
      scenario={model.scenario}
      setScenario={model.setScenario}
      units={model.units}
      setUnits={model.setUnits}
      clubSpec={model.clubSpec}
      setClubSpec={model.setClubSpec}
      generatedHead={model.generatedHead}
      setGeneratedHead={model.setGeneratedHead}
      explained={model.explained}
      onExplainedChange={model.setExplained}
      onOpenGlossary={props.onOpenGlossary}
    />
  );
}

export function PrimaryWorkspacePanel(props: WorkspacePanelProps) {
  const { active, model, onOpenGlossary } = props;
  switch (active) {
    case "glossary":
      return (
        <GlossaryPanel
          key={model.glossaryTerm ?? "none"}
          initialTerm={model.glossaryTerm}
        />
      );
    case "putting":
      return (
        <PuttingPanel
          distanceUnit={model.units.distance}
          onGlossary={onOpenGlossary}
        />
      );
    case "variation":
      return (
        <VariationPanel
          spatialTarget={model.spatialTarget}
          distanceUnit={model.units.distance}
          ballSetup={model.ballSetup}
          variationWorkspace={model.variationWorkspace}
          onVariationWorkspaceChange={model.setVariationWorkspace}
        />
      );
    case "flight":
      return (
        <FlightExplorerPanel
          distanceUnit={model.units.distance}
          spatialTarget={model.spatialTarget}
          onSpatialTargetChange={model.setSpatialTarget}
        />
      );
    case "launch-monitor-analytics":
      return <LaunchMonitorAnalyticsPanel />;
    case "capability-optimization":
      return (
        <Suspense
          fallback={
            <section role="status" aria-label="Shot optimizer loading">
              Loading…
            </section>
          }
        >
          <LazyCapabilityOptimizationPanel />
        </Suspense>
      );
    case "plots":
      return <PlotsPanel scenario={model.scenario} loftDeg={10.5} />;
    case "simulation":
      return (
        <SimulationWorkspace
          model={model}
          viewWorkspace={props.viewWorkspace}
          viewCommandRevision={props.viewCommandRevision}
          onViewWorkspaceChange={props.onViewWorkspaceChange}
        />
      );
    case "calculation":
      return <Derivation scenario={model.scenario} />;
    default:
      return (
        <ExplorerWorkspace model={model} onOpenGlossary={onOpenGlossary} />
      );
  }
}
