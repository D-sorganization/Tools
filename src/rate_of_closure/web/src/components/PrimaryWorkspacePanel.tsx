import type { ImpactAppModel } from "../hooks/useImpactAppModel";
import type { PrimaryViewId } from "../model/viewPreferences";
import { Derivation } from "./Derivation";
import { FlightExplorerPanel } from "./FlightExplorerPanel";
import { GlossaryPanel } from "./GlossaryPanel";
import { ImpactExplorerPanel } from "./ImpactExplorerPanel";
import { LaunchMonitorAnalyticsPanel } from "./LaunchMonitorAnalyticsPanel";
import { PlotsPanel } from "./PlotsPanel";
import { PuttingPanel } from "./PuttingPanel";
import { SimulationPanel } from "./SimulationPanel";
import { VariationPanel } from "./VariationPanel";
import type { MorrisAuthorityClient } from "../model/morrisAuthorityClient";
import { defaultMorrisAuthorityBase } from "../model/morrisWorkflowDefaults";

interface WorkspacePanelProps {
  readonly active: PrimaryViewId;
  readonly model: ImpactAppModel;
  readonly onOpenGlossary: (term: string | undefined) => void;
  readonly morrisClient: MorrisAuthorityClient | null;
}

function SimulationWorkspace({ model }: { readonly model: ImpactAppModel }) {
  return (
    <SimulationPanel scenario={model.scenario} loftDeg={10.5}
      clubSpec={model.clubSpec}
      onScenarioChange={(updates) => model.setScenario((state) => ({ ...state, ...updates }))}
      spatialTarget={model.spatialTarget}
      onSpatialTargetChange={model.setSpatialTarget}
      distanceUnit={model.units.distance} />
  );
}

function ExplorerWorkspace(props: Omit<WorkspacePanelProps, "active" | "morrisClient">) {
  const { model } = props;
  return (
    <ImpactExplorerPanel scenario={model.scenario} setScenario={model.setScenario}
      units={model.units} setUnits={model.setUnits} clubSpec={model.clubSpec}
      setClubSpec={model.setClubSpec} generatedHead={model.generatedHead}
      setGeneratedHead={model.setGeneratedHead} explained={model.explained}
      onExplainedChange={model.setExplained} onOpenGlossary={props.onOpenGlossary} />
  );
}

export function PrimaryWorkspacePanel(props: WorkspacePanelProps) {
  const { active, model, onOpenGlossary } = props;
  switch (active) {
    case "glossary":
      return <GlossaryPanel key={model.glossaryTerm ?? "none"} initialTerm={model.glossaryTerm} />;
    case "putting":
      return <PuttingPanel distanceUnit={model.units.distance} onGlossary={onOpenGlossary} />;
    case "variation":
      try {
        return <VariationPanel spatialTarget={model.spatialTarget} distanceUnit={model.units.distance}
          morrisClient={props.morrisClient}
          morrisBase={defaultMorrisAuthorityBase(model.clubSpec, model.scenario)} />;
      } catch (error: unknown) {
        return <VariationPanel spatialTarget={model.spatialTarget} distanceUnit={model.units.distance}
          morrisClient={null}
          morrisUnavailableReason={error instanceof Error ? error.message : "Current model context is unsupported"} />;
      }
    case "flight":
      return <FlightExplorerPanel distanceUnit={model.units.distance}
        spatialTarget={model.spatialTarget} onSpatialTargetChange={model.setSpatialTarget} />;
    case "launch-monitor-analytics":
      return <LaunchMonitorAnalyticsPanel />;
    case "plots":
      return <PlotsPanel scenario={model.scenario} loftDeg={10.5} />;
    case "simulation":
      return <SimulationWorkspace model={model} />;
    case "calculation":
      return <Derivation scenario={model.scenario} />;
    default:
      return <ExplorerWorkspace model={model} onOpenGlossary={onOpenGlossary} />;
  }
}
