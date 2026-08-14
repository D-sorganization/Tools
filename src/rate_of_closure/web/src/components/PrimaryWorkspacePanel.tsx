import { lazy, Suspense } from "react";

import type { ImpactAppModel } from "../hooks/useImpactAppModel";
import type { RegionalGroundVariationWorkspaceController } from "../hooks/useRegionalGroundVariationWorkspace";
import type { RegionalGroundExecutionWorkspace } from "../hooks/useRegionalGroundExecutionWorkspace";
import type { PrimaryViewId } from "../model/viewPreferences";
import type { ViewWorkspace } from "../model/viewWorkspace";
import {
  withCameraPreference,
  type CameraPreference,
  type CameraViewportId,
} from "../model/cameraPreferences";
import { Derivation } from "./Derivation";
import { FlightExplorerPanel } from "./FlightExplorerPanel";
import { GlossaryPanel } from "./GlossaryPanel";
import { ImpactExplorerPanel } from "./ImpactExplorerPanel";
import { LaunchMonitorAnalyticsPanel } from "./LaunchMonitorAnalyticsPanel";
import { PlotsPanel } from "./PlotsPanel";
import { PuttingPanel } from "./PuttingPanel";
import { RegionalSurfacePlanPanel } from "./RegionalSurfacePlanPanel";
import { SimulationPanel } from "./SimulationPanel";
import { VariationPanel } from "./VariationPanel";

const LazyCapabilityOptimizationPanel = lazy(() =>
  import("./CapabilityOptimizationPanel").then((module) => ({
    default: module.CapabilityOptimizationPanel,
  })),
);
const LazyGroundPlaybackPanel = lazy(() =>
  import("./GroundPlaybackPanel").then((module) => ({
    default: module.GroundPlaybackPanel,
  })),
);

interface WorkspacePanelProps {
  readonly active: PrimaryViewId;
  readonly model: ImpactAppModel;
  readonly regionalGroundVariation: RegionalGroundVariationWorkspaceController;
  readonly regionalGroundExecution: RegionalGroundExecutionWorkspace;
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
  props: Pick<
    WorkspacePanelProps,
    "model" | "onOpenGlossary" | "viewWorkspace" | "onViewWorkspaceChange"
  >,
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
      cameraPreference={props.viewWorkspace.cameraPreferences.viewports.swing}
      onCameraPreferenceChange={(preference) => {
        const cameraPreferences = withCameraPreference(
          props.viewWorkspace.cameraPreferences,
          "swing",
          preference,
        );
        if (cameraPreferences !== props.viewWorkspace.cameraPreferences) {
          props.onViewWorkspaceChange({ ...props.viewWorkspace, cameraPreferences });
        }
      }}
    />
  );
}

export function PrimaryWorkspacePanel(props: WorkspacePanelProps) {
  const { active, model, onOpenGlossary, regionalGroundVariation } = props;
  const updateCameraPreference = (
    viewportId: CameraViewportId,
    preference: CameraPreference,
  ) => {
    const cameraPreferences = withCameraPreference(
      props.viewWorkspace.cameraPreferences,
      viewportId,
      preference,
    );
    if (cameraPreferences !== props.viewWorkspace.cameraPreferences) {
      props.onViewWorkspaceChange({ ...props.viewWorkspace, cameraPreferences });
    }
  };
  // The regional-ground controller owns the live variation plan because the
  // execution request port reads it; mirror every change into the app-level
  // workspace snapshot so saved workspaces record the real study (#4144).
  const variationWorkspace = {
    ...model.variationWorkspace,
    plan: regionalGroundVariation.state.variationPlan,
    analysisExecution: regionalGroundVariation.state.analysisExecution,
  };
  const applyVariationWorkspace = (
    update:
      | typeof variationWorkspace
      | ((current: typeof variationWorkspace) => typeof variationWorkspace),
  ) => {
    const next =
      typeof update === "function" ? update(variationWorkspace) : update;
    regionalGroundVariation.replaceVariationPlan(next.plan);
    regionalGroundVariation.replaceAnalysisExecution(next.analysisExecution);
    model.setVariationWorkspace(next);
  };
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
          variationWorkspace={variationWorkspace}
          onVariationWorkspaceChange={applyVariationWorkspace}
        />
      );
    case "flight":
      return (
        <FlightExplorerPanel
          distanceUnit={model.units.distance}
          spatialTarget={model.spatialTarget}
          onSpatialTargetChange={model.setSpatialTarget}
          draft={model.flightExplorerDraft}
          onDraftChange={model.setFlightExplorerDraft}
          cameraPreference={props.viewWorkspace.cameraPreferences.viewports.flight}
          onCameraPreferenceChange={(preference) =>
            updateCameraPreference("flight", preference)
          }
        />
      );
    case "regional-surfaces":
      return (
        <RegionalSurfacePlanPanel
          draft={regionalGroundVariation.state.regionalDraft}
          importedRequest={regionalGroundVariation.state.importedRegionalRequest}
          onDraftChange={regionalGroundVariation.replaceRegionalDraft}
          onImport={regionalGroundVariation.applyRegionalImport}
        />
      );
    case "ground-playback":
      return (
        <Suspense
          fallback={
            <section role="status" aria-label="Ground playback loading">
              Loading…
            </section>
          }
        >
          <LazyGroundPlaybackPanel
            regionalGroundExecution={props.regionalGroundExecution}
          />
        </Suspense>
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
          <LazyCapabilityOptimizationPanel
            workflow={model.capabilityWorkflow}
            onWorkflowChange={model.setCapabilityWorkflow}
          />
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
        <ExplorerWorkspace model={model} onOpenGlossary={onOpenGlossary}
          viewWorkspace={props.viewWorkspace}
          onViewWorkspaceChange={props.onViewWorkspaceChange} />
      );
  }
}
