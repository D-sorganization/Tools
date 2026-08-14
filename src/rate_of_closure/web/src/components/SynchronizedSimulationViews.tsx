import type { RefObject } from "react";

import type { ClubSpec } from "../model/club";
import type { ImpactScenario } from "../model/impact";
import type { SimulationInput, SimulationRunTs } from "../model/simulation";
import type { SpatialTargetTs } from "../model/spatialTarget";
import type { ViewKind, ViewWorkspace } from "../model/viewWorkspace";
import {
  withCameraPreference,
  type CameraPreference,
} from "../model/cameraPreferences";
import { BallSetupDiagram } from "./BallSetupDiagram";
import { FlightPlayback3D } from "./FlightPlayback3D";
import { SpatialTargetSection } from "./SpatialTargetSection";
import { StrikeCanvas } from "./StrikeCanvas";
import { ViewCompositor } from "./ViewCompositor";

interface Props {
  readonly workspace: ViewWorkspace;
  readonly onWorkspaceChange: (workspace: ViewWorkspace) => void;
  readonly run: SimulationRunTs | null;
  readonly input: SimulationInput;
  readonly scenario: ImpactScenario;
  readonly effectiveLoftDeg: number;
  readonly clubSpec: ClubSpec | null;
  readonly spatialTarget: SpatialTargetTs;
  readonly onSpatialTargetChange: (target: SpatialTargetTs) => void;
  readonly timeS: number;
  readonly swingCanvasRef: RefObject<HTMLCanvasElement>;
  readonly deliveryAngles: { pathDeg: number; aoaDeg: number } | null;
}

function runIdentity(run: SimulationRunTs | null): string {
  if (run === null) return "no-run";
  return `${run.impactOutcome.status}:${run.totalDurationS.toFixed(6)}`;
}

function ImpactViewport(props: Props) {
  return (
    <div className="grid gap-3 xl:grid-cols-[1fr_220px]">
      <StrikeCanvas
        toeMm={props.scenario.impactOffsetToeMm}
        highMm={props.scenario.impactOffsetHighMm}
        loftDeg={props.effectiveLoftDeg}
        pathDeg={props.deliveryAngles?.pathDeg}
        aoaDeg={props.deliveryAngles?.aoaDeg}
        clubSpec={props.clubSpec}
      />
      <BallSetupDiagram setup={props.run?.ballSetup ?? props.input.ballSetup!} />
    </div>
  );
}

function SwingViewport(props: Props) {
  return (
    <canvas
      ref={props.swingCanvasRef}
      width={860}
      height={480}
      className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
      aria-label="Synchronized swing scene with task-specific overlays"
    />
  );
}

function FlightViewport(props: Props) {
  const impactTime = props.run?.impactTimeS ?? props.run?.impactOutcome.candidateTimeS ?? 0;
  return (
    <div className="space-y-3">
      <SpatialTargetSection target={props.spatialTarget}
        onChange={props.onSpatialTargetChange}
        flightPoints={props.run?.flight ?? []} />
      <FlightPlayback3D
        points={props.run?.flight ?? []}
        spatialTarget={props.spatialTarget}
        synchronizedTimeS={Math.max(0, props.timeS - impactTime)}
        hideTransport
        cameraPreference={props.workspace.cameraPreferences.viewports.flight}
        onCameraPreferenceChange={(preference: CameraPreference) => {
          const cameraPreferences = withCameraPreference(
            props.workspace.cameraPreferences,
            "flight",
            preference,
          );
          if (cameraPreferences !== props.workspace.cameraPreferences) {
            props.onWorkspaceChange({ ...props.workspace, cameraPreferences });
          }
        }}
      />
    </div>
  );
}

/** Bind three distinct visual hosts to one run and playback clock. */
export function SynchronizedSimulationViews(props: Props) {
  const renderViewport = (kind: ViewKind) => {
    if (kind === "impact") return <ImpactViewport {...props} />;
    if (kind === "flight") return <FlightViewport {...props} />;
    return <SwingViewport {...props} />;
  };
  return (
    <ViewCompositor
      workspace={props.workspace}
      onWorkspaceChange={props.onWorkspaceChange}
      renderViewport={renderViewport}
      runIdentity={runIdentity(props.run)}
      timeS={props.timeS}
    />
  );
}
