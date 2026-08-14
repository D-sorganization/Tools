/** Rate of Closure Impact Explorer — shareable web application shell. */

import { useMemo, useRef } from "react";

import { AppToolstrip } from "./components/AppToolstrip";
import { PrimaryViewTabs } from "./components/PrimaryViewTabs";
import { PrimaryWorkspacePanel } from "./components/PrimaryWorkspacePanel";
import { useAppWorkspace } from "./hooks/useAppWorkspace";
import { useImpactAppModel } from "./hooks/useImpactAppModel";
import { useRegionalGroundVariationWorkspace } from "./hooks/useRegionalGroundVariationWorkspace";
import { useRegionalGroundExecutionWorkspace } from "./hooks/useRegionalGroundExecutionWorkspace";
import { useWorkspaceFiles } from "./hooks/useWorkspaceFiles";
import { generatedHeadFor } from "./model/clubHeadGeneration";
import { HELP_TEXTS } from "./model/helptext";
import { primaryViewLabel, type PrimaryViewId } from "./model/viewPreferences";
import {
  LOCAL_COMPANION_WEB_RUNTIME,
  type WebRuntime,
} from "./model/webRuntime";

function AppHeader() {
  return (
    <header className="mb-4">
      <h1 className="bg-gradient-to-r from-sky-300 via-teal-200 to-emerald-300 bg-clip-text text-3xl font-bold tracking-tight text-transparent">
        Rate of Closure Impact Explorer
      </h1>
      <p className="mt-1 max-w-3xl text-sm text-slate-400">
        A rotating clubhead is a rigid body: the velocity of the impact point is
        v(P) = v(ref) + ω × r. Launch monitors track the reference point; the
        ball only feels the impact point. This explorer shows how far apart
        those two deliveries are.
      </p>
    </header>
  );
}

function ModuleHelp(props: {
  readonly active: PrimaryViewId;
  readonly open: boolean;
  readonly onOpenChange: (open: boolean) => void;
}) {
  const help = HELP_TEXTS[primaryViewLabel(props.active)];
  return (
    <details
      open={props.open}
      onToggle={(event) => props.onOpenChange(event.currentTarget.open)}
      className="mb-5 rounded-xl border border-slate-800/80 bg-slate-900/60 px-5 py-3 text-sm shadow-lg shadow-black/20 backdrop-blur"
      title="Usage instructions for this page"
    >
      <summary className="cursor-pointer font-semibold text-slate-300 hover:text-slate-100">
        {help.title}
      </summary>
      {help.paragraphs.map((paragraph, index) => (
        <p key={index} className="mt-2 max-w-3xl text-slate-400">
          {paragraph}
        </p>
      ))}
    </details>
  );
}

function AppFooter() {
  return (
    <footer className="mt-10 border-t border-slate-800/60 pt-4 text-xs text-slate-500">
      Companion tool to the{" "}
      <a
        href="https://www.affinedrift.com"
        target="_blank"
        rel="noreferrer"
        className="text-sky-400 underline-offset-2 hover:underline"
      >
        AffineDrift
      </a>{" "}
      launch-monitor research. Physics parity-tested against the canonical
      Python implementation; rate data from openly published sources (Cheetham
      2014; published launch-monitor material). MIT licensed.
    </footer>
  );
}

export interface AppProps {
  readonly runtime?: WebRuntime;
}

export default function App({ runtime = LOCAL_COMPANION_WEB_RUNTIME }: AppProps) {
  const workspace = useAppWorkspace();
  const model = useImpactAppModel();
  const regionalGroundVariation = useRegionalGroundVariationWorkspace();
  const preparationSource = useMemo(
    () =>
      model.flightPreparationLaunch === null
        ? undefined
        : {
            launch: model.flightPreparationLaunch,
            variationRequestPort: regionalGroundVariation.requestPort,
          },
    [model.flightPreparationLaunch, regionalGroundVariation.requestPort],
  );
  const regionalGroundExecution = useRegionalGroundExecutionWorkspace({
    authority: { runtime },
    preparationSource,
  });
  const snapshot = {
    scenario: model.scenario,
    club: model.clubSpec,
    units: model.units,
    simulation: {
      ballSetup: model.ballSetup,
      ballSetupUserOverridden: model.ballSetupUserOverridden,
      spatialTarget: model.spatialTarget,
    },
    torque: model.torqueWorkspace,
    variation: model.variationWorkspace,
    capability: model.capabilityWorkflow,
    modules: workspace.viewState,
    viewWorkspace: workspace.viewWorkspace,
  };
  const initialSnapshot = useRef(snapshot);
  const files = useWorkspaceFiles({
    snapshot,
    initialSnapshot: initialSnapshot.current,
    applySnapshot: (next) => {
      model.setScenario(next.scenario);
      model.setClubSpec(next.club);
      model.setGeneratedHead(generatedHeadFor(next.club));
      model.setUnits(next.units);
      model.setBallSetup(next.simulation.ballSetup);
      model.setBallSetupUserOverridden(next.simulation.ballSetupUserOverridden);
      model.setSpatialTarget(next.simulation.spatialTarget);
      model.setTorqueWorkspace(next.torque);
      model.setVariationWorkspace(next.variation);
      model.setCapabilityWorkflow(next.capability);
      workspace.setViewState(next.modules);
      workspace.setViewWorkspace(next.viewWorkspace);
    },
    applyViewWorkspace: workspace.setViewWorkspace,
  });
  const runCommand = (
    command: Parameters<typeof workspace.handleCommand>[0],
  ) => {
    if (!files.handleCommand(command)) workspace.handleCommand(command);
  };
  const active = workspace.viewState.active;
  const openGlossary = (term: string | undefined) => {
    model.setGlossaryTerm(term);
    workspace.activatePrimaryView("glossary");
  };
  return (
    <div
      data-app-theme={workspace.theme}
      className="mx-auto min-h-screen max-w-7xl p-5 text-slate-100 sm:p-8"
    >
      <AppToolstrip
        moduleState={workspace.viewState}
        theme={workspace.theme}
        shortcutHelpOpen={workspace.shortcutHelpOpen}
        onModuleStateChange={workspace.setViewState}
        onCommand={runCommand}
        regionalGroundVariationRequestPort={regionalGroundVariation.requestPort}
        regionalGroundExecutionWorkspace={regionalGroundExecution}
        onShortcutHelpOpenChange={workspace.setShortcutHelpOpen}
        fileStatus={files.dirty ? "Unsaved workspace changes" : files.status}
        fileError={files.error}
      />
      <input
        ref={files.fileInputRef}
        type="file"
        accept={files.fileAccept}
        aria-label="Workspace file chooser"
        hidden
        onChange={files.onFileChange}
      />
      <AppHeader />
      <PrimaryViewTabs
        state={workspace.viewState}
        onActiveChange={workspace.activatePrimaryView}
        onOrderChange={(order) =>
          workspace.setViewState((state) => ({ ...state, order }))
        }
      />
      <ModuleHelp
        active={active}
        open={workspace.moduleHelpOpen}
        onOpenChange={workspace.setModuleHelpOpen}
      />
      <main
        id={`primary-panel-${active}`}
        role="tabpanel"
        aria-labelledby={`primary-tab-${active}`}
      >
        <PrimaryWorkspacePanel
          active={active}
          model={model}
          regionalGroundVariation={regionalGroundVariation}
          regionalGroundExecution={regionalGroundExecution}
          onOpenGlossary={openGlossary}
          viewWorkspace={workspace.viewWorkspace}
          viewCommandRevision={workspace.viewCommandRevision}
          onViewWorkspaceChange={workspace.setViewWorkspace}
        />
      </main>
      <AppFooter />
    </div>
  );
}
