/** Web swing-to-impact simulation with scale-separated playback and exports. */
import { useEffect, useMemo, useState } from "react";

import {
  runSimulation,
  type SimulationInput,
  type SimulationRunTs,
  type WebSourceKind,
} from "../model/simulation";
import { FIELD_GUIDANCE } from "../model/units";
import { type ClubSpec } from "../model/club";
import { type ClubAssemblyBinding } from "../model/clubAssemblyBinding";
import { type ImpactScenario } from "../model/impact";
import { type TargetRegionTs } from "../model/targets";
import { type ContactMode } from "../model/contact";
import { SolverPanel } from "./SolverPanel";
import { SimulationLaunchNumbers } from "./SimulationLaunchNumbers";
import { ContactPolicyControl } from "./ContactPolicyControl";
import { SimulationStatusHeader } from "./SimulationStatusHeader";
import { PlaneTiltControls } from "./PlaneTiltControls";
import { TorqueProfilePanel } from "./TorqueProfilePanel";
import { JointLockControls } from "./JointLockControls";
import { BallSetupControl } from "./BallSetupControl";
import { SimulationDisplay } from "./SimulationDisplay";
import { SimulationPhysicsStatus } from "./SimulationPhysicsStatus";
import { simulationRunStatus } from "../model/simulationRunStatus";
import { readBrowserFileText } from "../model/browserFileText";
import {
  defaultBallSetupForClub,
  type BallSetup,
} from "../model/ballSetup";
import {
  ballSetupFromSimulationDocument,
  createSimulationRunDocument,
  loadBallSetupPreference,
  saveBallSetupPreference,
} from "../model/ballSetupPersistence";
import {
  PASSIVE_DOUBLE_PENDULUM_RUN,
  type DoublePendulumRunConfig,
  type PendulumState,
} from "../model/doublePendulum";

interface Props {
  scenario: ImpactScenario;
  loftDeg: number;
  /** Effective club spec from the Club group (H1: CG marker source). */
  clubSpec?: ClubSpec | null;
  /** Exact selected-spec binding imported by the Club panel. */
  assemblyBinding?: ClubAssemblyBinding;
  onScenarioChange: (updates: Partial<ImpactScenario>) => void;
  /** Target region (#4125 H7b), lifted to App for the Variation tie-in. */
  target: TargetRegionTs;
  onTargetChange: (target: TargetRegionTs) => void;
  /** Ball-flight distance display unit (#4125 H6): yards default. */
  distanceUnit?: string;
}

export function SimulationPanel({
  scenario,
  loftDeg,
  clubSpec = null,
  assemblyBinding,
  onScenarioChange,
  target,
  onTargetChange,
  distanceUnit = "yd",
}: Props) {
  const clubDefaultSetup = defaultBallSetupForClub(clubSpec);
  const [initialBallPreference] = useState(() => {
    const loaded = loadBallSetupPreference(undefined, clubDefaultSetup);
    return !loaded.userOverridden && loaded.warning === null
      ? { ...loaded, setup: clubDefaultSetup }
      : loaded;
  });
  const [ballSetup, setBallSetup] = useState<BallSetup>(initialBallPreference.setup);
  const [ballSetupOverridden, setBallSetupOverridden] = useState(
    initialBallPreference.userOverridden,
  );
  const [ballSetupMessage, setBallSetupMessage] = useState<string | null>(
    initialBallPreference.warning,
  );
  const [sourceKind, setSourceKind] = useState<WebSourceKind>("manual");
  const [contactMode, setContactMode] =
    useState<ContactMode>("delivery_inspection");
  const [doublePendulumRun, setDoublePendulumRun] =
    useState<DoublePendulumRunConfig>(PASSIVE_DOUBLE_PENDULUM_RUN);
  const [doublePendulumInitialState, setDoublePendulumInitialState] =
    useState<PendulumState>([-Math.PI / 2, 0, 0, 0]);
  const [tilts, setTilts] = useState({ yaw: 0, side: -45, forward: 0 });
  const [tauMs, setTauMs] = useState<number | null>(null);
  const [run, setRun] = useState<SimulationRunTs | null>(null);
  const [lastRunSignature, setLastRunSignature] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const effectiveLoftDeg = clubSpec?.loftDeg ?? loftDeg;

  useEffect(() => {
    if (ballSetupOverridden) return;
    const next = defaultBallSetupForClub(clubSpec);
    setBallSetup(next);
    const warning = saveBallSetupPreference({ setup: next, userOverridden: false });
    if (warning) setBallSetupMessage(warning);
  }, [clubSpec, ballSetupOverridden]);

  const input: SimulationInput = useMemo(
    () => ({
      sourceKind,
      clubheadSpeedMph: scenario.clubheadSpeedMph,
      omegaDps: [0, 0, 0],
      loftDeg: effectiveLoftDeg,
      impactOffsetToeMm: scenario.impactOffsetToeMm,
      impactOffsetHighMm: scenario.impactOffsetHighMm,
      planeYawDeg: tilts.yaw,
      planeSideTiltDeg: tilts.side,
      planeForwardTiltDeg: tilts.forward,
      impactTimeS: tauMs === null ? null : tauMs / 1000.0,
      swingDurationS: 1.5,
      club: clubSpec ?? undefined,
      contactMode,
      doublePendulumRun,
      doublePendulumInitialState,
      ballSetup,
      assemblyBinding,
      assemblyClubSpec: assemblyBinding ? (clubSpec ?? undefined) : undefined,
    }),
    [
      sourceKind,
      scenario,
      effectiveLoftDeg,
      tilts,
      tauMs,
      clubSpec,
      contactMode,
      doublePendulumRun,
      doublePendulumInitialState,
      ballSetup,
      assemblyBinding,
    ],
  );
  const inputSignature = useMemo(() => JSON.stringify(input), [input]);

  const runWithInput = (simulationInput: SimulationInput) => {
    try {
      const result = runSimulation(simulationInput);
      setRun(result);
      setLastRunSignature(JSON.stringify(simulationInput));
      setRunError(null);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    }
  };
  const doRun = () => runWithInput(input);
  const runIsStale = run !== null && lastRunSignature !== inputSignature;
  const runStatus = simulationRunStatus(run, runError, runIsStale);

  // Populate the default Swing view immediately instead of presenting a
  // blank canvas that depends on discovering the Run button first.
  useEffect(() => {
    if (run === null) doRun();
    // Initial population only; explicit Run remains the calculation action.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const exportJson = () => {
    if (!run) return;
    const doc = createSimulationRunDocument(
      input,
      run,
      doublePendulumRun.mode === "prescribed"
        ? doublePendulumRun.profile.toJsonObject()
        : null,
    );
    const blob = new Blob([JSON.stringify(doc, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "simulation_run.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const importJson = async (file?: File) => {
    if (!file) return;
    try {
      const imported = ballSetupFromSimulationDocument(
        JSON.parse(await readBrowserFileText(file)),
      );
      setBallSetup(imported);
      setBallSetupOverridden(true);
      const warning = saveBallSetupPreference({ setup: imported, userOverridden: true });
      setBallSetupMessage(
        warning ?? `Imported ${imported.supportMode === "tee" ? "Tee" : "Ground"} ball setup.`,
      );
    } catch (error) {
      setBallSetupMessage(`Cannot import ball setup: ${(error as Error).message}`);
    }
  };

  const swingDuration = run ? run.swing[run.swing.length - 1].t : 1.5;
  return (
    <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
      <SimulationStatusHeader
        sourceKind={sourceKind}
        onSourceKindChange={(next) => {
          setSourceKind(next);
          if (next !== "double_pendulum") {
            setDoublePendulumRun(PASSIVE_DOUBLE_PENDULUM_RUN);
          }
        }}
        status={runStatus}
        warning={
          Boolean(runError) ||
          runIsStale ||
          run?.impactOutcome.status === "miss"
        }
      />
      <section aria-label="Simulation setup" className="space-y-4">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Simulation Setup
          </h2>
          <ContactPolicyControl
            value={contactMode}
            onChange={(mode) => {
              setContactMode(mode);
              if (mode === "fixed_ball_contact") setTauMs(null);
            }}
          />
          <BallSetupControl
            setup={ballSetup}
            userOverridden={ballSetupOverridden}
            onChange={(next) => {
              setBallSetup(next);
              setBallSetupOverridden(true);
              setBallSetupMessage(
                saveBallSetupPreference({ setup: next, userOverridden: true }),
              );
            }}
            onUseClubDefault={() => {
              const next = defaultBallSetupForClub(clubSpec);
              setBallSetup(next);
              setBallSetupOverridden(false);
              setBallSetupMessage(
                saveBallSetupPreference({ setup: next, userOverridden: false }),
              );
            }}
          />
          {ballSetupMessage && (
            <p role="status" className="mb-3 text-xs text-sky-300">{ballSetupMessage}</p>
          )}
          <SimulationPhysicsStatus
            clubSpec={clubSpec}
            assemblyBinding={assemblyBinding}
            run={run}
            runIsStale={runIsStale}
          />
          <PlaneTiltControls tilts={tilts} onChange={setTilts} />
          {sourceKind === "double_pendulum" && (
            <JointLockControls
              initialState={doublePendulumInitialState}
              runConfig={doublePendulumRun}
              onInitialStateChange={setDoublePendulumInitialState}
              onRunConfigChange={setDoublePendulumRun}
            />
          )}
          <label
            className="mb-3 block text-sm"
            title={FIELD_GUIDANCE.impactTimeScrub}
          >
            <span className="mb-1 flex justify-between text-slate-300">
              <span>Impact Time τ</span>
              <span className="text-slate-500">
                {tauMs === null ? "auto" : `${tauMs.toFixed(0)} ms`}
              </span>
            </span>
            <input
              type="range"
              aria-label="Impact Time"
              min={0}
              max={swingDuration * 1000}
              step={1}
              value={
                tauMs ??
                (run
                  ? (run.impactTimeS ?? run.impactOutcome.candidateTimeS) * 1000
                  : swingDuration * 500)
              }
              disabled={contactMode === "fixed_ball_contact"}
              title={FIELD_GUIDANCE.impactTimeScrub}
              onChange={(e) => setTauMs(Number(e.target.value))}
              onMouseUp={doRun}
              onTouchEnd={doRun}
              className="w-full disabled:cursor-not-allowed disabled:opacity-40"
            />
            {contactMode === "fixed_ball_contact" && (
              <span className="mt-1 block text-xs text-amber-300/90">
                Impact time is detected from closest approach in this mode.
              </span>
            )}
          </label>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={doRun}
              title="Generate the swing, solve the impact at the scrubbed instant, and integrate the ball flight"
              className="flex-1 rounded-lg border border-sky-400/60 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-300 transition-all hover:bg-sky-500/20"
            >
              Run Simulation
            </button>
            <button
              type="button"
              onClick={() => {
                const automaticInput = { ...input, impactTimeS: null };
                setTauMs(null);
                runWithInput(automaticInput);
              }}
              title="Reset the impact instant to the moment of maximum clubhead speed"
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300 hover:border-slate-500"
            >
              Auto τ
            </button>
            <button
              type="button"
              onClick={exportJson}
              disabled={!run}
              title="Download the full run (delivery, launch numbers, trajectory) as JSON"
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300 hover:border-slate-500 disabled:opacity-40"
            >
              Export JSON
            </button>
            <label className="cursor-pointer rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300 hover:border-slate-500">
              Import JSON
              <input
                type="file"
                accept="application/json,.json"
                aria-label="Import Simulation JSON"
                className="sr-only"
                onChange={(event) => {
                  void importJson(event.target.files?.[0]);
                  event.currentTarget.value = "";
                }}
              />
            </label>
          </div>
        </div>

        <TorqueProfilePanel
          sourceKind={sourceKind}
          runConfig={doublePendulumRun}
          onRunConfigChange={setDoublePendulumRun}
          run={run}
        />

        <SimulationLaunchNumbers run={run} distanceUnit={distanceUnit} />

        <SolverPanel onApply={onScenarioChange} target={target} />
      </section>

      <SimulationDisplay
        run={run}
        input={input}
        scenario={scenario}
        effectiveLoftDeg={effectiveLoftDeg}
        clubSpec={clubSpec}
        target={target}
        onTargetChange={onTargetChange}
        distanceUnit={distanceUnit}
      />
    </div>
  );
}
