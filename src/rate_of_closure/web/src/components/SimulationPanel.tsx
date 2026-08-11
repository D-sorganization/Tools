/** Web swing-to-impact simulation with scale-separated playback and exports. */
import { useEffect, useMemo, useState } from "react";

import {
  DEFAULT_IMPACT_CLUB,
  runSimulation,
  type SimulationInput,
  type SimulationRunTs,
  type WebSourceKind,
} from "../model/simulation";
import { FIELD_GUIDANCE } from "../model/units";
import { type ClubSpec } from "../model/club";
import { solve, type ImpactScenario } from "../model/impact";
import { type SpatialTargetTs } from "../model/spatialTarget";
import { type ViewWorkspace } from "../model/viewWorkspace";
import { type ContactMode } from "../model/contact";
import { SpatialTargetSolverPanel } from "./SpatialTargetSolverPanel";
import { SimulationLaunchNumbers } from "./SimulationLaunchNumbers";
import { ContactPolicyControl } from "./ContactPolicyControl";
import { SimulationStatusHeader } from "./SimulationStatusHeader";
import { PlaneTiltControls } from "./PlaneTiltControls";
import { TorqueProfilePanel } from "./TorqueProfilePanel";
import { JointLockControls } from "./JointLockControls";
import { BallSetupControl } from "./BallSetupControl";
import { SimulationDisplay } from "./SimulationDisplay";
import {
  defaultBallSetupForClub,
  type BallSetup,
} from "../model/ballSetup";
import {
  loadBallSetupPreference,
  saveBallSetupPreference,
} from "../model/ballSetupPersistence";
import {
  PASSIVE_DOUBLE_PENDULUM_RUN,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
  type DoublePendulumRunConfig,
  type PendulumState,
} from "../model/doublePendulum";
import { SimulationRunFileControls } from "./SimulationRunFileControls";
import { ManualDeliveryControls } from "./ManualDeliveryControls";
import {
  DEFAULT_MANUAL_DELIVERY,
  type ManualDelivery,
} from "../model/manualDelivery";

interface Props {
  scenario: ImpactScenario;
  loftDeg: number;
  /** Effective club spec from the Club group (H1: CG marker source). */
  clubSpec?: ClubSpec | null;
  onScenarioChange: (updates: Partial<ImpactScenario>) => void;
  /** Canonical target shared with flight, solver, variation, and persistence. */
  spatialTarget: SpatialTargetTs;
  onSpatialTargetChange: (target: SpatialTargetTs) => void;
  /** Ball-flight distance display unit (#4125 H6): yards default. */
  distanceUnit?: string;
  viewWorkspace?: ViewWorkspace;
  viewCommandRevision?: number;
  onViewWorkspaceChange?: (workspace: ViewWorkspace) => void;
}

export function SimulationPanel({
  scenario,
  loftDeg,
  clubSpec = null,
  onScenarioChange,
  spatialTarget,
  onSpatialTargetChange,
  distanceUnit = "yd",
  viewWorkspace,
  viewCommandRevision,
  onViewWorkspaceChange,
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
  const [importError, setImportError] = useState<string | null>(null);
  const [sourceKind, setSourceKind] = useState<WebSourceKind>("manual");
  const [manualDelivery, setManualDelivery] = useState<ManualDelivery>(
    DEFAULT_MANUAL_DELIVERY,
  );
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
      omegaDps: solve(scenario).omegaDps,
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
      ...manualDelivery,
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
      manualDelivery,
    ],
  );
  const inputSignature = useMemo(() => JSON.stringify(input), [input]);

  const clubPhysicsGuidance = clubSpec
    ? `Impact physics uses ${clubSpec.name}: ${clubSpec.headMassKg.toFixed(3)} kg head mass, ${clubSpec.moiAboutShaftKgM2.toExponential(2)} kg m² MOI, and ${clubSpec.loftDeg.toFixed(1)}° nominal loft. COR uses the ${DEFAULT_IMPACT_CLUB.coefficientOfRestitution.toFixed(2)} driver default because the club library does not yet define measured COR.`
    : `No selected club specification was provided. Impact physics uses the default driver: ${DEFAULT_IMPACT_CLUB.headMassKg.toFixed(3)} kg head mass, ${DEFAULT_IMPACT_CLUB.moiAboutShaftKgM2.toExponential(2)} kg m² MOI, and ${DEFAULT_IMPACT_CLUB.coefficientOfRestitution.toFixed(2)} COR.`;

  const doRun = () => {
    try {
      const result = runSimulation(input);
      setRun(result);
      setLastRunSignature(inputSignature);
      setRunError(null);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    }
  };
  const runIsStale = run !== null && lastRunSignature !== inputSignature;
  const completedStatus = run?.impactOutcome.status === "miss"
    ? "Completed — no club–ball impact"
    : "Completed — impact and flight available";
  const completedDetails = run
    ? [
        run.torqueRun.mode === "prescribed"
          ? `prescribed torque profile ${run.torqueRun.profileId}`
          : null,
        run.torqueRun.lockedJointIds.includes(SHOULDER_JOINT_ID)
          ? "Shoulder locked (absolute ground frame)"
          : null,
        run.torqueRun.lockedJointIds.includes(WRIST_JOINT_ID)
          ? "Wrist locked (relative upper-segment frame)"
          : null,
      ].filter((detail): detail is string => detail !== null)
    : [];
  const runStatus = runError
    ? `Run failed: ${runError}`
    : runIsStale
      ? "Inputs changed — run required"
      : run ? [completedStatus, ...completedDetails].join("; ") : "Not run";

  // Populate the default Swing view immediately instead of presenting a
  // blank canvas that depends on discovering the Run button first.
  useEffect(() => {
    if (run === null) doRun();
    // Initial population only; explicit Run remains the calculation action.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
          {importError && (
            <p role="alert" className="mb-3 rounded-lg border border-rose-400/50 bg-rose-950/30 px-3 py-2 text-xs text-rose-200">
              {importError}
            </p>
          )}
          <p
            role="note"
            aria-label="Impact club physics"
            title={clubPhysicsGuidance}
            className="mb-3 rounded-lg border border-slate-700/80 bg-slate-950/50 px-3 py-2 text-xs leading-relaxed text-slate-400"
          >
            {clubPhysicsGuidance}
          </p>
          <PlaneTiltControls
            tilts={tilts}
            enabled={sourceKind !== "manual"}
            onChange={setTilts}
          />
          <ManualDeliveryControls
            enabled={sourceKind === "manual"}
            value={manualDelivery}
            onChange={setManualDelivery}
          />
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
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={doRun}
              title="Generate the swing, solve the impact at the scrubbed instant, and integrate the ball flight"
              className="min-w-32 flex-1 rounded-lg border border-sky-400/60 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-300 transition-all hover:bg-sky-500/20"
            >
              Run Simulation
            </button>
            <button
              type="button"
              onClick={() => {
                setTauMs(null);
                doRun();
              }}
              title="Reset the impact instant to the moment of maximum clubhead speed"
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300 hover:border-slate-500"
            >
              Auto τ
            </button>
            <SimulationRunFileControls input={input} run={run}
              prescribedTorqueProfile={doublePendulumRun.mode === "prescribed"
                ? doublePendulumRun.profile.toJsonObject() : null}
              spatialTarget={spatialTarget}
              onImported={({
                ballSetup: imported,
                spatialTarget: importedTarget,
                manualDelivery: importedManualDelivery,
              }) => {
                setImportError(null);
                setBallSetup(imported);
                setBallSetupOverridden(true);
                setManualDelivery(importedManualDelivery);
                onSpatialTargetChange(importedTarget);
                const warning = saveBallSetupPreference({
                  setup: imported,
                  userOverridden: true,
                });
                setBallSetupMessage(warning ??
                  `Imported ${imported.supportMode === "tee" ? "Tee" : "Ground"} ` +
                  "ball setup, spatial target, and manual delivery.");
              }}
              onImportError={(message) => {
                setBallSetupMessage(null);
                setImportError(`Cannot import simulation settings: ${message}`);
              }} />
          </div>
        </div>

        <TorqueProfilePanel
          sourceKind={sourceKind}
          runConfig={doublePendulumRun}
          onRunConfigChange={setDoublePendulumRun}
          run={run}
        />

        <SimulationLaunchNumbers run={run} distanceUnit={distanceUnit} />

        <SpatialTargetSolverPanel onApply={onScenarioChange}
          spatialTarget={spatialTarget} />
      </section>

      <SimulationDisplay
        run={run}
        input={input}
        scenario={scenario}
        effectiveLoftDeg={effectiveLoftDeg}
        clubSpec={clubSpec}
        spatialTarget={spatialTarget}
        onSpatialTargetChange={onSpatialTargetChange}
        distanceUnit={distanceUnit}
        viewWorkspace={viewWorkspace}
        viewCommandRevision={viewCommandRevision}
        onViewWorkspaceChange={onViewWorkspaceChange}
      />
    </div>
  );
}
