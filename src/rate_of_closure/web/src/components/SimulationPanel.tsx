/** Web swing-to-impact simulation with scale-separated playback and exports. */
import { useEffect, useMemo, useRef, useState } from "react";

import {
  DEFAULT_IMPACT_CLUB,
  runSimulation,
  type SimulationInput,
  type SimulationRunTs,
  type WebSourceKind,
} from "../model/simulation";
import { FIELD_GUIDANCE } from "../model/units";
import { type ClubSpec } from "../model/club";
import { type ImpactScenario } from "../model/impact";
import {
  DEFAULT_COURSE_LAYOUT,
  type CourseLayout,
} from "../model/course";
import { type TargetRegionTs } from "../model/targets";
import { type ContactMode } from "../model/contact";
import { FlightCanvases } from "./FlightCanvases";
import { TargetSection } from "./TargetSection";
import { KineticsSection } from "./KineticsSection";
import { SolverPanel } from "./SolverPanel";
import { StrikeCanvas } from "./StrikeCanvas";
import { drawSwingScene } from "./swingSceneDraw";
import { SimulationLaunchNumbers } from "./SimulationLaunchNumbers";
import { SwingPlaybackControls } from "./SwingPlaybackControls";
import { ContactPolicyControl } from "./ContactPolicyControl";
import { SimulationStatusHeader } from "./SimulationStatusHeader";
import { PlaneTiltControls } from "./PlaneTiltControls";
import { TorqueProfilePanel } from "./TorqueProfilePanel";
import { JointLockControls } from "./JointLockControls";
import {
  PASSIVE_DOUBLE_PENDULUM_RUN,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
  type DoublePendulumRunConfig,
  type PendulumState,
} from "../model/doublePendulum";

const SWING_TOGGLE_GUIDANCE = {
  ball: FIELD_GUIDANCE.ballVisible,
  ground: FIELD_GUIDANCE.groundVisible,
  course: FIELD_GUIDANCE.courseVisible,
  flight: `Warning: expands the scene to flight scale, dwarfing the swing. ${FIELD_GUIDANCE.swingFlightToggle}`,
};

const VIEWS = ["Strike", "Swing", "Kinetics", "Flight"] as const;
type ViewName = (typeof VIEWS)[number];
interface Props {
  scenario: ImpactScenario;
  loftDeg: number;
  /** Effective club spec from the Club group (H1: CG marker source). */
  clubSpec?: ClubSpec | null;
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
  onScenarioChange,
  target,
  onTargetChange,
  distanceUnit = "yd",
}: Props) {
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
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [rate, setRate] = useState(1.0);
  const [time, setTime] = useState(0);
  const [showBall, setShowBall] = useState(true);
  const [showGround, setShowGround] = useState(true);
  // Course scene (#4125 H7a): fairway strip, green + flag, tee marker.
  const [showCourse, setShowCourse] = useState(true);
  // Target region (#4125 H7b): drives the course green + solver goal.
  const targetLayout = useMemo<CourseLayout>(
    () =>
      target.kind === "green"
        ? {
            ...DEFAULT_COURSE_LAYOUT,
            greenDistanceM: target.distanceM,
            greenRadiusM: target.radiusM,
          }
        : {
            ...DEFAULT_COURSE_LAYOUT,
            greenDistanceM: target.distanceM + target.bandHalfLengthM,
            fairwayHalfWidthM: target.halfWidthM,
          },
    [target],
  );
  // Latest landing point (carry, + right lateral) for containment stats.
  const landing = useMemo(() => {
    const flight = run?.flight ?? [];
    if (flight.length < 2) return null;
    const last = flight[flight.length - 1].position;
    return { carryM: last[0], lateralM: last[2] };
  }, [run]);
  // Scale separation (#4120): flight display in the swing view is
  // opt-in — its envelope dwarfs the swing envelope.
  const [showFlight, setShowFlight] = useState(false);
  const [view, setView] = useState<ViewName>("Swing");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const effectiveLoftDeg = clubSpec?.loftDeg ?? loftDeg;

  // Delivered path / attack angle at impact, for the strike view.
  const deliveryAngles = useMemo(() => {
    if (!run || run.impactTimeS === null || run.swing.length < 2) return null;
    const dt = run.swing[1].t - run.swing[0].t;
    const index = Math.min(
      run.swing.length - 1,
      Math.round(run.impactTimeS / dt),
    );
    const v = run.swing[index].velocity;
    const degOf = (r: number) => (r * 180.0) / Math.PI;
    return {
      pathDeg: degOf(Math.atan2(v[2], v[0])),
      aoaDeg: degOf(Math.atan2(v[1], Math.hypot(v[0], v[2]))),
    };
  }, [run]);

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
      setTime(0);
      setPlaying(false);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
      setPlaying(false);
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

  // Playback clock: advance simulated time at the selected rate.
  useEffect(() => {
    if (!playing || !run) return undefined;
    let last = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const dt = ((now - last) / 1000.0) * rate;
      last = now;
      setTime((t) => {
        const next = t + dt;
        if (next > run.totalDurationS) {
          if (loop) return 0;
          setPlaying(false);
          return run.totalDurationS;
        }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, run, rate, loop]);

  // Scene drawing: swing-scale renderer (see swingSceneDraw.ts).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawSwingScene(canvas, run, {
      time,
      showBall,
      showGround,
      showCourse,
      showFlight,
    });
  }, [run, time, showBall, showGround, showCourse, showFlight, view]);

  const exportJson = () => {
    if (!run) return;
    const doc = {
      format: "rate_of_closure.simulation_run.web/2",
      parameters: input,
      impactOutcome: run.impactOutcome,
      launch: run.launch,
      impactTimeS: run.impactTimeS,
      torqueRun: run.torqueRun,
      prescribedTorqueProfile: doublePendulumRun.mode === "prescribed"
        ? doublePendulumRun.profile.toJsonObject()
        : null,
      series: {
        swing: run.swing,
        flight: run.flight,
      },
    };
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
          <p
            role="note"
            aria-label="Impact club physics"
            title={clubPhysicsGuidance}
            className="mb-3 rounded-lg border border-slate-700/80 bg-slate-950/50 px-3 py-2 text-xs leading-relaxed text-slate-400"
          >
            {clubPhysicsGuidance}
          </p>
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
                setTauMs(null);
                doRun();
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

      <section className="min-w-0 space-y-3">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
          <div
            className="mb-3 flex gap-2"
            role="tablist"
            aria-label="Display views (scale-separated)"
          >
            {VIEWS.map((name) => (
              <button
                key={name}
                type="button"
                role="tab"
                aria-selected={view === name}
                onClick={() => setView(name)}
                title={`Switch the display to the ${name}-scale view`}
                className={
                  "rounded-full border px-4 py-1 text-sm font-medium transition-all " +
                  (view === name
                    ? "border-sky-400/60 bg-sky-500/10 text-sky-300"
                    : "border-slate-700/80 bg-slate-900/60 text-slate-300 hover:border-slate-500")
                }
              >
                {name}
              </button>
            ))}
          </div>
          {view === "Strike" && (
            <StrikeCanvas
              toeMm={scenario.impactOffsetToeMm}
              highMm={scenario.impactOffsetHighMm}
              loftDeg={effectiveLoftDeg}
              pathDeg={deliveryAngles?.pathDeg}
              aoaDeg={deliveryAngles?.aoaDeg}
              clubSpec={clubSpec}
            />
          )}
          {view === "Kinetics" && <KineticsSection input={input} run={run} />}
          {view === "Flight" && (
            <>
              <TargetSection
                target={target}
                onChange={onTargetChange}
                landing={landing ?? undefined}
                unit={distanceUnit}
              />
              <FlightCanvases
                points={run?.flight ?? []}
                emptyText="Run a simulation to populate the flight view."
                layout={targetLayout}
                target={target}
                distanceUnit={distanceUnit}
              />
            </>
          )}
          {view === "Swing" && (
            <>
          <SwingPlaybackControls
            run={run}
            playing={playing}
            setPlaying={setPlaying}
            time={time}
            setTime={setTime}
            loop={loop}
            setLoop={setLoop}
            rate={rate}
            setRate={setRate}
            toggles={[
              ["Ball", showBall, setShowBall, SWING_TOGGLE_GUIDANCE.ball, "text-slate-300"],
              ["Ground", showGround, setShowGround, SWING_TOGGLE_GUIDANCE.ground, "text-slate-300"],
              ["Course Elements", showCourse, setShowCourse, SWING_TOGGLE_GUIDANCE.course, "text-slate-300"],
              ["Show Ball Flight", showFlight, setShowFlight, SWING_TOGGLE_GUIDANCE.flight, "text-amber-300/90"],
            ]}
          />
          <canvas
            ref={canvasRef}
            width={860}
            height={480}
            className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
            aria-label="Simulation scene (side view, swing scale)"
          />
            </>
          )}
        </div>
      </section>
    </div>
  );
}
