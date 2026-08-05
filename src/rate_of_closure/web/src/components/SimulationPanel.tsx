/**
 * Simulation tab for the web clone (epic #4103, practical parity).
 *
 * Mirrors the PyQt6 Simulation tab where practical: swing-source picker
 * (manual constant twist / double pendulum), plane-tilt inputs with
 * sourced hover guidance, impact-time scrubber (fixed ball, the swing
 * translates so the clubhead at tau meets it, delivery updating live),
 * launch-number readout, a canvas scene with ball/ground toggles + the
 * flight trajectory polyline, video-style playback (play/pause, timeline
 * scrub, rate presets incl. 1x real-time, loop), and JSON export as a
 * download. Screw-axis overlay is deliberately absent on web — it lands
 * with the WASM kernels in P7 (see model/simulation.ts).
 */

import { useEffect, useMemo, useRef, useState } from "react";

import {
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
import { FlightCanvases } from "./FlightCanvases";
import { TargetSection } from "./TargetSection";
import { KineticsSection } from "./KineticsSection";
import { SolverPanel } from "./SolverPanel";
import { StrikeCanvas } from "./StrikeCanvas";
import { drawSwingScene } from "./swingSceneDraw";

/** Scale-separated display views (epic #4120): face / swing / flight. */
const VIEWS = ["Strike", "Swing", "Kinetics", "Flight"] as const;
type ViewName = (typeof VIEWS)[number];

const RATE_PRESETS: Array<{ label: string; rate: number }> = [
  { label: "0.1×", rate: 0.1 },
  { label: "0.25×", rate: 0.25 },
  { label: "0.5×", rate: 0.5 },
  { label: "1× real-time", rate: 1.0 },
  { label: "2×", rate: 2.0 },
];

const LAUNCH_ROWS: Array<{ key: keyof SimulationRunTs["launch"]; label: string; unit: string }> = [
  { key: "ballSpeedMph", label: "Ball Speed", unit: "mph" },
  { key: "launchAngleDeg", label: "Launch Angle", unit: "°" },
  { key: "launchAzimuthDeg", label: "Launch Azimuth", unit: "°" },
  { key: "spinRpm", label: "Total Spin", unit: "rpm" },
  { key: "carryM", label: "Carry", unit: "m" },
  { key: "maxHeightM", label: "Apex", unit: "m" },
  { key: "flightTimeS", label: "Flight Time", unit: "s" },
  { key: "landingAngleDeg", label: "Landing Angle", unit: "°" },
];

interface Props {
  scenario: ImpactScenario;
  loftDeg: number;
  /** Effective club spec from the Club group (H1: CG marker source). */
  clubSpec?: ClubSpec | null;
  onScenarioChange: (updates: Partial<ImpactScenario>) => void;
  /** Target region (#4125 H7b), lifted to App for the Variation tie-in. */
  target: TargetRegionTs;
  onTargetChange: (target: TargetRegionTs) => void;
}

export function SimulationPanel({
  scenario,
  loftDeg,
  clubSpec = null,
  onScenarioChange,
  target,
  onTargetChange,
}: Props) {
  const [sourceKind, setSourceKind] = useState<WebSourceKind>("manual");
  const [tilts, setTilts] = useState({ yaw: 0, side: -45, forward: 0 });
  const [tauMs, setTauMs] = useState<number | null>(null);
  const [run, setRun] = useState<SimulationRunTs | null>(null);
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

  // Delivered path / attack angle at impact, for the strike view.
  const deliveryAngles = useMemo(() => {
    if (!run) return null;
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
      loftDeg,
      impactOffsetToeMm: scenario.impactOffsetToeMm,
      impactOffsetHighMm: scenario.impactOffsetHighMm,
      planeYawDeg: tilts.yaw,
      planeSideTiltDeg: tilts.side,
      planeForwardTiltDeg: tilts.forward,
      impactTimeS: tauMs === null ? null : tauMs / 1000.0,
      swingDurationS: 1.5,
    }),
    [sourceKind, scenario, loftDeg, tilts, tauMs],
  );

  const doRun = () => {
    const result = runSimulation(input);
    setRun(result);
    setTime(0);
    setPlaying(false);
  };

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
      format: "rate_of_closure.simulation_run.web/1",
      parameters: input,
      launch: run.launch,
      impactTimeS: run.impactTimeS,
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
  const numberInput = (
    label: string,
    value: number,
    guidanceKey: string,
    onChange: (value: number) => void,
  ) => (
    <label className="mb-2 block text-sm" title={FIELD_GUIDANCE[guidanceKey]}>
      <span className="mb-1 flex justify-between text-slate-300">
        <span>{label}</span>
        <span className="text-slate-500">deg</span>
      </span>
      <input
        type="number"
        inputMode="decimal"
        value={value}
        title={FIELD_GUIDANCE[guidanceKey]}
        onChange={(e) => {
          const parsed = Number(e.target.value);
          if (Number.isFinite(parsed)) onChange(parsed);
        }}
        className="no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
      />
    </label>
  );

  return (
    <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
      <section aria-label="Simulation setup" className="space-y-4">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Simulation Setup
          </h2>
          <label
            className="mb-3 block text-sm"
            title={FIELD_GUIDANCE.swingSource}
          >
            <span className="mb-1 block text-slate-300">Swing Source</span>
            <select
              value={sourceKind}
              title={FIELD_GUIDANCE.swingSource}
              onChange={(e) => setSourceKind(e.target.value as WebSourceKind)}
              className="w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
            >
              <option value="manual">Manual Scenario (Constant Twist)</option>
              <option value="double_pendulum">Double Pendulum</option>
            </select>
          </label>
          {numberInput("Plane Yaw", tilts.yaw, "planeYawDeg", (v) =>
            setTilts((t) => ({ ...t, yaw: v })),
          )}
          {numberInput("Plane Side Tilt", tilts.side, "planeSideTiltDeg", (v) =>
            setTilts((t) => ({ ...t, side: v })),
          )}
          {numberInput(
            "Plane Forward Tilt",
            tilts.forward,
            "planeForwardTiltDeg",
            (v) => setTilts((t) => ({ ...t, forward: v })),
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
              min={0}
              max={swingDuration * 1000}
              step={1}
              value={tauMs ?? (run ? run.impactTimeS * 1000 : swingDuration * 500)}
              title={FIELD_GUIDANCE.impactTimeScrub}
              onChange={(e) => setTauMs(Number(e.target.value))}
              onMouseUp={doRun}
              onTouchEnd={doRun}
              className="w-full"
            />
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

        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Launch Numbers
          </h2>
          <div className="grid gap-2">
            {LAUNCH_ROWS.map(({ key, label, unit }) => (
              <div
                key={key}
                className="flex items-center justify-between rounded-lg border border-slate-800/80 bg-slate-900/50 px-3 py-2 text-sm"
              >
                <span className="text-slate-400">{label}</span>
                <span className="font-semibold tabular-nums text-slate-100">
                  {run ? `${run.launch[key].toFixed(1)} ${unit}` : "—"}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Physics parity-pinned against the Python session (pendulum RK4,
            rigid-body COR impact, Waterloo/Penner flight). Gear effect,
            triple pendulum, and the screw-axis overlay arrive with the
            WASM kernels in P7.
          </p>
        </div>

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
              loftDeg={loftDeg}
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
              />
              <FlightCanvases
                points={run?.flight ?? []}
                emptyText="Run a simulation to populate the flight view."
                layout={targetLayout}
                target={target}
              />
            </>
          )}
          {view === "Swing" && (
            <>
          <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
            <button
              type="button"
              onClick={() => setPlaying((p) => !p && run !== null)}
              disabled={!run}
              title="Play or pause the swing playback"
              className="rounded border border-slate-700 bg-slate-800 px-3 py-1 text-slate-200 hover:border-slate-500 disabled:opacity-40"
            >
              {playing ? "Pause" : "Play"}
            </button>
            <button
              type="button"
              onClick={() => setTime((t) => Math.max(0, t - 0.001))}
              disabled={!run}
              title="Step the playback one millisecond backward"
              className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-300 disabled:opacity-40"
            >
              −1 frame
            </button>
            <button
              type="button"
              onClick={() =>
                setTime((t) => Math.min(run?.totalDurationS ?? 0, t + 0.001))
              }
              disabled={!run}
              title="Step the playback one millisecond forward"
              className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-300 disabled:opacity-40"
            >
              +1 frame
            </button>
            <input
              type="range"
              min={0}
              max={run?.totalDurationS ?? 1}
              step={0.001}
              value={time}
              onChange={(e) => setTime(Number(e.target.value))}
              disabled={!run}
              className="min-w-32 flex-1"
              aria-label="Playback timeline"
            />
            <span className="tabular-nums text-slate-400">
              {time.toFixed(3)} s
            </span>
            <label className="flex items-center gap-1 text-slate-300">
              <input
                type="checkbox"
                checked={loop}
                title="Restart the playback automatically when it reaches the end"
                onChange={(e) => setLoop(e.target.checked)}
              />
              Loop
            </label>
            <select
              value={rate}
              onChange={(e) => setRate(Number(e.target.value))}
              className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100"
              aria-label="Playback rate"
            >
              {RATE_PRESETS.map(({ label, rate: r }) => (
                <option key={label} value={r}>
                  {label}
                </option>
              ))}
            </select>
            <label
              className="flex items-center gap-1 text-slate-300"
              title={FIELD_GUIDANCE.ballVisible}
            >
              <input
                type="checkbox"
                checked={showBall}
                onChange={(e) => setShowBall(e.target.checked)}
              />
              Ball
            </label>
            <label
              className="flex items-center gap-1 text-slate-300"
              title={FIELD_GUIDANCE.groundVisible}
            >
              <input
                type="checkbox"
                checked={showGround}
                onChange={(e) => setShowGround(e.target.checked)}
              />
              Ground
            </label>
            <label
              className="flex items-center gap-1 text-slate-300"
              title={FIELD_GUIDANCE.courseVisible}
            >
              <input
                type="checkbox"
                checked={showCourse}
                onChange={(e) => setShowCourse(e.target.checked)}
              />
              Course Elements
            </label>
            <label
              className="flex items-center gap-1 text-amber-300/90"
              title={
                "Warning: expands the scene to flight scale, dwarfing " +
                "the swing. " + FIELD_GUIDANCE.swingFlightToggle
              }
            >
              <input
                type="checkbox"
                checked={showFlight}
                onChange={(e) => setShowFlight(e.target.checked)}
              />
              Show Ball Flight
            </label>
          </div>
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
