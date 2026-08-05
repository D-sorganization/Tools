import { useEffect, useMemo, useRef, useState } from "react";

import type { ClubSpec } from "../model/club";
import { DEFAULT_COURSE_LAYOUT, type CourseLayout } from "../model/course";
import type { ImpactScenario } from "../model/impact";
import type { SimulationInput, SimulationRunTs } from "../model/simulation";
import type { TargetRegionTs } from "../model/targets";
import { FIELD_GUIDANCE } from "../model/units";
import { BallSetupDiagram } from "./BallSetupDiagram";
import { FlightCanvases } from "./FlightCanvases";
import { KineticsSection } from "./KineticsSection";
import { StrikeCanvas } from "./StrikeCanvas";
import { SwingPlaybackControls } from "./SwingPlaybackControls";
import { TargetSection } from "./TargetSection";
import { drawSwingScene } from "./swingSceneDraw";

const VIEWS = ["Strike", "Swing", "Kinetics", "Flight"] as const;
type ViewName = (typeof VIEWS)[number];

const TOGGLE_GUIDANCE = {
  ball: FIELD_GUIDANCE.ballVisible,
  ground: FIELD_GUIDANCE.groundVisible,
  course: FIELD_GUIDANCE.courseVisible,
  flight: `Warning: expands the scene to flight scale, dwarfing the swing. ${FIELD_GUIDANCE.swingFlightToggle}`,
};

interface Props {
  run: SimulationRunTs | null;
  input: SimulationInput;
  scenario: ImpactScenario;
  effectiveLoftDeg: number;
  clubSpec: ClubSpec | null;
  target: TargetRegionTs;
  onTargetChange: (target: TargetRegionTs) => void;
  distanceUnit: string;
}

export function SimulationDisplay({
  run,
  input,
  scenario,
  effectiveLoftDeg,
  clubSpec,
  target,
  onTargetChange,
  distanceUnit,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [rate, setRate] = useState(1);
  const [time, setTime] = useState(0);
  const [showBall, setShowBall] = useState(true);
  const [showGround, setShowGround] = useState(true);
  const [showCourse, setShowCourse] = useState(true);
  const [showFlight, setShowFlight] = useState(false);
  const [view, setView] = useState<ViewName>("Swing");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const targetLayout = useMemo<CourseLayout>(() =>
    target.kind === "green"
      ? { ...DEFAULT_COURSE_LAYOUT, greenDistanceM: target.distanceM, greenRadiusM: target.radiusM }
      : {
          ...DEFAULT_COURSE_LAYOUT,
          greenDistanceM: target.distanceM + target.bandHalfLengthM,
          fairwayHalfWidthM: target.halfWidthM,
        }, [target]);
  const landing = useMemo(() => {
    if (!run || run.flight.length < 2) return undefined;
    const last = run.flight[run.flight.length - 1].position;
    return { carryM: last[0], lateralM: last[2] };
  }, [run]);
  const deliveryAngles = useMemo(() => {
    if (!run || run.impactTimeS === null || run.swing.length < 2) return null;
    const dt = run.swing[1].t - run.swing[0].t;
    const index = Math.min(run.swing.length - 1, Math.round(run.impactTimeS / dt));
    const velocity = run.swing[index].velocity;
    return {
      pathDeg: Math.atan2(velocity[2], velocity[0]) * 180 / Math.PI,
      aoaDeg: Math.atan2(velocity[1], Math.hypot(velocity[0], velocity[2])) * 180 / Math.PI,
    };
  }, [run]);

  useEffect(() => {
    setTime(0);
    setPlaying(false);
  }, [run]);
  useEffect(() => {
    if (!playing || !run) return undefined;
    let last = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const elapsed = (now - last) / 1000 * rate;
      last = now;
      setTime((current) => {
        const next = current + elapsed;
        if (next <= run.totalDurationS) return next;
        if (loop) return 0;
        setPlaying(false);
        return run.totalDurationS;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, run, rate, loop]);
  useEffect(() => {
    if (canvasRef.current) {
      drawSwingScene(canvasRef.current, run, {
        time, showBall, showGround, showCourse, showFlight,
      });
    }
  }, [run, time, showBall, showGround, showCourse, showFlight, view]);

  return (
    <section className="min-w-0 space-y-3">
      <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
        <div className="mb-3 flex gap-2" role="tablist" aria-label="Display views (scale-separated)">
          {VIEWS.map((name) => (
            <button key={name} type="button" role="tab" aria-selected={view === name}
              onClick={() => setView(name)}
              title={`Switch the display to the ${name}-scale view`}
              className={`rounded-full border px-4 py-1 text-sm font-medium transition-all ${view === name
                ? "border-sky-400/60 bg-sky-500/10 text-sky-300"
                : "border-slate-700/80 bg-slate-900/60 text-slate-300 hover:border-slate-500"}`}>
              {name}
            </button>
          ))}
        </div>
        {view === "Strike" && (
          <div className="grid gap-3 xl:grid-cols-[1fr_240px]">
            <StrikeCanvas toeMm={scenario.impactOffsetToeMm}
              highMm={scenario.impactOffsetHighMm} loftDeg={effectiveLoftDeg}
              pathDeg={deliveryAngles?.pathDeg} aoaDeg={deliveryAngles?.aoaDeg}
              clubSpec={clubSpec} />
            <BallSetupDiagram setup={run?.ballSetup ?? input.ballSetup!} />
          </div>
        )}
        {view === "Kinetics" && <KineticsSection input={input} run={run} />}
        {view === "Flight" && <>
          <TargetSection target={target} onChange={onTargetChange} landing={landing}
            unit={distanceUnit} />
          <FlightCanvases points={run?.flight ?? []}
            emptyText="Run a simulation to populate the flight view."
            layout={targetLayout} target={target} distanceUnit={distanceUnit} />
        </>}
        {view === "Swing" && <>
          <SwingPlaybackControls run={run} playing={playing} setPlaying={setPlaying}
            time={time} setTime={setTime} loop={loop} setLoop={setLoop}
            rate={rate} setRate={setRate} toggles={[
              ["Ball", showBall, setShowBall, TOGGLE_GUIDANCE.ball, "text-slate-300"],
              ["Ground", showGround, setShowGround, TOGGLE_GUIDANCE.ground, "text-slate-300"],
              ["Course Elements", showCourse, setShowCourse, TOGGLE_GUIDANCE.course, "text-slate-300"],
              ["Show Ball Flight", showFlight, setShowFlight, TOGGLE_GUIDANCE.flight, "text-amber-300/90"],
            ]} />
          <canvas ref={canvasRef} width={860} height={480}
            className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
            aria-label="Simulation scene (side view, swing scale)" />
        </>}
      </div>
    </section>
  );
}
