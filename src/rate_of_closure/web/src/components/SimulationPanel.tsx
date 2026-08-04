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
  BALL_POSITION,
  runSimulation,
  type SimulationInput,
  type SimulationRunTs,
  type WebSourceKind,
} from "../model/simulation";
import { FIELD_GUIDANCE } from "../model/units";
import { type ImpactScenario } from "../model/impact";

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
}

export function SimulationPanel({ scenario, loftDeg }: Props) {
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
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

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

  // Scene drawing: side-on orthographic projection (x right, y up).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    if (!run) {
      ctx.fillStyle = "#64748b";
      ctx.font = "14px sans-serif";
      ctx.fillText("Run a simulation to populate the scene.", 16, 28);
      return;
    }

    const swingEnd = run.swing[run.swing.length - 1].t;
    const inFlight = time > run.impactTimeS;
    // Extent: swing envelope near impact, flight envelope once airborne.
    const extentX = inFlight
      ? Math.max(10, ...run.flight.map((p) => Math.abs(p.position[0]))) * 1.05
      : Math.max(1.5, ...run.swing.map((p) => Math.abs(p.position[0]))) * 1.15;
    const extentY = inFlight
      ? Math.max(5, ...run.flight.map((p) => p.position[1])) * 1.3
      : Math.max(1.5, ...run.swing.map((p) => Math.abs(p.position[1]))) * 1.15;
    const originX = inFlight ? 30 : width / 2;
    const scaleX = (width - 60) / (inFlight ? extentX : 2 * extentX);
    const scaleY = (height - 40) / (inFlight ? extentY : 2 * extentY);
    const s = Math.min(scaleX, scaleY);
    const groundY = inFlight ? height - 24 : height / 2 + extentY * s * 0.5;
    const px = (x: number) => originX + x * s;
    const py = (y: number) => groundY - y * s;

    if (showGround) {
      ctx.strokeStyle = "#475569";
      ctx.beginPath();
      ctx.moveTo(0, py(0));
      ctx.lineTo(width, py(0));
      ctx.stroke();
    }
    if (showBall) {
      ctx.fillStyle = "#facc15";
      ctx.beginPath();
      ctx.arc(px(BALL_POSITION[0]), py(BALL_POSITION[1]), 4, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Swing path (faint full arc + traversed portion + head marker).
    const drawPath = (
      points: Array<{ position: [number, number, number] }>,
      color: string,
      widthPx: number,
    ) => {
      if (points.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = widthPx;
      ctx.beginPath();
      points.forEach((point, index) => {
        const cx = px(point.position[0]);
        const cy = py(point.position[1]);
        if (index === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.stroke();
      ctx.lineWidth = 1;
    };
    drawPath(run.swing, "rgba(56,189,248,0.25)", 1);
    const swingIndex = Math.min(
      run.swing.length - 1,
      Math.round((Math.min(time, swingEnd) / swingEnd) * (run.swing.length - 1)),
    );
    drawPath(run.swing.slice(0, swingIndex + 1), "#38bdf8", 2);
    const head = run.swing[swingIndex].position;
    ctx.fillStyle = "#f472b6";
    ctx.beginPath();
    ctx.arc(px(head[0]), py(head[1]), 4, 0, 2 * Math.PI);
    ctx.fill();

    // Flight trajectory polyline (faint + traversed).
    drawPath(run.flight, "rgba(52,211,153,0.25)", 1);
    if (inFlight) {
      const flightT = time - run.impactTimeS;
      const upto = run.flight.filter((p) => p.time <= flightT);
      drawPath(upto, "#34d399", 2);
      if (upto.length) {
        const ball = upto[upto.length - 1].position;
        ctx.fillStyle = "#facc15";
        ctx.beginPath();
        ctx.arc(px(ball[0]), py(ball[1]), 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px sans-serif";
    ctx.fillText(
      `t = ${time.toFixed(3)} s (${inFlight ? "flight" : "swing"}) — impact at ${run.impactTimeS.toFixed(3)} s`,
      12,
      16,
    );
  }, [run, time, showBall, showGround]);

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
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300 hover:border-slate-500"
            >
              Auto τ
            </button>
            <button
              type="button"
              onClick={exportJson}
              disabled={!run}
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
      </section>

      <section className="space-y-3">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
            <button
              type="button"
              onClick={() => setPlaying((p) => !p && run !== null)}
              disabled={!run}
              className="rounded border border-slate-700 bg-slate-800 px-3 py-1 text-slate-200 hover:border-slate-500 disabled:opacity-40"
            >
              {playing ? "Pause" : "Play"}
            </button>
            <button
              type="button"
              onClick={() => setTime((t) => Math.max(0, t - 0.001))}
              disabled={!run}
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
          </div>
          <canvas
            ref={canvasRef}
            width={860}
            height={480}
            className="w-full rounded-lg border border-slate-800 bg-slate-950/60"
            aria-label="Simulation scene (side view)"
          />
        </div>
      </section>
    </div>
  );
}
