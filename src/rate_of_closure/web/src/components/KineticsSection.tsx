/**
 * Kinetics plots section (#4125 H2) — the web twin of the PyQt6
 * KineticsPanel: Joint Torques, Joint Power, and Reaction Forces line
 * charts for the current run plus a peak table (timing as % of the
 * downswing). Data comes from the TS inverse-dynamics mirror
 * (`model/kinetics.ts`, parity-pinned against pytest).
 *
 * Axis-label conventions mirror the movement optimizer's
 * plot_renderer.py: "Time (s)", "Torque (N·m)", "Power (W)",
 * "Force (N)". The 3D playback overlay is deferred to the P7 WASM
 * pass (SPEC.md deviation note); these charts are the web
 * presentation of the kinetics.
 */

import { useEffect, useMemo, useRef } from "react";

import { kineticsForInput, type KineticsSeriesTs } from "../model/kinetics";
import { type SimulationInput, type SimulationRunTs } from "../model/simulation";
import { getChartColor } from "../model/theme";

interface Props {
  input: SimulationInput;
  run: SimulationRunTs | null;
}

const MARGIN = 36;
// H6 accent alignment (#4125): draw from the shared chart palette
// (model/theme.ts) instead of widget-local hex.
const COLORS = [getChartColor(0), getChartColor(2), getChartColor(1)];

interface Trace {
  label: string;
  values: number[];
  dashed?: boolean;
}

function drawChart(
  canvas: HTMLCanvasElement,
  t: number[],
  traces: Trace[],
  yLabel: string,
  title: string,
  impactTimeS: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const all = traces.flatMap((trace) => trace.values);
  const lo = Math.min(0, ...all);
  const hi = Math.max(0, ...all);
  const span = hi - lo || 1;
  const tMax = t[t.length - 1] || 1;
  const px = (x: number) => MARGIN + (x / tMax) * (width - MARGIN - 8);
  const py = (v: number) =>
    height - MARGIN - ((v - lo) / span) * (height - MARGIN - 22);

  // Zero line (signed quantities) + impact marker.
  ctx.strokeStyle = "#475569";
  ctx.beginPath();
  ctx.moveTo(MARGIN, py(0));
  ctx.lineTo(width, py(0));
  ctx.stroke();
  ctx.strokeStyle = "#facc15";
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(px(impactTimeS), 16);
  ctx.lineTo(px(impactTimeS), height - MARGIN);
  ctx.stroke();
  ctx.setLineDash([]);

  traces.forEach((trace, index) => {
    ctx.strokeStyle = COLORS[index % COLORS.length];
    ctx.lineWidth = 1.8;
    if (trace.dashed) ctx.setLineDash([6, 4]);
    ctx.beginPath();
    trace.values.forEach((v, i) => {
      if (i === 0) ctx.moveTo(px(t[i]), py(v));
      else ctx.lineTo(px(t[i]), py(v));
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS[index % COLORS.length];
    ctx.font = "11px sans-serif";
    ctx.fillText(trace.label, MARGIN + 6 + index * 110, 12);
  });
  ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  ctx.fillText(title, width / 2 - 30, 12);
  ctx.save();
  ctx.translate(10, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, -24, 0);
  ctx.restore();
  ctx.fillText("Time (s)", width / 2 - 20, height - 6);
}

function peak(t: number[], values: number[], tau: number): [number, number] {
  let best = 0;
  let bestT = 0;
  for (let i = 0; i < values.length && t[i] <= tau; i += 1) {
    if (Math.abs(values[i]) > best) {
      best = Math.abs(values[i]);
      bestT = t[i];
    }
  }
  return [best, tau > 0 ? (100 * bestT) / tau : 0];
}

export function KineticsSection({ input, run }: Props) {
  const series: KineticsSeriesTs | null = useMemo(
    () => (run ? kineticsForInput(input) : null),
    [input, run],
  );
  const torqueRef = useRef<HTMLCanvasElement | null>(null);
  const ztcfTorqueRef = useRef<HTMLCanvasElement | null>(null);
  const powerRef = useRef<HTMLCanvasElement | null>(null);
  const forceRef = useRef<HTMLCanvasElement | null>(null);
  const ztcfForceRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!series || !run) return;
    const tau = run.impactTimeS ?? run.swing[run.swing.length - 1].t;
    if (torqueRef.current)
      drawChart(
        torqueRef.current,
        series.tS,
        [
          { label: "shoulder", values: series.shoulderTorqueNm },
          { label: "wrist", values: series.wristTorqueNm },
        ],
        "Torque (N·m)",
        "Joint Torques",
        tau,
      );
    if (powerRef.current)
      drawChart(
        powerRef.current,
        series.tS,
        [
          { label: "shoulder", values: series.shoulderPowerW },
          { label: "wrist", values: series.wristPowerW },
          {
            label: "Total",
            values: series.shoulderPowerW.map(
              (v, i) => v + series.wristPowerW[i],
            ),
            dashed: true,
          },
        ],
        "Power (W)",
        "Joint Power",
        tau,
      );
    if (ztcfTorqueRef.current)
      drawChart(
        ztcfTorqueRef.current,
        series.tS,
        [
          { label: "shoulder ZTCF", values: series.shoulderZtcfTorqueNm },
          { label: "wrist ZTCF", values: series.wristZtcfTorqueNm },
        ],
        "Torque (N·m)",
        "State-Matched ZTCF Torque",
        tau,
      );
    if (forceRef.current)
      drawChart(
        forceRef.current,
        series.tS,
        [
          { label: "shoulder", values: series.shoulderForceN },
          { label: "wrist", values: series.wristForceN },
          { label: "clubhead", values: series.clubheadForceN },
        ],
        "Force (N)",
        "Reaction Forces",
        tau,
      );
    if (ztcfForceRef.current)
      drawChart(
        ztcfForceRef.current,
        series.tS,
        [
          { label: "shoulder ZTCF", values: series.shoulderZtcfForceN },
          { label: "wrist ZTCF", values: series.wristZtcfForceN },
          { label: "clubhead ZTCF", values: series.clubheadZtcfForceN },
        ],
        "Force (N)",
        "State-Matched ZTCF Reaction Forces",
        tau,
      );
  }, [series, run]);

  if (!run || !series) {
    return (
      <p className="text-sm text-slate-400">
        Kinetics need the pendulum joint states — select the Double
        Pendulum swing source and run the simulation (manual source not
        supported).
      </p>
    );
  }

  const tau = run.impactTimeS ?? run.swing[run.swing.length - 1].t;
  const rows = [
    {
      name: "shoulder",
      torque: peak(series.tS, series.shoulderTorqueNm, tau),
      power: peak(series.tS, series.shoulderPowerW, tau),
      force: peak(series.tS, series.shoulderForceN, tau),
    },
    {
      name: "wrist",
      torque: peak(series.tS, series.wristTorqueNm, tau),
      power: peak(series.tS, series.wristPowerW, tau),
      force: peak(series.tS, series.wristForceN, tau),
    },
    {
      name: "clubhead",
      torque: null,
      power: null,
      force: peak(series.tS, series.clubheadForceN, tau),
    },
  ];
  return (
    <div className="space-y-3">
      {run.impactTimeS === null && (
        <p role="status" className="rounded-lg border border-amber-500/30 bg-amber-950/20 p-3 text-sm text-amber-200">
          No club–ball impact occurred. Kinetics remain available for the
          complete simulated swing; peak timing is referenced to the end of
          that swing instead of a fabricated impact instant.
        </p>
      )}
      <p className="rounded-lg border border-sky-500/30 bg-sky-950/20 p-3 text-sm text-sky-100">
        <strong>Zero-Torque Counterfactual (ZTCF):</strong> at every recorded
        state, commanded shoulder and wrist torques are set to zero while
        gravity, damping, and velocity-dependent coupling remain active. Each
        sample is evaluated at the original state; this is a passive-drift
        diagnostic, not one continuously integrated alternate swing.
      </p>
      {[torqueRef, ztcfTorqueRef, powerRef, forceRef, ztcfForceRef].map((ref, index) => (
        <canvas
          key={index}
          ref={ref}
          width={840}
          height={190}
          className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
          aria-label={
            [
              "Joint torques chart",
              "Zero-torque counterfactual torques chart",
              "Joint power chart",
              "Reaction forces chart",
              "Zero-torque counterfactual forces chart",
            ][index]
          }
        />
      ))}
      <table
        className="w-full text-left text-xs text-slate-300"
        title="Peak torque, power, and reaction-force magnitudes per joint with timing as % of the downswing (swing start to impact)"
      >
        <thead>
          <tr className="text-slate-400">
            <th className="pr-3">Joint</th>
            <th className="pr-3">Peak |Torque| (N·m)</th>
            <th className="pr-3">@ % downswing</th>
            <th className="pr-3">Peak |Power| (W)</th>
            <th className="pr-3">@ % downswing</th>
            <th className="pr-3">Peak |Force| (N)</th>
            <th>@ % downswing</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name}>
              <td className="pr-3">{row.name}</td>
              <td className="pr-3">{row.torque ? row.torque[0].toFixed(1) : "—"}</td>
              <td className="pr-3">{row.torque ? `${row.torque[1].toFixed(0)}%` : "—"}</td>
              <td className="pr-3">{row.power ? row.power[0].toFixed(0) : "—"}</td>
              <td className="pr-3">{row.power ? `${row.power[1].toFixed(0)}%` : "—"}</td>
              <td className="pr-3">{row.force[0].toFixed(0)}</td>
              <td>{`${row.force[1].toFixed(0)}%`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
