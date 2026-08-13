/** Visual-first green and speed evidence for the putting workspace. */

import { captureSpeedMps, simulatePutt } from "../model/putting";

type PuttResult = ReturnType<typeof simulatePutt> | null;

export function PuttingVisuals(props: {
  result: PuttResult; holeX: number; grade: number; aspect: number;
}) {
  return <>
    <GreenView {...props} />
    <SpeedPlot result={props.result} />
  </>;
}

function GreenView({ result, holeX, grade, aspect }: {
  result: PuttResult; holeX: number; grade: number; aspect: number;
}) {
  const width = 640; const height = 320;
  if (!result) return <p className="text-sm text-slate-400">Inputs out of range.</p>;
  const xs = result.pathXM; const ys = result.pathYM;
  const maxX = Math.max(holeX + 0.5, ...xs) + 0.3;
  const minX = Math.min(0, ...xs) - 0.3;
  const spanY = Math.max(0.8, 2 * Math.max(...ys.map(Math.abs), 0.3));
  const sx = (value: number) => ((value - minX) / (maxX - minX)) * width;
  const sy = (value: number) => height / 2 - (value / spanY) * height;
  const points = (from: number, to: number) => xs.slice(from, to)
    .map((value, index) => `${sx(value).toFixed(1)},${sy(ys[from + index]).toFixed(1)}`)
    .join(" ");
  const split = result.skidEndIndex; const arrowLength = 40;
  const arrowX = sx(holeX * 0.5); const arrowY = sy(0);
  return <figure aria-label="Top-down green view: skid phase orange, pure roll green, hole circle, downhill arrow"
    className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Putt path on the green"
      className="min-h-[180px] w-full rounded-lg bg-emerald-950/60 sm:min-h-0">
      <polyline points={points(0, split + 1)} fill="none" stroke="#fb923c" strokeWidth={3} />
      <polyline points={points(split, xs.length)} fill="none" stroke="#4ade80" strokeWidth={3} />
      <circle cx={sx(holeX)} cy={sy(0)} r={Math.max(5, (0.054 / (maxX - minX)) * width)}
        fill="none" stroke="#f8fafc" strokeWidth={2} />
      <circle cx={sx(0)} cy={sy(0)} r={4} fill="#f8fafc" />
      {grade > 0 && <g stroke="#94a3b8" strokeWidth={2}>
        <line x1={arrowX} y1={arrowY}
          x2={arrowX + arrowLength * Math.cos((aspect * Math.PI) / 180)}
          y2={arrowY - arrowLength * Math.sin((aspect * Math.PI) / 180)}
          markerEnd="url(#downhill-arrow)" />
        <defs><marker id="downhill-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3"
          orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#94a3b8" /></marker></defs>
      </g>}
      {result.holed && <text x={sx(holeX)} y={sy(0) - 14} textAnchor="middle"
        fill="#4ade80" fontSize="13">HOLED</text>}
    </svg>
    <figcaption className="mt-2 text-xs text-slate-400">Orange = skid phase, green = pure roll;
      the circle is the hole, the grey arrow points downhill. Left is the putt's left (+y).</figcaption>
  </figure>;
}

function SpeedPlot({ result }: { result: PuttResult }) {
  const width = 640; const height = 180;
  if (!result) return null;
  const distances = result.pathXM.map((_, index) => index === 0 ? 0 : Math.hypot(
    result.pathXM[index] - result.pathXM[index - 1],
    result.pathYM[index] - result.pathYM[index - 1],
  )).reduce<number[]>((values, increment) => [
    ...values, (values[values.length - 1] ?? 0) + increment,
  ], []);
  const maxDistance = Math.max(distances[distances.length - 1], 0.1);
  const maxSpeed = Math.max(...result.speedsMps, captureSpeedMps()) * 1.08;
  const sx = (value: number) => (value / maxDistance) * (width - 20) + 10;
  const sy = (value: number) => height - 16 - (value / maxSpeed) * (height - 32);
  const points = distances.map((value, index) =>
    `${sx(value).toFixed(1)},${sy(result.speedsMps[index]).toFixed(1)}`).join(" ");
  const split = distances[Math.min(result.skidEndIndex, distances.length - 1)];
  return <figure aria-label="Ball speed versus distance with the capture-speed bound and the skid-to-roll transition marked"
    className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Speed vs distance plot"
      className="w-full rounded-lg bg-slate-950/60">
      <polyline points={points} fill="none" stroke="#38bdf8" strokeWidth={2.5} />
      <line x1={10} x2={width - 10} y1={sy(captureSpeedMps())} y2={sy(captureSpeedMps())}
        stroke="#f87171" strokeWidth={1.5} strokeDasharray="6 4" />
      <line x1={sx(split)} x2={sx(split)} y1={12} y2={height - 16}
        stroke="#fb923c" strokeWidth={1.5} strokeDasharray="3 4" />
    </svg>
    <figcaption className="mt-2 text-xs text-slate-400">Blue: speed vs distance. Red:
      capture-speed bound. Orange: skid-to-roll transition.</figcaption>
  </figure>;
}
