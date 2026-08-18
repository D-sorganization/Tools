import { type TorqueFit } from "../model/torqueProfileEditor";

interface Props { fit: TorqueFit | null }

const COLORS = { shoulder: "#38bdf8", wrist: "#c084fc" };
const MAX_POINT_MARKERS = 41;

function representativeRows<T>(values: readonly T[]): readonly T[] {
  if (values.length <= MAX_POINT_MARKERS) return values;
  return Array.from({ length: MAX_POINT_MARKERS }, (_, index) =>
    values[Math.round((index * (values.length - 1)) / (MAX_POINT_MARKERS - 1))]);
}

export function TorqueFitPreview({ fit }: Props) {
  if (!fit) {
    return <p className="mt-2 text-xs text-slate-500">Fit the point table to preview both torque curves.</p>;
  }
  const width = 360; const height = 180; const padding = 28;
  const start = fit.rows[0].timeS;
  const end = fit.rows[fit.rows.length - 1].timeS;
  const markerRows = representativeRows(fit.rows);
  const samples = Array.from({ length: 81 }, (_, index) => {
    const time = start + (index / 80) * (end - start);
    return { time, shoulder: fit.shoulder.evaluate(time), wrist: fit.wrist.evaluate(time) };
  });
  const torques = [
    ...samples.flatMap((point) => [point.shoulder, point.wrist]),
    ...fit.rows.flatMap((row) => [row.shoulderNm, row.wristNm]),
  ];
  const minimum = Math.min(...torques); const maximum = Math.max(...torques);
  const span = maximum === minimum ? 1 : maximum - minimum;
  const x = (time: number) => padding + ((time - start) / (end - start)) * (width - 2 * padding);
  const y = (torque: number) => height - padding - ((torque - minimum) / span) * (height - 2 * padding);
  const path = (joint: "shoulder" | "wrist") => samples.map((point, index) =>
    `${index === 0 ? "M" : "L"}${x(point.time).toFixed(2)},${y(point[joint]).toFixed(2)}`).join(" ");
  return (
    <figure className="mt-3 rounded border border-slate-700 bg-slate-950/70 p-2">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Fitted shoulder and wrist torque curves with sample points" className="w-full">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#475569" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#475569" />
        <path d={path("shoulder")} fill="none" stroke={COLORS.shoulder} strokeWidth="2" />
        <path d={path("wrist")} fill="none" stroke={COLORS.wrist} strokeWidth="2" />
        {markerRows.flatMap((row) => ([
          <circle key={`s-${row.timeS}`} cx={x(row.timeS)} cy={y(row.shoulderNm)} r="3" fill={COLORS.shoulder} />,
          <circle key={`w-${row.timeS}`} cx={x(row.timeS)} cy={y(row.wristNm)} r="3" fill={COLORS.wrist} />,
        ]))}
        <text x={padding} y={height - 6} fill="#94a3b8" fontSize="10">{start.toFixed(2)} s</text>
        <text x={width - padding} y={height - 6} textAnchor="end" fill="#94a3b8" fontSize="10">{end.toFixed(2)} s</text>
      </svg>
      <figcaption className="flex gap-4 text-xs text-slate-400">
        <span style={{ color: COLORS.shoulder }}>● Shoulder</span>
        <span style={{ color: COLORS.wrist }}>● Wrist</span>
        <span>Dots: samples · Lines: fitted curves</span>
        {markerRows.length < fit.rows.length && <span>{markerRows.length} of {fit.rows.length} points shown</span>}
      </figcaption>
    </figure>
  );
}
