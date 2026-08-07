import type { LaunchMonitorRow } from "../model/launchMonitorAnalysis";
import {
  metricLabel,
  type DispersionSummary,
  type SessionTrend,
} from "../model/launchMonitorPlayerAnalytics";

const frame = { left: 54, top: 20, width: 560, height: 194 };

const scale = (value: number, low: number, high: number, start: number, span: number) =>
  start + (value - low) / Math.max(Number.EPSILON, high - low) * span;

export function ScatterPlot({ rows, outcome, predictor }: {
  rows: LaunchMonitorRow[]; outcome: string; predictor: string;
}) {
  const pairs = rows.map((row) => [Number(row[predictor]), Number(row[outcome])])
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pairs.length < 2) return <p className="text-sm text-slate-500">Select two populated variables.</p>;
  const xs = pairs.map(([x]) => x); const ys = pairs.map(([, y]) => y);
  const xMin = Math.min(...xs); const xMax = Math.max(...xs);
  const yMin = Math.min(...ys); const yMax = Math.max(...ys);
  return (
    <svg id="launch-monitor-relationship-plot" viewBox="0 0 640 250" role="img"
      aria-label={`${metricLabel(outcome)} versus ${metricLabel(predictor)} scatter plot`}
      className="h-64 w-full rounded-lg border border-slate-800 bg-slate-950">
      <title>{metricLabel(outcome)} versus {metricLabel(predictor)}; each point is one retained shot</title>
      <line x1="54" y1="214" x2="614" y2="214" stroke="#475569" />
      <line x1="54" y1="20" x2="54" y2="214" stroke="#475569" />
      {pairs.map(([x, y], index) => (
        <circle key={index} cx={scale(x, xMin, xMax, frame.left, frame.width)}
          cy={frame.top + frame.height - scale(y, yMin, yMax, 0, frame.height)}
          r="3" fill="#38bdf8" opacity="0.72">
          <title>{`${metricLabel(predictor)}: ${x}; ${metricLabel(outcome)}: ${y}`}</title>
        </circle>
      ))}
      <text x="334" y="242" textAnchor="middle" fill="#94a3b8" fontSize="12">{metricLabel(predictor)}</text>
      <text x="15" y="117" textAnchor="middle" fill="#94a3b8" fontSize="12"
        transform="rotate(-90 15 117)">{metricLabel(outcome)}</text>
    </svg>
  );
}

export function DispersionPlot({ summary }: { summary: DispersionSummary }) {
  const carries = summary.points.map((point) => point.carryYards ?? 0);
  const laterals = summary.points.map((point) => point.lateralYards);
  const carryMin = Math.min(...carries); const carryMax = Math.max(...carries);
  const lateralBound = Math.max(1, ...laterals.map(Math.abs));
  return (
    <svg id="launch-monitor-dispersion-plot" viewBox="0 0 640 250" role="img"
      aria-label="Shot dispersion in yards left or right of target"
      className="h-64 w-full rounded-lg border border-slate-800 bg-slate-950">
      <title>Directional dispersion; negative lateral yards are left and positive lateral yards are right</title>
      <line x1={scale(0, -lateralBound, lateralBound, frame.left, frame.width)} y1={frame.top}
        x2={scale(0, -lateralBound, lateralBound, frame.left, frame.width)} y2={frame.top + frame.height}
        stroke="#f8fafc" strokeDasharray="5 4" opacity="0.55" />
      {summary.points.map((point, index) => {
        const carry = point.carryYards ?? index;
        return <circle key={`${point.shotId}-${index}`}
          cx={scale(point.lateralYards, -lateralBound, lateralBound, frame.left, frame.width)}
          cy={frame.top + frame.height - scale(carry, carryMin, carryMax, 0, frame.height)}
          r="3" fill={point.lateralYards < 0 ? "#f59e0b" : "#38bdf8"} opacity="0.72">
          <title>{`${point.shotId}: ${point.lateralYards.toFixed(2)} yd lateral${point.carryYards === null ? "" : `, ${point.carryYards.toFixed(2)} yd carry`}`}</title>
        </circle>;
      })}
      <text x="334" y="242" textAnchor="middle" fill="#94a3b8" fontSize="12">Lateral from target line (yd; left − / right +)</text>
      <text x="15" y="117" textAnchor="middle" fill="#94a3b8" fontSize="12"
        transform="rotate(-90 15 117)">{summary.carryColumn ? "Carry distance (yd)" : "Shot sequence (unitless)"}</text>
    </svg>
  );
}

export function SessionTrendPlot({ trend }: { trend: SessionTrend }) {
  const points = trend.players.flatMap((player) => player.points);
  const values = points.map((point) => point.mean);
  const low = Math.min(...values); const high = Math.max(...values);
  const maxOrder = Math.max(1, ...points.map((point) => point.order));
  const colors = ["#34d399", "#38bdf8", "#f59e0b", "#c084fc", "#fb7185"];
  return (
    <svg id="launch-monitor-session-plot" viewBox="0 0 640 250" role="img"
      aria-label={`${metricLabel(trend.outcome)} session mean trend`}
      className="h-64 w-full rounded-lg border border-slate-800 bg-slate-950">
      <title>Per-player session means; each player is sequenced and fitted independently</title>
      {trend.players.map((player, playerIndex) => {
        const color = colors[playerIndex % colors.length];
        const path = player.points.map((point, index) => {
          const x = scale(point.order, 0, maxOrder, frame.left, frame.width);
          const y = frame.top + frame.height - scale(point.mean, low, high, 0, frame.height);
          return `${index ? "L" : "M"}${x},${y}`;
        }).join(" ");
        return <g key={player.playerId} aria-label={`Player ${player.playerId}`}>
          <path d={path} fill="none" stroke={color} strokeWidth="2" />
          {player.points.map((point) => <circle key={`${point.playerId}-${point.sessionId}`}
            cx={scale(point.order, 0, maxOrder, frame.left, frame.width)}
            cy={frame.top + frame.height - scale(point.mean, low, high, 0, frame.height)}
            r="4" fill={color}><title>{`${point.playerId} · ${point.sessionId}: ${point.mean.toFixed(3)} ${trend.unit}, N=${point.sampleCount}`}</title></circle>)}
          <text x={510} y={30 + playerIndex * 14} fill={color} fontSize="10">{player.playerId}</text>
        </g>;
      })}
      <text x="334" y="242" textAnchor="middle" fill="#94a3b8" fontSize="12">Session order</text>
      <text x="15" y="117" textAnchor="middle" fill="#94a3b8" fontSize="12"
        transform="rotate(-90 15 117)">{metricLabel(trend.outcome)}</text>
    </svg>
  );
}
