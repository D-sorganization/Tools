import type { LaunchMonitorRow } from "../model/launchMonitorAnalysis";
import { finiteLaunchMonitorScalar } from "../model/launchMonitorAnalysisTypes";
import { metricLabel } from "../model/launchMonitorMetricUnits";

const colors = ["#38bdf8", "#f59e0b", "#34d399", "#c084fc", "#fb7185", "#a3e635"];
const frame = { left: 60, top: 22, width: 548, height: 190 };

const scale = (value: number, low: number, high: number, start: number, span: number) =>
  start + (value - low) / Math.max(Number.EPSILON, high - low) * span;

interface PlotPoint { playerId: string; x: number; y: number; shotId: string }

const pointsFromRows = (rows: LaunchMonitorRow[], columns: {
  x: string; y: string; player: string; selectedPlayer: string;
}): PlotPoint[] => rows.flatMap((row, index) => {
  const x = finiteLaunchMonitorScalar(row[columns.x]);
  const y = finiteLaunchMonitorScalar(row[columns.y]);
  const playerId = String(row[columns.player] ?? "").trim();
  if (!playerId || x === null || y === null ||
    (columns.selectedPlayer && playerId !== columns.selectedPlayer)) return [];
  return [{ playerId, x, y, shotId: String(row.shot_id ?? index + 1) }];
});

const centered = (points: PlotPoint[]) => {
  const grouped = new Map<string, PlotPoint[]>();
  points.forEach((point) => grouped.set(point.playerId, [...(grouped.get(point.playerId) ?? []), point]));
  return [...grouped.values()].flatMap((group) => {
    const xMean = group.reduce((sum, point) => sum + point.x, 0) / group.length;
    const yMean = group.reduce((sum, point) => sum + point.y, 0) / group.length;
    return group.map((point) => ({ ...point, x: point.x - xMean, y: point.y - yMean }));
  });
};

export function LaunchMonitorCovariationChart({ rows, xColumn, yColumn, playerColumn, selectedPlayer }: {
  rows: LaunchMonitorRow[]; xColumn: string; yColumn: string;
  playerColumn: string; selectedPlayer: string;
}) {
  const original = pointsFromRows(rows, {
    x: xColumn, y: yColumn, player: playerColumn, selectedPlayer,
  });
  const points = centered(original);
  if (points.length < 2) return <p className="text-sm text-amber-200">At least two complete shots are needed for the plot.</p>;
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  const xLow = Math.min(...xValues); const xHigh = Math.max(...xValues);
  const yLow = Math.min(...yValues); const yHigh = Math.max(...yValues);
  const players = [...new Set(points.map((point) => point.playerId))].sort();
  return <svg id="launch-monitor-covariation-plot" viewBox="0 0 640 250" role="img"
    aria-label={`Within-player centered ${metricLabel(yColumn)} versus ${metricLabel(xColumn)} scatter plot`}
    className="h-64 w-full rounded-lg border border-slate-800 bg-slate-950">
    <title>Player-mean-centered shot values; colors identify players and hover reveals backing values</title>
    <line x1={frame.left} y1={frame.top + frame.height} x2={frame.left + frame.width}
      y2={frame.top + frame.height} stroke="#475569" />
    <line x1={frame.left} y1={frame.top} x2={frame.left} y2={frame.top + frame.height}
      stroke="#475569" />
    {points.map((point, index) => <circle key={`${point.playerId}-${point.shotId}-${index}`}
      cx={scale(point.x, xLow, xHigh, frame.left, frame.width)}
      cy={frame.top + frame.height - scale(point.y, yLow, yHigh, 0, frame.height)}
      r="3" fill={colors[players.indexOf(point.playerId) % colors.length]} opacity="0.75">
      <title>{`${point.playerId} · ${point.shotId}: centered ${metricLabel(xColumn)}=${point.x.toFixed(3)}; centered ${metricLabel(yColumn)}=${point.y.toFixed(3)}`}</title>
    </circle>)}
    <text x="334" y="242" textAnchor="middle" fill="#94a3b8" fontSize="12">
      Player-centered {metricLabel(xColumn)}
    </text>
    <text x="15" y="117" textAnchor="middle" fill="#94a3b8" fontSize="12"
      transform="rotate(-90 15 117)">Player-centered {metricLabel(yColumn)}</text>
  </svg>;
}
