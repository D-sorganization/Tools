import type { LearningCurvePoint, ModelMetric } from "../model/neuralModelBundle";

const axis = "#64748b";
const label = "#94a3b8";

export interface PredictionPoint {
  row: number;
  output: string;
  unit: string;
  predicted: number;
  observed?: number;
}

function polyline(points: Array<[number, number]>): string {
  return points.map(([xValue, yValue]) => `${xValue.toFixed(1)},${yValue.toFixed(1)}`).join(" ");
}

export function LearningCurveChart({ points }: { points: LearningCurvePoint[] }) {
  if (!points.length) return <p className="text-xs text-slate-500">No learning-curve history was included.</p>;
  const xValues = points.map((point, index) => point.trainingRows ?? point.epoch ?? index + 1);
  const maximumX = Math.max(1, ...xValues);
  const losses = points.flatMap(({ trainingLoss, validationLoss }) =>
    trainingLoss === undefined ? [validationLoss] : [trainingLoss, validationLoss]);
  const maximumLoss = Math.max(1e-12, ...losses);
  const project = (xValue: number, loss: number): [number, number] =>
    [45 + xValue / maximumX * 560, 220 - loss / maximumLoss * 190];
  return <svg viewBox="0 0 640 250" role="img" aria-label="Training and validation learning curve"
    className="h-64 w-full rounded border border-slate-800 bg-slate-950">
    <title>Loss by epoch from the imported training artifact; units follow the training objective.</title>
    <line x1="45" y1="220" x2="610" y2="220" stroke={axis} />
    <line x1="45" y1="25" x2="45" y2="220" stroke={axis} />
    {points.some(({ trainingLoss }) => trainingLoss !== undefined) && <polyline fill="none" stroke="#38bdf8" strokeWidth="2" points={polyline(points.flatMap(
      ({ trainingRows, epoch, trainingLoss }, index) => trainingLoss === undefined ? [] :
      [project(trainingRows ?? epoch ?? index + 1, trainingLoss)]))} />}
    <polyline fill="none" stroke="#f59e0b" strokeWidth="2" points={polyline(points.map(
      ({ trainingRows, epoch, validationLoss }, index) => project(trainingRows ?? epoch ?? index + 1, validationLoss)))} />
    <text x="325" y="243" textAnchor="middle" fill={label} fontSize="12">Training rows or epoch (count)</text>
    <text x="15" y="125" textAnchor="middle" fill={label} fontSize="12" transform="rotate(-90 15 125)">Validation standardized RMSE or loss</text>
    {points.some(({ trainingLoss }) => trainingLoss !== undefined) && <text x="460" y="18" fill="#38bdf8" fontSize="11">training</text>}
    <text x="530" y="18" fill="#f59e0b" fontSize="11">validation</text>
  </svg>;
}

export function MetricCards({ metrics }: { metrics: ModelMetric[] }) {
  if (!metrics.length) return <p className="text-xs text-slate-500">No held-out metrics were included.</p>;
  return <div className="grid gap-2 sm:grid-cols-3">{metrics.map((metric) =>
    <div key={`${metric.name}-${metric.split}`} title="Metric imported from the artifact; interpret using its named split and provenance."
      className="rounded border border-slate-800 p-2 text-sm">
      <strong>{metric.name}</strong><br />{metric.value.toFixed(4)} {metric.unit ?? "unitless"}<br />
      <span className="text-xs text-slate-400">{metric.split} split</span>
    </div>)}</div>;
}

export function ResidualChart({ points }: { points: PredictionPoint[] }) {
  const residuals = points.filter((point) => point.observed !== undefined).map((point) => ({
    ...point, residual: point.predicted - (point.observed ?? 0),
  }));
  if (!residuals.length) return <p className="text-xs text-slate-500">Batch rows with observed output columns enable residual plotting.</p>;
  const outputs = [...new Set(residuals.map(({ output }) => output))];
  return <div className="space-y-3">{outputs.map((output) => {
    const series = residuals.filter((point) => point.output === output);
    const bound = Math.max(1e-12, ...series.map(({ residual }) => Math.abs(residual)));
    const denominator = Math.max(1, series.length - 1);
    const unit = series[0].unit;
    return <svg key={output} viewBox="0 0 640 250" role="img" aria-label={`${output} prediction residuals by batch row`}
      className="h-64 w-full rounded border border-slate-800 bg-slate-950">
      <title>{`${output} residual is predicted minus observed in ${unit}.`}</title>
      <line x1="45" y1="120" x2="610" y2="120" stroke={axis} />
      <line x1="45" y1="25" x2="45" y2="220" stroke={axis} />
      {series.map((point, index) => <circle key={`${point.row}-${point.output}`} r="3" fill="#a78bfa"
        cx={45 + index / denominator * 560} cy={120 - point.residual / bound * 90}>
        <title>{`Row ${point.row}: ${point.residual.toFixed(4)} ${point.unit}`}</title>
      </circle>)}
      <text x="325" y="243" textAnchor="middle" fill={label} fontSize="12">Batch row (count)</text>
      <text x="15" y="125" textAnchor="middle" fill={label} fontSize="12" transform="rotate(-90 15 125)">{`Residual (${unit})`}</text>
    </svg>;
  })}</div>;
}
