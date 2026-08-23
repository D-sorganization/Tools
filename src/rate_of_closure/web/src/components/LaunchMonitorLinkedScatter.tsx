import { useMemo, type KeyboardEvent, type PointerEvent } from "react";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysis";
import {
  navigateLinkedScatter,
  planLinkedScatter,
  projectPlotAxis,
  type LinkedScatterPlan,
  type LinkedScatterPoint,
} from "../model/launchMonitorLinkedScatter";

const WIDTH = 640; const HEIGHT = 250;
const LEFT = 52; const TOP = 18; const PLOT_WIDTH = 568; const PLOT_HEIGHT = 197;

const coordinates = (plan: LinkedScatterPlan) => {
  if (plan.points.length === 0) return [];
  const x = projectPlotAxis(plan.points.map((point) => point.x));
  const y = projectPlotAxis(plan.points.map((point) => point.y));
  return plan.points.map((point, index) => ({
    point,
    cx: LEFT + (x.coordinates[index] + 1) / 2 * PLOT_WIDTH,
    cy: TOP + (1 - y.coordinates[index]) / 2 * PLOT_HEIGHT,
  }));
};

const selectedText = (
  point: LinkedScatterPoint | null, selectedRawIndex: number | null,
  xField: string, yField: string,
) => {
  if (selectedRawIndex === null) return "No retained source row selected.";
  if (point === null) {
    return `Retained row index ${selectedRawIndex} is unavailable for the current axes.`;
  }
  const fields = [
    point.shotId && `shot ${point.shotId}`,
    point.sessionId && `session ${point.sessionId}`,
    point.monitorVendor && `vendor ${point.monitorVendor}`,
  ].filter(Boolean);
  return `Retained row index ${point.rawIndex} (zero-based); ${fields.join("; ") || "no source identifiers present"}; ` +
    `${xField} ${point.x}; ${yField} ${point.y}.`;
};

export function LaunchMonitorLinkedScatter({
  rows, xField, yField, selectedRawIndex, onSelectedRawIndex,
}: {
  readonly rows: readonly LaunchMonitorRow[];
  readonly xField: string;
  readonly yField: string;
  readonly selectedRawIndex: number | null;
  readonly onSelectedRawIndex: (rawIndex: number | null) => void;
}) {
  const axesValid = Boolean(xField && yField && xField !== yField);
  const plan = useMemo<LinkedScatterPlan>(
    () => axesValid ? planLinkedScatter(rows, xField, yField, selectedRawIndex) : Object.freeze({
      xField, yField, rawCount: rows.length, finiteCount: 0, displayedCount: 0,
      selectedRawIndex: null, points: Object.freeze([]),
    }), [rows, xField, yField, selectedRawIndex, axesValid],
  );
  const plotted = useMemo(() => coordinates(plan), [plan]);
  const selected = plotted.find(({ point }) => point.rawIndex === selectedRawIndex) ?? null;
  // ⚡ Bolt Optimization: Wrap path generation in useMemo and use a single-pass loop
  // to eliminate intermediate array allocations during large SVG path string generation.
  const path = useMemo(() => {
    let d = "";
    for (let i = 0; i < plotted.length; i++) {
      d += `M${plotted[i].cx.toFixed(2)} ${plotted[i].cy.toFixed(2)}l0.01 0`;
    }
    return d;
  }, [plotted]);
  const chooseNearest = (event: PointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * WIDTH / bounds.width;
    const y = (event.clientY - bounds.top) * HEIGHT / bounds.height;
    const nearest = plotted.reduce<typeof plotted[number] | null>((best, candidate) => {
      if (best === null) return candidate;
      const distance = (candidate.cx - x) ** 2 + (candidate.cy - y) ** 2;
      const bestDistance = (best.cx - x) ** 2 + (best.cy - y) ** 2;
      return distance < bestDistance ? candidate : best;
    }, null);
    if (nearest !== null) onSelectedRawIndex(nearest.point.rawIndex);
  };
  const navigate = (event: KeyboardEvent<SVGSVGElement>) => {
    const command = ({
      ArrowLeft: "previous", ArrowRight: "next", Home: "home", End: "end", Escape: "clear",
    } as const)[event.key];
    if (command === undefined) return;
    event.preventDefault();
    onSelectedRawIndex(navigateLinkedScatter(plan, selectedRawIndex, command));
  };
  if (!axesValid) {
    return <p className="text-sm text-slate-500">Select two populated variables.</p>;
  }
  if (plan.finiteCount < 2) {
    return <p className="text-sm text-slate-500">
      Selected variables need at least two jointly finite pairs ({plan.finiteCount} available).
    </p>;
  }
  return <div>
    <p id="linked-scatter-instructions" className="mb-2 text-xs text-slate-400">
      Select the nearest displayed point with the pointer. Left/Right, Home/End navigate;
      Escape clears. Selection identifies only the retained row ordinal.
    </p>
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img" tabIndex={0}
      aria-label={`${yField} versus ${xField} linked scatter plot`}
      aria-describedby="linked-scatter-instructions linked-scatter-status"
      onPointerDown={chooseNearest} onKeyDown={navigate}
      className="h-64 min-h-[180px] w-full rounded-lg border border-slate-800 bg-slate-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-300">
      <line x1={LEFT} y1={215} x2={620} y2={215} stroke="#475569" />
      <line x1={LEFT} y1={TOP} x2={LEFT} y2={215} stroke="#475569" />
      <path d={path} fill="none" stroke="#38bdf8" strokeWidth="6" strokeLinecap="round" opacity="0.72" />
      {selected && <circle aria-label={`Selected retained row ${selected.point.rawIndex}`}
        cx={selected.cx} cy={selected.cy} r="7" fill="none" stroke="#fbbf24" strokeWidth="3" />}
      <text x="336" y="242" textAnchor="middle" fill="#94a3b8" fontSize="12">{xField}</text>
      <text x="15" y="116" textAnchor="middle" fill="#94a3b8" fontSize="12"
        transform="rotate(-90 15 116)">{yField}</text>
    </svg>
    <p className="mt-2 text-xs text-slate-500">
      Displayed {plan.displayedCount.toLocaleString()} of {plan.finiteCount.toLocaleString()} finite
      pairs from {plan.rawCount.toLocaleString()} retained rows. All rows remain retained for export;
      the selected missing-data policy controls analysis inclusion.
    </p>
    <p id="linked-scatter-status" aria-live="polite" className="mt-1 text-sm text-sky-200">
      {selectedText(selected?.point ?? null, selectedRawIndex, xField, yField)}
    </p>
  </div>;
}
