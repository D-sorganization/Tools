/** Bounded synchronized path/speed evidence and exact sample inspection. */

import type { KeyboardEvent, PointerEvent } from "react";

import { captureSpeedMps, type PuttResult } from "../model/putting";
import {
  navigatePuttingSamples,
  nearestPuttingSample,
  type PuttingDisplaySample,
  type PuttingSamplePlan,
} from "../model/puttingSampleInspector";

interface PuttingVisualsProps {
  readonly result: PuttResult | null;
  readonly plan: PuttingSamplePlan | null;
  readonly selectedRawIndex: number | null;
  readonly onSelectionChange: (rawIndex: number | null) => void;
  readonly holeX: number;
  readonly grade: number;
  readonly aspect: number;
}

interface PlotPoint {
  readonly rawIndex: number;
  readonly x: number;
  readonly y: number;
}

const PATH_WIDTH = 640;
const PATH_HEIGHT = 320;
const SPEED_WIDTH = 640;
const SPEED_HEIGHT = 180;

function renderedPoints(svg: SVGSVGElement, points: readonly PlotPoint[]): PlotPoint[] {
  const matrix = svg.getScreenCTM();
  if (!matrix) throw new Error("putting visual has no rendered screen transform");
  return points.map(({ rawIndex, x, y }) => ({
    rawIndex,
    x: matrix.a * x + matrix.c * y + matrix.e,
    y: matrix.b * x + matrix.d * y + matrix.f,
  }));
}

function selectPointer(
  points: readonly PlotPoint[], event: PointerEvent<SVGSVGElement>,
  onSelectionChange: (rawIndex: number | null) => void,
) {
  onSelectionChange(nearestPuttingSample(
    renderedPoints(event.currentTarget, points)
      .map(({ rawIndex, x, y }) => [rawIndex, x, y]),
    [event.clientX, event.clientY],
  ));
}

function selectionKeys(
  event: KeyboardEvent<SVGSVGElement>, plan: PuttingSamplePlan,
  selectedRawIndex: number | null, onSelectionChange: (rawIndex: number | null) => void,
) {
  const commands = {
    ArrowLeft: "previous", ArrowRight: "next", Home: "home", End: "end", Escape: "clear",
  } as const;
  const command = commands[event.key as keyof typeof commands];
  if (!command) return;
  event.preventDefault();
  onSelectionChange(navigatePuttingSamples(plan, selectedRawIndex, command));
}

function pointString(points: readonly PlotPoint[], rawIndices: readonly number[]): string {
  const included = new Set(rawIndices);
  let result = "";
  let first = true;
  for (let i = 0; i < points.length; i++) {
    const pt = points[i];
    if (included.has(pt.rawIndex)) {
      if (!first) result += " ";
      result += `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
      first = false;
    }
  }
  return result;
}

function selectedPoint(points: readonly PlotPoint[], rawIndex: number | null): PlotPoint | null {
  return points.find((point) => point.rawIndex === rawIndex) ?? null;
}

function GreenView(props: PuttingVisualsProps) {
  const { result, plan } = props;
  if (!result || !plan) return <p className="text-sm text-slate-400">Inputs out of range.</p>;
  const samples = plan.samples;
  const maxX = Math.max(props.holeX + 0.5, ...samples.map(({ xM }) => xM)) + 0.3;
  const minX = Math.min(0, ...samples.map(({ xM }) => xM)) - 0.3;
  const spanY = Math.max(0.8, 2 * Math.max(...samples.map(({ yM }) => Math.abs(yM)), 0.3));
  const scaleX = (value: number) => ((value - minX) / (maxX - minX)) * PATH_WIDTH;
  const scaleY = (value: number) => PATH_HEIGHT / 2 - (value / spanY) * PATH_HEIGHT;
  const points = samples.map(({ rawIndex, xM, yM }) => ({
    rawIndex, x: scaleX(xM), y: scaleY(yM),
  }));
  const selected = selectedPoint(points, props.selectedRawIndex);
  const arrowX = scaleX(props.holeX * 0.5); const arrowY = scaleY(0);
  return <figure aria-label="Top-down green view: skid phase orange, pure roll green, hole circle, downhill arrow"
    className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
    <p id="putting-sample-instructions" className="mb-2 text-xs text-slate-400">
      Select a displayed sample with pointer. Left/Right moves, Home/End jumps, Escape clears.
    </p>
    <svg viewBox={`0 0 ${PATH_WIDTH} ${PATH_HEIGHT}`} role="img" tabIndex={0}
      aria-label="Interactive putt path sample inspector" aria-describedby="putting-sample-instructions"
      className="min-h-[180px] w-full rounded-lg bg-emerald-950/60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400 sm:min-h-0"
      onKeyDown={(event) => selectionKeys(event, plan, props.selectedRawIndex, props.onSelectionChange)}
      onPointerDown={(event) => selectPointer(points, event, props.onSelectionChange)}>
      {plan.skidPolylineIndices.length > 0 && <polyline
        points={pointString(points, plan.skidPolylineIndices)} fill="none" stroke="#fb923c" strokeWidth={3} />}
      <polyline points={pointString(points, plan.pureRollPolylineIndices)}
        fill="none" stroke="#4ade80" strokeWidth={3} />
      <circle cx={scaleX(props.holeX)} cy={scaleY(0)}
        r={Math.max(5, (0.054 / (maxX - minX)) * PATH_WIDTH)}
        fill="none" stroke="#f8fafc" strokeWidth={2} />
      <circle cx={scaleX(0)} cy={scaleY(0)} r={4} fill="#f8fafc" />
      {props.grade > 0 && <g stroke="#94a3b8" strokeWidth={2}>
        <line x1={arrowX} y1={arrowY}
          x2={arrowX + 40 * Math.cos((props.aspect * Math.PI) / 180)}
          y2={arrowY - 40 * Math.sin((props.aspect * Math.PI) / 180)}
          markerEnd="url(#downhill-arrow)" />
        <defs><marker id="downhill-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3"
          orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#94a3b8" /></marker></defs>
      </g>}
      {result.holed && <text x={scaleX(props.holeX)} y={scaleY(0) - 14} textAnchor="middle"
        fill="#4ade80" fontSize="13">HOLED</text>}
      {selected && <circle data-testid="putting-selected-sample" cx={selected.x} cy={selected.y}
        r={8} fill="none" stroke="#facc15" strokeWidth={3} />}
    </svg>
    <figcaption className="mt-2 text-xs text-slate-400">Orange = skid; green = pure roll;
      the circle is the hole. Left is the putt&apos;s left (+y).</figcaption>
  </figure>;
}

function SpeedPlot(props: PuttingVisualsProps) {
  const { result, plan } = props;
  if (!result || !plan) return null;
  const maxDistance = Math.max(plan.cumulativeDistanceM[plan.rawCount - 1], 0.1);
  const maxSpeed = Math.max(...plan.samples.map(({ speedMps }) => speedMps), captureSpeedMps()) * 1.08;
  const scaleX = (value: number) => (value / maxDistance) * (SPEED_WIDTH - 20) + 10;
  const scaleY = (value: number) => SPEED_HEIGHT - 16 - (value / maxSpeed) * (SPEED_HEIGHT - 32);
  const points = plan.samples.map(({ rawIndex, cumulativeDistanceM, speedMps }) => ({
    rawIndex, x: scaleX(cumulativeDistanceM), y: scaleY(speedMps),
  }));
  const selected = selectedPoint(points, props.selectedRawIndex);
  const split = plan.cumulativeDistanceM[plan.skidEndIndex];
  return <figure aria-label="Ball speed versus distance with the capture-speed bound and the skid-to-roll transition marked"
    className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
    <svg viewBox={`0 0 ${SPEED_WIDTH} ${SPEED_HEIGHT}`} role="img"
      aria-label="Interactive synchronized speed versus distance plot"
      className="w-full rounded-lg bg-slate-950/60"
      onPointerDown={(event) => selectPointer(points, event, props.onSelectionChange)}>
      <polyline points={pointString(points, plan.displayedRawIndices)}
        fill="none" stroke="#38bdf8" strokeWidth={2.5} />
      <line x1={10} x2={SPEED_WIDTH - 10} y1={scaleY(captureSpeedMps())} y2={scaleY(captureSpeedMps())}
        stroke="#f87171" strokeWidth={1.5} strokeDasharray="6 4" />
      <line x1={scaleX(split)} x2={scaleX(split)} y1={12} y2={SPEED_HEIGHT - 16}
        stroke="#fb923c" strokeWidth={1.5} strokeDasharray="3 4" />
      {selected && <circle data-testid="putting-selected-sample" cx={selected.x} cy={selected.y}
        r={7} fill="none" stroke="#facc15" strokeWidth={3} />}
    </svg>
    <figcaption className="mt-2 text-xs text-slate-400">Blue: speed vs distance. Red:
      capture-speed bound. Orange: first pure-roll sample.</figcaption>
  </figure>;
}

function sampleStatus(sample: PuttingDisplaySample | null): string {
  if (!sample) return "No trajectory sample selected.";
  const phase = sample.phase === "pure-roll" ? "pure roll" : "skid";
  return `Source sample ${sample.rawIndex} (zero-based); t ${sample.timeS.toFixed(3)} s; ` +
    `distance ${sample.cumulativeDistanceM.toFixed(3)} m; x ${sample.xM.toFixed(3)} m; ` +
    `y ${sample.yM.toFixed(3)} m; speed ${sample.speedMps.toFixed(3)} m/s; ${phase}.`;
}

export function PuttingVisuals(props: PuttingVisualsProps) {
  const sample = props.plan !== null && props.selectedRawIndex !== null
    ? props.plan.rawSample(props.selectedRawIndex) : null;
  return <>
    <p role="status" aria-live="polite" className="rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-200">
      {sampleStatus(sample)}
    </p>
    <GreenView {...props} />
    <SpeedPlot {...props} />
  </>;
}
