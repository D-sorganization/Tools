import { useEffect, useMemo, useRef, useState } from "react";

import type { SwingVariationResultTs } from "../model/variationSwingEnsemble";
import type { SwingTrialStatusTs } from "../model/variationSwingEnsemble";
import {
  geometricVariability,
  swingTraceRows,
  type SwingPointKindTs,
} from "../model/variationGeometry";
import {
  makeVariationPlotDefinition,
  variationPlotDefinitionToJson,
  variationResultFingerprint,
} from "../model/variationPlotDefinition";
import { VariationVariabilityTimeline } from "./VariationVariabilityTimeline";
import {
  drawVariationArcScene,
  type VariationCameraState,
} from "./VariationArcDrawing";
import {
  BUTTON_CLASS,
  downloadBlob,
  downloadSvgElement,
  downloadText,
  INPUT_CLASS,
} from "./variationUi";

interface VariationArcOverlayProps {
  ensemble: SwingVariationResultTs;
  selectedTrialIndex: number | null;
  onSelectedTrialChange: (trialIndex: number | null) => void;
}

const INITIAL_CAMERA: VariationCameraState = { yaw: -0.65, pitch: 0.38, zoom: 1 };
const MAX_VERTICES = 200_000;

export function VariationArcOverlay({
  ensemble,
  selectedTrialIndex,
  onSelectedTrialChange,
}: VariationArcOverlayProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const variabilitySvgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [pointKind, setPointKind] = useState<SwingPointKindTs>("clubhead");
  const [quietThresholdMm, setQuietThresholdMm] = useState(5);
  const [outcomeFilter, setOutcomeFilter] = useState<"all" | SwingTrialStatusTs>("all");
  const [phaseEndPercent, setPhaseEndPercent] = useState(100);
  const [sourceKey, setSourceKey] = useState("all");
  const [sourceBand, setSourceBand] = useState("all");
  const [camera, setCamera] = useState<VariationCameraState>(INITIAL_CAMERA);
  const traces = useMemo(() => {
    const rows = swingTraceRows(ensemble, pointKind);
    const sourceIndex = ensemble.dataset.inputNames.indexOf(sourceKey);
    const ranked = sourceIndex < 0 ? [] : ensemble.dataset.inputs
      .map((row, trialIndex) => ({ trialIndex, value: row[sourceIndex] }))
      .filter((item) => Number.isFinite(item.value))
      .sort((a, b) => a.value - b.value || a.trialIndex - b.trialIndex);
    const bandByTrial = new Map(ranked.map((item, rank) => [
      item.trialIndex,
      rank < ranked.length / 3 ? "lower" : rank < 2 * ranked.length / 3 ? "middle" : "upper",
    ]));
    return rows.filter((row) => {
      if (outcomeFilter !== "all" && row.status !== outcomeFilter) return false;
      if (sourceIndex < 0 || sourceBand === "all") return true;
      return bandByTrial.get(row.trialIndex) === sourceBand;
    }).map((row) => {
      const count = Math.max(2, Math.ceil(row.points.length * phaseEndPercent / 100));
      return { ...row, points: row.points.slice(0, count), timesS: row.timesS.slice(0, count) };
    });
  }, [ensemble, outcomeFilter, phaseEndPercent, pointKind, sourceBand, sourceKey]);
  const variability = useMemo(
    () => geometricVariability(traces, quietThresholdMm / 1000),
    [quietThresholdMm, traces],
  );
  const validCount = traces.length;
  const rawVertices = traces.reduce((total, trace) => total + trace.points.length, 0);
  const stride = Math.max(1, Math.ceil(rawVertices / MAX_VERTICES));
  const resultId = variationResultFingerprint(ensemble);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const width = Math.max(canvas.clientWidth, 640);
    const height = Math.max(Math.round(width * 0.62), 360);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawVariationArcScene(
      context, width, height, traces, variability, camera, stride, selectedTrialIndex,
    );
  }, [camera, selectedTrialIndex, stride, traces, variability]);

  const rotate = (dx: number, dy: number) => setCamera((current) => ({
    ...current,
    yaw: current.yaw + dx * 0.009,
    pitch: Math.max(-1.35, Math.min(1.35, current.pitch + dy * 0.009)),
  }));
  const zoom = (factor: number) => setCamera((current) => ({
    ...current,
    zoom: Math.max(0.35, Math.min(5, current.zoom * factor)),
  }));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-56 flex-1 text-xs text-slate-300">
          <span className="mb-1 block">Modeled Point</span>
          <select
            aria-label="Arc modeled point"
            className={INPUT_CLASS}
            value={pointKind}
            onChange={(event) => setPointKind(event.target.value as SwingPointKindTs)}
          >
            <option value="clubhead">Clubhead Reference</option>
            <option value="wrist">Wrist</option>
            <option value="pivot">Pivot</option>
          </select>
        </label>
        <label className="min-w-48 flex-1 text-xs text-slate-300">
          <span className="mb-1 block">Quiet-Zone RMS Threshold [mm]</span>
          <input
            aria-label="Quiet-zone RMS threshold millimetres"
            className={INPUT_CLASS}
            type="number"
            min="0.01"
            max="1000"
            step="0.1"
            value={quietThresholdMm}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isFinite(value) && value > 0) setQuietThresholdMm(value);
            }}
          />
        </label>
        <label className="min-w-48 flex-1 text-xs text-slate-300">
          <span className="mb-1 block">Outcome Cohort</span>
          <select aria-label="Arc outcome cohort" className={INPUT_CLASS} value={outcomeFilter} onChange={(event) => setOutcomeFilter(event.target.value as "all" | SwingTrialStatusTs)}>
            <option value="all">All Outcomes</option><option value="evaluated_hit">Hits</option><option value="evaluated_no_impact">No Impact</option><option value="numerical_failure">Numerical Failures</option>
          </select>
        </label>
        <label className="min-w-48 flex-1 text-xs text-slate-300">
          <span className="mb-1 block">Perturbation Source</span>
          <select aria-label="Arc perturbation source" className={INPUT_CLASS} value={sourceKey} onChange={(event) => setSourceKey(event.target.value)}>
            <option value="all">All Sources</option>{ensemble.dataset.inputNames.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
        <label className="min-w-40 flex-1 text-xs text-slate-300">
          <span className="mb-1 block">Source Quantile Band</span>
          <select aria-label="Arc perturbation band" className={INPUT_CLASS} value={sourceBand} onChange={(event) => setSourceBand(event.target.value)} disabled={sourceKey === "all"}>
            <option value="all">All Values</option><option value="lower">Lower Third</option><option value="middle">Middle Third</option><option value="upper">Upper Third</option>
          </select>
        </label>
        <label className="min-w-56 flex-1 text-xs text-slate-300">
          <span className="mb-1 block">Displayed Swing Phase: 0–{phaseEndPercent}%</span>
          <input aria-label="Arc phase end percent" className="w-full" type="range" min="5" max="100" step="1" value={phaseEndPercent} onChange={(event) => setPhaseEndPercent(Number(event.target.value))} />
        </label>
        <label className="min-w-48 flex-1 text-xs text-slate-300">
          <span className="mb-1 block">Highlighted Trial (Linked Across Plots)</span>
          <select
            aria-label="Arc highlighted trial"
            className={INPUT_CLASS}
            value={selectedTrialIndex ?? ""}
            onChange={(event) => onSelectedTrialChange(event.target.value === "" ? null : Number(event.target.value))}
          >
            <option value="">All Trials</option>
            {traces.map((trace) => <option key={trace.trialIndex} value={trace.trialIndex}>Trial {trace.trialIndex + 1}</option>)}
          </select>
        </label>
        <button
          type="button"
          className={BUTTON_CLASS}
          onClick={() => setCamera(INITIAL_CAMERA)}
        >
          Reset View
        </button>
      </div>
      <p className="text-xs text-slate-400" aria-live="polite">
        {validCount}/{ensemble.runs.length} trials shown · {Math.ceil(rawVertices / stride).toLocaleString()}/{rawVertices.toLocaleString()} vertices · {variability.quietMask.filter(Boolean).length}/{variability.quietMask.length} quiet samples · Frame: {ensemble.coordinateFrame}; alignment: common simulation time. Drag to rotate; scroll or use +/− to zoom.
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className={BUTTON_CLASS}
          onClick={() => canvasRef.current?.toBlob((blob) => {
            if (blob) downloadBlob(`${resultId}-swing-arcs.png`, blob);
          }, "image/png")}
        >
          Swing Arcs PNG
        </button>
        <button
          type="button"
          className={BUTTON_CLASS}
          onClick={() => variabilitySvgRef.current
            && downloadSvgElement(`${resultId}-geometric-variability.svg`, variabilitySvgRef.current)}
        >
          Variability SVG
        </button>
        <button
          type="button"
          className={BUTTON_CLASS}
          onClick={() => downloadText(
            `${resultId}-swing-arcs.plot.json`,
            variationPlotDefinitionToJson(makeVariationPlotDefinition(ensemble, {
              plotType: "swing_arc_overlay",
              coordinateFrame: ensemble.coordinateFrame,
              xVariableKey: null,
              yVariableKey: null,
              pointId: `swing.${pointKind === "clubhead" ? "clubhead.reference" : pointKind}`,
              positionUnit: "m",
              alignmentBasis: variability.alignmentBasis,
              quietThresholdM: variability.quietThresholdM,
              selectedTrialIndex,
              cameraYawDeg: camera.yaw * 180 / Math.PI,
              cameraPitchDeg: camera.pitch * 180 / Math.PI,
              cameraZoom: camera.zoom,
              outcomeFilter,
              phaseEndFraction: phaseEndPercent / 100,
              perturbationSourceKey: sourceKey === "all" ? null : sourceKey,
              perturbationBand: sourceBand === "all" ? null : sourceBand,
              variableKeys: null,
            })),
            "application/json",
          )}
        >
          Arc Plot Definition JSON
        </button>
      </div>
      <canvas
        ref={canvasRef}
        className="h-auto min-h-96 w-full touch-none rounded-lg border border-slate-800 bg-slate-950/70"
        role="img"
        aria-label="Interactive all-trial swing arcs in the app coordinate frame"
        tabIndex={0}
        onPointerDown={(event) => {
          dragRef.current = { x: event.clientX, y: event.clientY };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!dragRef.current) return;
          rotate(event.clientX - dragRef.current.x, event.clientY - dragRef.current.y);
          dragRef.current = { x: event.clientX, y: event.clientY };
        }}
        onPointerUp={() => { dragRef.current = null; }}
        onPointerCancel={() => { dragRef.current = null; }}
        onWheel={(event) => {
          event.preventDefault();
          zoom(event.deltaY < 0 ? 1.12 : 1 / 1.12);
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") rotate(-8, 0);
          else if (event.key === "ArrowRight") rotate(8, 0);
          else if (event.key === "ArrowUp") rotate(0, -8);
          else if (event.key === "ArrowDown") rotate(0, 8);
          else if (event.key === "+" || event.key === "=") zoom(1.12);
          else if (event.key === "-") zoom(1 / 1.12);
          else return;
          event.preventDefault();
        }}
      />
      <VariationVariabilityTimeline data={variability} svgRef={variabilitySvgRef} />
    </div>
  );
}
