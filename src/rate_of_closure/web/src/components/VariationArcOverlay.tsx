import { useEffect, useMemo, useRef, useState } from "react";

import type { Vec3 } from "../model/simulation";
import type {
  SwingTrialStatusTs,
  SwingVariationResultTs,
} from "../model/variationSwingEnsemble";
import { BUTTON_CLASS, INPUT_CLASS } from "./variationUi";

interface VariationArcOverlayProps {
  ensemble: SwingVariationResultTs;
}

type PointKind = "pivot" | "wrist" | "clubhead";
interface CameraState { yaw: number; pitch: number; zoom: number }

const INITIAL_CAMERA: CameraState = { yaw: -0.65, pitch: 0.38, zoom: 1 };
const MAX_VERTICES = 200_000;

export function VariationArcOverlay({ ensemble }: VariationArcOverlayProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [pointKind, setPointKind] = useState<PointKind>("clubhead");
  const [camera, setCamera] = useState<CameraState>(INITIAL_CAMERA);
  const traces = useMemo(
    () => traceRows(ensemble, pointKind),
    [ensemble, pointKind],
  );
  const validCount = traces.length;
  const rawVertices = traces.reduce((total, trace) => total + trace.points.length, 0);
  const stride = Math.max(1, Math.ceil(rawVertices / MAX_VERTICES));

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
    drawScene(context, width, height, traces, camera, stride);
  }, [camera, stride, traces]);

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
            onChange={(event) => setPointKind(event.target.value as PointKind)}
          >
            <option value="clubhead">Clubhead Reference</option>
            <option value="wrist">Wrist</option>
            <option value="pivot">Pivot</option>
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
        {validCount}/{ensemble.runs.length} trials shown · {Math.ceil(rawVertices / stride).toLocaleString()}/{rawVertices.toLocaleString()} vertices · Frame: {ensemble.coordinateFrame}. Drag to rotate; scroll or use +/− to zoom.
      </p>
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
    </div>
  );
}

interface TraceRow {
  points: Vec3[];
  status: SwingTrialStatusTs;
}

function traceRows(
  ensemble: SwingVariationResultTs,
  pointKind: PointKind,
): TraceRow[] {
  return ensemble.runs.flatMap((trial) => {
    if (trial.run === null) return [];
    const points = trial.run.swing.map((sample) => {
      if (pointKind === "clubhead") return sample.position;
      const index = pointKind === "pivot" ? 0 : Math.max(sample.joints.length - 2, 0);
      return sample.joints[index] ?? sample.position;
    });
    return [{ points, status: trial.status }];
  });
}

function drawScene(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  traces: TraceRow[],
  camera: CameraState,
  stride: number,
): void {
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#07101f";
  context.fillRect(0, 0, width, height);
  if (traces.length === 0) {
    context.fillStyle = "#94a3b8";
    context.textAlign = "center";
    context.fillText("No evaluated swing traces", width / 2, height / 2);
    return;
  }
  const allPoints = traces.flatMap((trace) => trace.points);
  const center = boundsCenter(allPoints);
  const radius = Math.max(...allPoints.map((point) => distance(point, center)), 1e-6);
  const project = (point: Vec3): [number, number] => {
    const rotated = rotatePoint(point, center, camera);
    const scale = 0.42 * Math.min(width, height) * camera.zoom / radius;
    return [width / 2 + rotated[0] * scale, height / 2 - rotated[1] * scale];
  };
  drawAxes(context, center, radius, project);
  traces.forEach((trace) => {
    context.beginPath();
    context.strokeStyle = trace.status === "evaluated_hit" ? "#38bdf8" : "#f59e0b";
    context.globalAlpha = 0.28;
    context.lineWidth = 1;
    trace.points.forEach((point, index) => {
      if (index % stride !== 0 && index !== trace.points.length - 1) return;
      const [x, y] = project(point);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  });
  const median = medianTrace(traces.map((trace) => trace.points));
  context.beginPath();
  context.strokeStyle = "#f8fafc";
  context.globalAlpha = 0.95;
  context.lineWidth = 2.4;
  median.forEach((point, index) => {
    if (index % stride !== 0 && index !== median.length - 1) return;
    const [x, y] = project(point);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  context.globalAlpha = 1;
}

function rotatePoint(point: Vec3, center: Vec3, camera: CameraState): Vec3 {
  const x = point[0] - center[0];
  const y = point[1] - center[1];
  const z = point[2] - center[2];
  const cy = Math.cos(camera.yaw);
  const sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch);
  const sp = Math.sin(camera.pitch);
  const yawX = cy * x - sy * z;
  const yawZ = sy * x + cy * z;
  return [yawX, cp * y - sp * yawZ, sp * y + cp * yawZ];
}

function boundsCenter(points: Vec3[]): Vec3 {
  const bounds = [0, 1, 2].map((axis) => {
    const values = points.map((point) => point[axis]);
    return (Math.min(...values) + Math.max(...values)) / 2;
  });
  return bounds as Vec3;
}

const distance = (point: Vec3, center: Vec3): number => Math.hypot(
  point[0] - center[0], point[1] - center[1], point[2] - center[2],
);

function medianTrace(traces: Vec3[][]): Vec3[] {
  const count = Math.min(...traces.map((trace) => trace.length));
  return Array.from({ length: count }, (_, sampleIndex) => [0, 1, 2].map((axis) => {
    const values = traces.map((trace) => trace[sampleIndex][axis]).sort((a, b) => a - b);
    const middle = Math.floor(values.length / 2);
    return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
  }) as Vec3);
}

function drawAxes(
  context: CanvasRenderingContext2D,
  center: Vec3,
  radius: number,
  project: (point: Vec3) => [number, number],
): void {
  const axes: Array<{ end: Vec3; color: string; label: string }> = [
    { end: [center[0] + radius * 0.4, center[1], center[2]], color: "#ef6464", label: "x Target" },
    { end: [center[0], center[1] + radius * 0.4, center[2]], color: "#4ade80", label: "y Up" },
    { end: [center[0], center[1], center[2] + radius * 0.4], color: "#60a5fa", label: "z Right" },
  ];
  const origin = project(center);
  context.globalAlpha = 0.85;
  context.font = "12px system-ui";
  axes.forEach((axis) => {
    const end = project(axis.end);
    context.beginPath();
    context.strokeStyle = axis.color;
    context.moveTo(...origin);
    context.lineTo(...end);
    context.stroke();
    context.fillStyle = axis.color;
    context.fillText(axis.label, end[0] + 4, end[1] - 4);
  });
  context.globalAlpha = 1;
}
