import type { Vec3 } from "../model/simulation";
import type { ConfidenceEllipsoidMeshTs } from "../model/confidenceEllipsoidMesh";
import type {
  GeometricVariabilityTs,
  SwingTraceRowTs,
} from "../model/variationGeometry";

export interface VariationCameraState { yaw: number; pitch: number; zoom: number }

export function drawVariationArcScene(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  traces: SwingTraceRowTs[],
  variability: GeometricVariabilityTs,
  camera: VariationCameraState,
  stride: number,
  selectedTrialIndex: number | null,
  ellipsoidMesh: ConfidenceEllipsoidMeshTs | null,
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
  const renderStride = Math.max(1, Math.floor(stride));
  const bounds = sceneBounds(traces, ellipsoidMesh);
  const center = boundsCenter(bounds);
  const radius = boundsRadius(bounds);
  const project = (point: Vec3): [number, number] => {
    const rotated = rotatePoint(point, center, camera);
    const scale = 0.42 * Math.min(width, height) * camera.zoom / radius;
    return [width / 2 + rotated[0] * scale, height / 2 - rotated[1] * scale];
  };
  drawAxes(context, center, radius, project);
  if (ellipsoidMesh !== null) {
    drawConfidenceEllipsoids(
      context,
      ellipsoidMesh,
      project,
      (point) => rotatePoint(point, center, camera)[2],
    );
  }
  traces.forEach((trace) => {
    const selected = trace.trialIndex === selectedTrialIndex;
    context.beginPath();
    context.strokeStyle = trace.status === "evaluated_hit" ? "#38bdf8" : "#f59e0b";
    context.globalAlpha = selectedTrialIndex === null ? 0.28 : selected ? 1 : 0.1;
    context.lineWidth = selected ? 3 : 1;
    trace.points.forEach((point, index) => {
      if (!isRenderedSample(index, trace.points.length, renderStride)) return;
      const [x, y] = project(point);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  });
  drawPrincipalSpread(context, variability, project);
  const median = medianTrace(traces.map((trace) => trace.points));
  context.beginPath();
  context.strokeStyle = "#f8fafc";
  context.globalAlpha = 0.95;
  context.lineWidth = 2.4;
  median.forEach((point, index) => {
    if (!isRenderedSample(index, median.length, renderStride)) return;
    const [x, y] = project(point);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  context.strokeStyle = "#34d399";
  context.lineWidth = 3.4;
  variability.quietIntervals.forEach(({ startIndex, endIndex }) => {
    context.beginPath();
    let started = false;
    for (let index = startIndex; index <= endIndex && index < median.length; index += 1) {
      if (index % renderStride !== 0 && index !== endIndex) continue;
      const [x, y] = project(median[index]);
      if (!started) context.moveTo(x, y); else context.lineTo(x, y);
      started = true;
    }
    context.stroke();
  });
  context.globalAlpha = 1;
}

function drawConfidenceEllipsoids(
  context: CanvasRenderingContext2D,
  mesh: ConfidenceEllipsoidMeshTs,
  project: (point: Vec3) => [number, number],
  depth: (point: Vec3) => number,
): void {
  const triangles = mesh.triangles.map((indices) => {
    const points = indices.map((index) => mesh.verticesM[index]) as [Vec3, Vec3, Vec3];
    return { points, depth: points.reduce((sum, point) => sum + depth(point), 0) / 3 };
  }).sort((left, right) => left.depth - right.depth);
  context.fillStyle = "#22d3ee";
  context.strokeStyle = "#67e8f9";
  context.lineWidth = 0.35;
  context.globalAlpha = 0.16;
  triangles.forEach(({ points }) => {
    context.beginPath();
    context.moveTo(...project(points[0]));
    context.lineTo(...project(points[1]));
    context.lineTo(...project(points[2]));
    context.closePath();
    context.fill();
    context.stroke();
  });
  context.globalAlpha = 1;
}

function drawPrincipalSpread(
  context: CanvasRenderingContext2D,
  data: GeometricVariabilityTs,
  project: (point: Vec3) => [number, number],
): void {
  const stride = Math.max(1, Math.floor(data.sampleTimesS.length / 14));
  context.strokeStyle = "#fbbf24";
  context.globalAlpha = 0.8;
  context.lineWidth = 1.1;
  for (let index = 0; index < data.sampleTimesS.length; index += stride) {
    const mean = data.meanPositionsM[index];
    const axis = data.principalAxes[index];
    const extent = 2 * data.principalSigmaM[index];
    const low = mean.map((value, component) => value - extent * axis[component]) as Vec3;
    const high = mean.map((value, component) => value + extent * axis[component]) as Vec3;
    context.beginPath();
    context.moveTo(...project(low));
    context.lineTo(...project(high));
    context.stroke();
  }
}

function rotatePoint(point: Vec3, center: Vec3, camera: VariationCameraState): Vec3 {
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

interface SceneBounds { min: Vec3; max: Vec3; count: number }

function sceneBounds(
  traces: SwingTraceRowTs[],
  mesh: ConfidenceEllipsoidMeshTs | null,
): SceneBounds {
  const bounds: SceneBounds = {
    min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
    max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
    count: 0,
  };
  traces.forEach((trace) => {
    trace.points.forEach((point) => includePoint(bounds, point));
  });
  mesh?.verticesM.forEach((point) => includePoint(bounds, point));
  return bounds;
}

function includePoint(bounds: SceneBounds, point: Vec3): void {
  for (let axis = 0; axis < 3; axis += 1) {
    bounds.min[axis] = Math.min(bounds.min[axis], point[axis]);
    bounds.max[axis] = Math.max(bounds.max[axis], point[axis]);
  }
  bounds.count += 1;
}

function boundsCenter(bounds: SceneBounds): Vec3 {
  if (bounds.count === 0) return [0, 0, 0];
  return [0, 1, 2].map((axis) => (
    bounds.min[axis] / 2 + bounds.max[axis] / 2
  )) as Vec3;
}

function boundsRadius(bounds: SceneBounds): number {
  if (bounds.count === 0) return 1e-6;
  return Math.max(Math.hypot(
    bounds.max[0] / 2 - bounds.min[0] / 2,
    bounds.max[1] / 2 - bounds.min[1] / 2,
    bounds.max[2] / 2 - bounds.min[2] / 2,
  ), 1e-6);
}

const isRenderedSample = (index: number, count: number, stride: number): boolean => (
  index % stride === 0 || index === count - 1
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
