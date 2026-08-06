import type { Vec3 } from "../model/simulation";
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
    const selected = trace.trialIndex === selectedTrialIndex;
    context.beginPath();
    context.strokeStyle = trace.status === "evaluated_hit" ? "#38bdf8" : "#f59e0b";
    context.globalAlpha = selectedTrialIndex === null ? 0.28 : selected ? 1 : 0.1;
    context.lineWidth = selected ? 3 : 1;
    trace.points.forEach((point, index) => {
      if (index % stride !== 0 && index !== trace.points.length - 1) return;
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
    if (index % stride !== 0 && index !== median.length - 1) return;
    const [x, y] = project(point);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  context.strokeStyle = "#34d399";
  context.lineWidth = 3.4;
  variability.quietIntervals.forEach(({ startIndex, endIndex }) => {
    context.beginPath();
    median.slice(startIndex, endIndex + 1).forEach((point, offset) => {
      const [x, y] = project(point);
      if (offset === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
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

function boundsCenter(points: Vec3[]): Vec3 {
  return [0, 1, 2].map((axis) => {
    const values = points.map((point) => point[axis]);
    return (Math.min(...values) + Math.max(...values)) / 2;
  }) as Vec3;
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
