/** Dependency-free, scale-locked orthographic 3D ball-flight drawing. */

import type { FlightPoint } from "../model/flight";
import type { Vec3 } from "../model/simulation";

export interface PlaybackCamera {
  yawRad: number;
  pitchRad: number;
  zoom: number;
}

interface ProjectedPoint {
  x: number;
  y: number;
  depth: number;
}

interface Projection {
  point: (position: Vec3) => ProjectedPoint;
  pixelsPerMeter: number;
}

const PADDING_PX = 34;
const MIN_EXTENT_M = 1;

function rotate(position: Vec3, camera: PlaybackCamera, center: Vec3): ProjectedPoint {
  const forward = position[0] - center[0];
  const up = position[1] - center[1];
  const right = position[2] - center[2];
  const cosineYaw = Math.cos(camera.yawRad);
  const sineYaw = Math.sin(camera.yawRad);
  const horizontal = cosineYaw * right - sineYaw * forward;
  const depth = sineYaw * right + cosineYaw * forward;
  const cosinePitch = Math.cos(camera.pitchRad);
  const sinePitch = Math.sin(camera.pitchRad);
  return {
    x: horizontal,
    y: -(cosinePitch * up - sinePitch * depth),
    depth: sinePitch * up + cosinePitch * depth,
  };
}

function extents(points: readonly FlightPoint[], comparison: readonly FlightPoint[]) {
  const positions = [...points, ...comparison].map((point) => point.position);
  const carry = Math.max(MIN_EXTENT_M, ...positions.map((position) => position[0]));
  const height = Math.max(MIN_EXTENT_M, ...positions.map((position) => position[1]));
  const lateral = Math.max(MIN_EXTENT_M, ...positions.map((position) => Math.abs(position[2])));
  return { carry, height, lateral };
}

function createProjection(
  points: readonly FlightPoint[],
  comparison: readonly FlightPoint[],
  camera: PlaybackCamera,
  width: number,
  height: number,
): Projection {
  const bounds = extents(points, comparison);
  const center: Vec3 = [bounds.carry / 2, bounds.height / 2, 0];
  const corners: Vec3[] = [];
  for (const carry of [0, bounds.carry]) {
    for (const up of [0, bounds.height]) {
      for (const right of [-bounds.lateral, bounds.lateral]) corners.push([carry, up, right]);
    }
  }
  const rotated = corners.map((position) => rotate(position, camera, center));
  const minX = Math.min(...rotated.map((point) => point.x));
  const maxX = Math.max(...rotated.map((point) => point.x));
  const minY = Math.min(...rotated.map((point) => point.y));
  const maxY = Math.max(...rotated.map((point) => point.y));
  const scaleX = (width - 2 * PADDING_PX) / Math.max(maxX - minX, MIN_EXTENT_M);
  const scaleY = (height - 2 * PADDING_PX) / Math.max(maxY - minY, MIN_EXTENT_M);
  const pixelsPerMeter = Math.min(scaleX, scaleY) * camera.zoom;
  const centerX = width / 2 - ((minX + maxX) / 2) * pixelsPerMeter;
  const centerY = height / 2 - ((minY + maxY) / 2) * pixelsPerMeter;
  return {
    pixelsPerMeter,
    point: (position) => {
      const projected = rotate(position, camera, center);
      return {
        x: centerX + projected.x * pixelsPerMeter,
        y: centerY + projected.y * pixelsPerMeter,
        depth: projected.depth,
      };
    },
  };
}

function drawPath(
  context: CanvasRenderingContext2D,
  points: readonly FlightPoint[],
  projection: Projection,
  color: string,
  dash: number[] = [],
): void {
  if (points.length < 2) return;
  context.beginPath();
  context.setLineDash(dash);
  points.forEach((point, index) => {
    const screen = projection.point(point.position);
    if (index === 0) context.moveTo(screen.x, screen.y);
    else context.lineTo(screen.x, screen.y);
  });
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.stroke();
  context.setLineDash([]);
}

function drawGroundAxes(
  context: CanvasRenderingContext2D,
  projection: Projection,
  carryM: number,
  heightM: number,
  lateralM: number,
): void {
  const origin: Vec3 = [0, 0, 0];
  const axes: Array<[Vec3, string, string]> = [
    [[carryM, 0, 0], "#38bdf8", "x target [m]"],
    [[0, heightM, 0], "#f59e0b", "y up [m]"],
    [[0, 0, lateralM], "#a78bfa", "z right [m]"],
  ];
  const start = projection.point(origin);
  axes.forEach(([endPosition, color, label]) => {
    const end = projection.point(endPosition);
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.strokeStyle = color;
    context.lineWidth = 1.2;
    context.stroke();
    context.fillStyle = color;
    context.fillText(label, end.x + 4, end.y - 4);
  });
}

/** Draw a rotatable orthographic view with one identical pixel scale per metre. */
export function drawFlightPlayback(
  canvas: HTMLCanvasElement,
  points: readonly FlightPoint[],
  comparison: readonly FlightPoint[],
  ballPosition: Vec3,
  camera: PlaybackCamera,
): void {
  const context = canvas.getContext("2d");
  if (!context || points.length === 0) return;
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#020617";
  context.fillRect(0, 0, width, height);
  const projection = createProjection(points, comparison, camera, width, height);
  const bounds = extents(points, comparison);
  drawGroundAxes(context, projection, bounds.carry, bounds.height, bounds.lateral);
  drawPath(context, comparison, projection, "#60a5fa", [7, 5]);
  drawPath(context, points, projection, "#34d399");
  const ball = projection.point(ballPosition);
  context.beginPath();
  context.arc(ball.x, ball.y, 6, 0, 2 * Math.PI);
  context.fillStyle = "#fb923c";
  context.fill();
  context.strokeStyle = "#f8fafc";
  context.lineWidth = 1.5;
  context.stroke();
  context.fillStyle = "#cbd5e1";
  context.font = "12px sans-serif";
  context.fillText(
    `Locked physical scale: ${projection.pixelsPerMeter.toFixed(2)} px/m`,
    10,
    height - 12,
  );
}
