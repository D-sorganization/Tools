/** Dependency-free, scale-locked orthographic 3D ball-flight drawing. */

import type { FlightPoint } from "../model/flight";
import type { GroundTrajectoryPoint } from "../model/flightGroundTypes";
import type { Vec3 } from "../model/simulation";
import {
  spatialTargetHalfExtents,
  type SpatialTargetTs,
} from "../model/spatialTarget";
import { canvasContext } from "./canvasDisplay";

export const FLIGHT_PLAYBACK_LOGICAL_SIZE = { width: 860, height: 420 } as const;

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

interface Bounds {
  min: Vec3;
  max: Vec3;
}

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

function extents(
  points: readonly FlightPoint[],
  comparison: readonly FlightPoint[],
  target?: SpatialTargetTs,
  includeOrigin = true,
): Bounds {
  const positions = [...points, ...comparison].map((point) => point.position);
  if (includeOrigin) positions.push([0, 0, 0]);
  if (target) {
    const center = target.point.appCoordinatesM;
    const half = spatialTargetHalfExtents(target);
    positions.push(
      center.map((value, axis) => value - half[axis]) as Vec3,
      center.map((value, axis) => value + half[axis]) as Vec3,
    );
  }
  return {
    min: [0, 1, 2].map((axis) => Math.min(...positions.map((value) => value[axis]))) as Vec3,
    max: [0, 1, 2].map((axis) => Math.max(...positions.map((value) => value[axis]))) as Vec3,
  };
}

function createProjection(
  points: readonly FlightPoint[],
  comparison: readonly FlightPoint[],
  camera: PlaybackCamera,
  width: number,
  height: number,
  target?: SpatialTargetTs,
  minimumExtentM = MIN_EXTENT_M,
  includeOrigin = true,
): Projection {
  const bounds = extents(points, comparison, target, includeOrigin);
  const center = bounds.min.map((value, axis) =>
    (value + bounds.max[axis]) / 2,
  ) as Vec3;
  const corners: Vec3[] = [];
  for (const carry of [bounds.min[0], bounds.max[0]]) {
    for (const up of [bounds.min[1], bounds.max[1]]) {
      for (const right of [bounds.min[2], bounds.max[2]]) corners.push([carry, up, right]);
    }
  }
  const rotated = corners.map((position) => rotate(position, camera, center));
  const minX = Math.min(...rotated.map((point) => point.x));
  const maxX = Math.max(...rotated.map((point) => point.x));
  const minY = Math.min(...rotated.map((point) => point.y));
  const maxY = Math.max(...rotated.map((point) => point.y));
  const scaleX = (width - 2 * PADDING_PX) / Math.max(maxX - minX, minimumExtentM);
  const scaleY = (height - 2 * PADDING_PX) / Math.max(maxY - minY, minimumExtentM);
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

function drawPolyline(
  context: CanvasRenderingContext2D,
  projection: Projection,
  positions: readonly Vec3[],
  close = false,
): void {
  context.beginPath();
  positions.forEach((position, index) => {
    const point = projection.point(position);
    if (index === 0) context.moveTo(point.x, point.y);
    else context.lineTo(point.x, point.y);
  });
  if (close) context.closePath();
  context.stroke();
}

function circlePoints(center: Vec3, radius: number, plane: "xy" | "xz" | "yz"): Vec3[] {
  return Array.from({ length: 33 }, (_, index) => {
    const angle = index / 32 * 2 * Math.PI;
    const cosine = radius * Math.cos(angle);
    const sine = radius * Math.sin(angle);
    if (plane === "xy") return [center[0] + cosine, center[1] + sine, center[2]];
    if (plane === "xz") return [center[0] + cosine, center[1], center[2] + sine];
    return [center[0], center[1] + cosine, center[2] + sine];
  });
}

function boxCorners(center: Vec3, half: Vec3): Vec3[] {
  const corners: Vec3[] = [];
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
    corners.push([
      center[0] + x * half[0], center[1] + y * half[1], center[2] + z * half[2],
    ]);
  }
  return corners;
}

function drawBox(
  context: CanvasRenderingContext2D,
  projection: Projection,
  center: Vec3,
  half: Vec3,
): void {
  const corners = boxCorners(center, half);
  const edges = [
    [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
    [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
  ];
  edges.forEach(([left, right]) => drawPolyline(
    context, projection, [corners[left], corners[right]],
  ));
}

function drawSpatialTarget(
  context: CanvasRenderingContext2D,
  projection: Projection,
  target: SpatialTargetTs,
): void {
  const center: Vec3 = [...target.point.appCoordinatesM];
  const tolerance = target.tolerance;
  context.strokeStyle = "#f59e0b";
  context.fillStyle = "#fbbf24";
  context.lineWidth = 2;
  context.setLineDash([5, 3]);
  if (tolerance.kind === "sphere") {
    (["xy", "xz", "yz"] as const).forEach((plane) =>
      drawPolyline(context, projection, circlePoints(center, tolerance.radiusM, plane), true));
  } else if (tolerance.kind === "box") {
    drawBox(context, projection, center, [...tolerance.halfExtentsM]);
  } else if (tolerance.kind === "surface_circle") {
    drawPolyline(context, projection, circlePoints(center, tolerance.radiusM, "xz"), true);
  } else {
    drawBox(context, projection, center, [tolerance.halfLengthM, 0, tolerance.halfWidthM]);
  }
  context.setLineDash([]);
  const label = projection.point(center);
  context.font = "bold 12px sans-serif";
  context.fillText(`ACTIVE TARGET · ${target.label}`, label.x + 7, label.y - 8);
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
  origin: Vec3 = [0, 0, 0],
): void {
  const axes: Array<[Vec3, string, string]> = [
    [[origin[0] + carryM, origin[1], origin[2]], "#38bdf8", "x target [m]"],
    [[origin[0], origin[1] + heightM, origin[2]], "#f59e0b", "y up [m]"],
    [[origin[0], origin[1], origin[2] + lateralM], "#a78bfa", "z right [m]"],
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

const GROUND_PHASE_COLORS = {
  impact: "#ef476f",
  bounce: "#a78bfa",
  skid: "#f59e0b",
  roll: "#34d399",
  rest: "#60a5fa",
} as const;
const GROUND_MINIMUM_EXTENT_M = 0.05;

/** Draw a strict phase-colored ground result with first-contact and end markers. */
export function drawGroundPlayback(
  canvas: HTMLCanvasElement,
  points: readonly GroundTrajectoryPoint[],
  ballPosition: Vec3,
  camera: PlaybackCamera,
  endLabel: string,
): void {
  const drawable = points.map((point) => ({
    time: point.time_s,
    position: point.position_m as Vec3,
    velocity: point.velocity_m_s as Vec3,
  }));
  const context = canvasContext(canvas, FLIGHT_PLAYBACK_LOGICAL_SIZE);
  if (!context || drawable.length === 0) return;
  const { width, height } = FLIGHT_PLAYBACK_LOGICAL_SIZE;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#020617";
  context.fillRect(0, 0, width, height);
  const projection = createProjection(
    drawable,
    [],
    camera,
    width,
    height,
    undefined,
    GROUND_MINIMUM_EXTENT_M,
    false,
  );
  const bounds = extents(drawable, [], undefined, false);
  const axisExtents = bounds.max.map((value, axis) =>
    Math.max(GROUND_MINIMUM_EXTENT_M, Math.abs(value - bounds.min[axis])),
  ) as Vec3;
  drawGroundAxes(
    context,
    projection,
    axisExtents[0],
    axisExtents[1],
    axisExtents[2],
    points[0].position_m as Vec3,
  );
  points.slice(0, -1).forEach((point, index) => {
    drawPath(
      context,
      drawable.slice(index, index + 2),
      projection,
      GROUND_PHASE_COLORS[point.phase],
    );
  });
  const marker = (
    position: Vec3,
    color: string,
    label: string | null,
    offset: number,
  ) => {
    const screen = projection.point(position);
    context.beginPath();
    context.arc(screen.x, screen.y, 6, 0, 2 * Math.PI);
    context.fillStyle = color;
    context.fill();
    if (label) {
      context.fillStyle = "#e2e8f0";
      context.fillText(label, screen.x + 8, screen.y + offset);
    }
  };
  marker(points[0].position_m as Vec3, "#38bdf8", "Carry / first contact", -8);
  marker(points[points.length - 1].position_m as Vec3, "#f8fafc", endLabel, 14);
  marker(ballPosition, "#fb923c", null, -8);
  context.fillStyle = "#cbd5e1";
  context.fillText(
    `Locked physical scale: ${projection.pixelsPerMeter.toFixed(2)} px/m`,
    10,
    height - 12,
  );
}

/** Draw a rotatable orthographic view with one identical pixel scale per metre. */
export function drawFlightPlayback(
  canvas: HTMLCanvasElement,
  points: readonly FlightPoint[],
  comparison: readonly FlightPoint[],
  ballPosition: Vec3 | null,
  camera: PlaybackCamera,
  spatialTarget?: SpatialTargetTs,
): void {
  const context = canvasContext(canvas, FLIGHT_PLAYBACK_LOGICAL_SIZE);
  if (!context || (points.length === 0 && !spatialTarget)) return;
  const { width, height } = FLIGHT_PLAYBACK_LOGICAL_SIZE;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#020617";
  context.fillRect(0, 0, width, height);
  const projection = createProjection(points, comparison, camera, width, height, spatialTarget);
  const bounds = extents(points, comparison, spatialTarget);
  const axisExtents = bounds.max.map((value, axis) =>
    Math.max(MIN_EXTENT_M, Math.abs(value), Math.abs(bounds.min[axis])),
  ) as Vec3;
  drawGroundAxes(context, projection, axisExtents[0], axisExtents[1], axisExtents[2]);
  drawPath(context, comparison, projection, "#60a5fa", [7, 5]);
  drawPath(context, points, projection, "#34d399");
  if (spatialTarget) drawSpatialTarget(context, projection, spatialTarget);
  if (ballPosition) {
    const ball = projection.point(ballPosition);
    context.beginPath();
    context.arc(ball.x, ball.y, 6, 0, 2 * Math.PI);
    context.fillStyle = "#fb923c";
    context.fill();
    context.strokeStyle = "#f8fafc";
    context.lineWidth = 1.5;
    context.stroke();
  }
  context.fillStyle = "#cbd5e1";
  context.font = "12px sans-serif";
  context.fillText(
    `Locked physical scale: ${projection.pixelsPerMeter.toFixed(2)} px/m`,
    10,
    height - 12,
  );
}
