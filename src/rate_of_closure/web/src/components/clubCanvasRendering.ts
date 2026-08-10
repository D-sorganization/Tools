import type { HeadMesh, Triangle } from "../model/mesh";
import { add, apply, project, type Vec3 } from "./clubCanvasGeometry";

const LIGHT_LENGTH = Math.hypot(0.3, 0.8, 0.5);
const LIGHT_DIRECTION: Vec3 = [
  0.3 / LIGHT_LENGTH,
  0.8 / LIGHT_LENGTH,
  0.5 / LIGHT_LENGTH,
];
const MESH_BASE_RGB = [0.56, 0.62, 0.7] as const;
const MESH_AMBIENT = 0.22;
const MESH_SPECULAR = 0.32;
const MESH_SHININESS = 20;
const ARROW_SCALE = 0.0035;
const ARROW_HEAD_ANGLE_RAD = 0.45;

export interface ProjectionView {
  width: number;
  height: number;
  zoom: number;
  yaw: number;
  pitch: number;
  dpr: number;
}

export interface PreparedTriangle {
  placed: Triangle;
  depth: number;
  intensity: number;
}

interface PrepareMeshOptions {
  mesh: HeadMesh;
  rotation: number[][];
  shift: Vec3;
  offset: Vec3;
  yaw: number;
  pitch: number;
}

interface LineOptions {
  points: Vec3[];
  color: string;
  lineWidth: number;
  view: ProjectionView;
}

interface ArrowOptions {
  origin: Vec3;
  vector: Vec3;
  color: string;
  view: ProjectionView;
}

/** Return the translation that aligns a mesh's front extent to the face plane. */
export function computeMeshFaceShift(
  mesh: HeadMesh,
  comToFaceMm: number,
): Vec3 {
  let maximumX = -Infinity;
  for (const triangle of mesh.triangles) {
    for (const vertex of triangle) {
      if (vertex[0] > maximumX) maximumX = vertex[0];
    }
  }
  return [comToFaceMm / 1000 - maximumX, 0, 0];
}

/** Transform, depth-sort, and shade mesh triangles for painter rendering. */
export function prepareShadedTriangles({
  mesh,
  rotation,
  shift,
  offset,
  yaw,
  pitch,
}: PrepareMeshOptions): PreparedTriangle[] {
  const cameraForward: Vec3 = [
    Math.cos(pitch) * Math.cos(yaw),
    Math.sin(pitch),
    Math.cos(pitch) * Math.sin(yaw),
  ];
  const place = (vertex: Vec3): Vec3 =>
    add(apply(rotation, add(vertex, shift)), offset);

  return mesh.triangles
    .map((triangle, index): PreparedTriangle => {
      const placed = triangle.map(place) as Triangle;
      const centroid: Vec3 = [
        (placed[0][0] + placed[1][0] + placed[2][0]) / 3,
        (placed[0][1] + placed[1][1] + placed[2][1]) / 3,
        (placed[0][2] + placed[1][2] + placed[2][2]) / 3,
      ];
      const depth =
        centroid[0] * cameraForward[0] +
        centroid[1] * cameraForward[1] +
        centroid[2] * cameraForward[2];
      const normal = apply(rotation, mesh.normals[index]);
      const lambert = Math.abs(
        normal[0] * LIGHT_DIRECTION[0] +
          normal[1] * LIGHT_DIRECTION[1] +
          normal[2] * LIGHT_DIRECTION[2],
      );
      const diffuse = (1 - MESH_AMBIENT - MESH_SPECULAR) * lambert;
      const specular = MESH_SPECULAR * lambert ** MESH_SHININESS;
      return {
        placed,
        depth,
        intensity: MESH_AMBIENT + diffuse + specular,
      };
    })
    .sort((first, second) => first.depth - second.depth);
}

/** Paint the canvas backdrop at its device-pixel dimensions. */
export function drawCanvasBackdrop(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
): void {
  context.clearRect(0, 0, width, height);
  const backdrop = context.createRadialGradient(
    width / 2,
    height * 0.55,
    height * 0.1,
    width / 2,
    height * 0.55,
    height * 0.9,
  );
  backdrop.addColorStop(0, "rgba(30, 41, 59, 0.55)");
  backdrop.addColorStop(1, "rgba(2, 6, 23, 0)");
  context.fillStyle = backdrop;
  context.fillRect(0, 0, width, height);
}

/** Draw a polyline after applying the shared orthographic projection. */
export function drawProjectedLine(
  context: CanvasRenderingContext2D,
  { points, color, lineWidth, view }: LineOptions,
): void {
  context.strokeStyle = color;
  context.lineWidth = lineWidth * view.dpr;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  points.forEach((point, index) => {
    const [x, y] = project(
      point,
      view.width,
      view.height,
      view.zoom,
      view.yaw,
      view.pitch,
    );
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

/** Draw prepared mesh triangles with the established flat-shaded palette. */
export function drawShadedTriangles(
  context: CanvasRenderingContext2D,
  triangles: PreparedTriangle[],
  view: ProjectionView,
): void {
  for (const { placed, intensity } of triangles) {
    const rgb = MESH_BASE_RGB.map((component) =>
      Math.round(Math.min(1, component * intensity) * 255),
    );
    context.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    context.beginPath();
    placed.forEach((point, index) => {
      const [x, y] = project(
        point,
        view.width,
        view.height,
        view.zoom,
        view.yaw,
        view.pitch,
      );
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.closePath();
    context.fill();
  }
}

/** Draw a projected velocity arrow with a filled engineering-style head. */
export function drawVelocityArrow(
  context: CanvasRenderingContext2D,
  { origin, vector, color, view }: ArrowOptions,
): void {
  const tip: Vec3 = [
    origin[0] + vector[0] * ARROW_SCALE,
    origin[1] + vector[1] * ARROW_SCALE,
    origin[2] + vector[2] * ARROW_SCALE,
  ];
  const projectionArgs = [
    view.width,
    view.height,
    view.zoom,
    view.yaw,
    view.pitch,
  ] as const;
  const [originX, originY] = project(origin, ...projectionArgs);
  const [tipX, tipY] = project(tip, ...projectionArgs);
  const angle = Math.atan2(tipY - originY, tipX - originX);
  const headLength = 11 * view.dpr;
  const baseX = tipX - Math.cos(angle) * headLength * 0.7;
  const baseY = tipY - Math.sin(angle) * headLength * 0.7;

  context.strokeStyle = color;
  context.lineWidth = 2.5 * view.dpr;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(originX, originY);
  context.lineTo(baseX, baseY);
  context.stroke();
  context.fillStyle = color;
  context.beginPath();
  context.moveTo(tipX, tipY);
  context.lineTo(
    tipX - headLength * Math.cos(angle - ARROW_HEAD_ANGLE_RAD),
    tipY - headLength * Math.sin(angle - ARROW_HEAD_ANGLE_RAD),
  );
  context.lineTo(
    tipX - headLength * Math.cos(angle + ARROW_HEAD_ANGLE_RAD),
    tipY - headLength * Math.sin(angle + ARROW_HEAD_ANGLE_RAD),
  );
  context.closePath();
  context.fill();
}
