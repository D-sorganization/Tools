/** Deterministic vector export for the orbitable impact engineering still. */

import { project } from "./clubCanvasGeometry";
import type { ImpactSceneGeometryTs } from "./impactSceneGeometry";

interface Camera { yaw: number; pitch: number; zoom: number }

const escapeXml = (value: string): string => value
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

export function impactSceneSvg(
  geometry: ImpactSceneGeometryTs,
  camera: Camera,
  width = 1600,
  height = 920,
): string {
  const fills = geometry.fills.map((fill) => {
    const points = fill.points.map((point) =>
      project(point, width, height, camera.zoom, camera.yaw, camera.pitch)
        .map((value) => value.toFixed(2)).join(","),
    ).join(" ");
    return `<polygon points="${points}" fill="${fill.color}" fill-opacity="${fill.alpha}"><title>${escapeXml(fill.label)}</title></polygon>`;
  }).join("");
  const lines = geometry.lines.map((line) => {
    const points = line.points.map((point) =>
      project(point, width, height, camera.zoom, camera.yaw, camera.pitch)
        .map((value) => value.toFixed(2)).join(","),
    ).join(" ");
    const dash = line.dash ? ` stroke-dasharray="${line.dash.join(" ")}"` : "";
    const marker = line.arrow ? ' marker-end="url(#arrow)"' : "";
    return `<polyline points="${points}" fill="none" stroke="${line.color}" stroke-width="${line.width * 2}"${dash}${marker}><title>${escapeXml(line.label)}</title></polyline>`;
  }).join("");
  const [contactX, contactY] = project(
    geometry.contactPoint, width, height, camera.zoom, camera.yaw, camera.pitch,
  );
  const [ballX, ballY] = project(
    geometry.ballCenter, width, height, camera.zoom, camera.yaw, camera.pitch,
  );
  const ballRadius = Math.min(width, height) * camera.zoom * 0.02135;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <title>Exact Wedge Impact Engineering Scene</title>
  <desc>Locked-scale vector export in the app frame: x target, y up, z right.</desc>
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>
  <rect width="100%" height="100%" fill="#020617"/>
  ${fills}
  ${lines}
  <circle cx="${ballX.toFixed(2)}" cy="${ballY.toFixed(2)}" r="${ballRadius.toFixed(2)}" fill="#f8fafc" fill-opacity="0.16" stroke="#e2e8f0" stroke-width="3"><title>Golf Ball</title></circle>
  <circle cx="${contactX.toFixed(2)}" cy="${contactY.toFixed(2)}" r="12" fill="none" stroke="#fde047" stroke-width="4"><title>Declared Contact Point</title></circle>
</svg>`;
}
