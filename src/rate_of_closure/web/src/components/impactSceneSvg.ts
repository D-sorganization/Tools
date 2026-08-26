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
  // ⚡ Bolt Optimization: Replace chained array .map().join() with a single-pass loop
  // to eliminate intermediate array allocations and reduce GC pressure for complex SVG exports.
  let fills = "";
  for (let i = 0; i < geometry.fills.length; i++) {
    const fill = geometry.fills[i];
    let points = "";
    for (let j = 0; j < fill.points.length; j++) {
      const p = project(fill.points[j], width, height, camera.zoom, camera.yaw, camera.pitch);
      if (j > 0) points += " ";
      points += `${p[0].toFixed(2)},${p[1].toFixed(2)}`;
    }
    fills += `<polygon points="${points}" fill="${fill.color}" fill-opacity="${fill.alpha}"><title>${escapeXml(fill.label)}</title></polygon>`;
  }

  let lines = "";
  for (let i = 0; i < geometry.lines.length; i++) {
    const line = geometry.lines[i];
    let points = "";
    for (let j = 0; j < line.points.length; j++) {
      const p = project(line.points[j], width, height, camera.zoom, camera.yaw, camera.pitch);
      if (j > 0) points += " ";
      points += `${p[0].toFixed(2)},${p[1].toFixed(2)}`;
    }
    const dash = line.dash ? ` stroke-dasharray="${line.dash.join(" ")}"` : "";
    const marker = line.arrow ? ' marker-end="url(#arrow)"' : "";
    lines += `<polyline points="${points}" fill="none" stroke="${line.color}" stroke-width="${line.width * 2}"${dash}${marker}><title>${escapeXml(line.label)}</title></polyline>`;
  }
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
