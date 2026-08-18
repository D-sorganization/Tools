import { useEffect, type MutableRefObject, type RefObject } from "react";

import { type ImpactResult, type ImpactScenario } from "../model/impact";
import { type ClubCamera } from "../model/clubCamera";
import { type ClubMeshSource } from "../model/clubMeshSource";
import { getChartColor } from "../model/theme";
import {
  SHAFT_LEN,
  add,
  apply,
  headParts,
  project,
  rodrigues,
  type Vec3,
} from "./clubCanvasGeometry";
import { drawEngineeringCgSymbol } from "./engineeringSymbols";
import { type ViewMode } from "./ClubCanvas";

const SPAN_MS = 8.0;
const STEPS = 48;
const COLORS = {
  face: getChartColor(0),
  body: "#8b949e",
  shaft: getChartColor(7),
  vRef: getChartColor(1),
  vPoint: getChartColor(3),
  impact: getChartColor(6),
  cog: getChartColor(2),
};
const LIGHT_LEN = Math.hypot(0.3, 0.8, 0.5);
const LIGHT_DIR: Vec3 = [0.3 / LIGHT_LEN, 0.8 / LIGHT_LEN, 0.5 / LIGHT_LEN];
const MESH_BASE_RGB = [0.56, 0.62, 0.7] as const;
const MESH_AMBIENT = 0.22;
const MESH_SPECULAR = 0.32;

export function useClubCanvasRenderer({
  canvasRef,
  phaseRef,
  scenario,
  result,
  playing,
  speed,
  mode,
  source,
  camera,
  showCg,
  onRenderError,
}: {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  phaseRef: MutableRefObject<number>;
  scenario: ImpactScenario;
  result: ImpactResult;
  playing: boolean;
  speed: number;
  mode: ViewMode;
  source: ClubMeshSource;
  camera: ClubCamera;
  showCg: boolean;
  onRenderError: (message: string | null) => void;
}): void {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const staging = document.createElement("canvas");

    const draw = () => {
      const omega = result.omegaDps.map((c) => (c * Math.PI) / 180) as Vec3;
      const parts = headParts(scenario);
      const moving = mode === "Head Moving Through Space";
      const baseZoom = moving ? 0.9 : 1.6;
      const speedMps = scenario.clubheadSpeedMph * 0.44704;
      const rawDpr = window.devicePixelRatio || 1;
      const dpr = Number.isFinite(rawDpr)
        ? Math.max(1, Math.min(2.5, rawDpr))
        : 1;
      const rect = canvas.getBoundingClientRect();
      const cssWidth = Number.isFinite(rect.width)
        ? Math.max(1, Math.min(1600, rect.width))
        : 1;
      const desiredWidth = cssWidth * dpr;
      const desiredHeight = desiredWidth * 0.68;
      const pixelScale = Math.min(
        1,
        2048 / desiredWidth,
        2048 / desiredHeight,
        Math.sqrt(4_194_304 / (desiredWidth * desiredHeight)),
      );
      const bw = Math.max(1, Math.round(desiredWidth * pixelScale));
      const bh = Math.max(1, Math.round(desiredHeight * pixelScale));
      const renderScale = bw / cssWidth;
      if (staging.width !== bw) staging.width = bw;
      if (staging.height !== bh) staging.height = bh;
      const ctx = staging.getContext("2d");
      if (!ctx) throw new Error("club canvas rendering context is unavailable");
      const { width: w, height: h } = staging;
      ctx.clearRect(0, 0, w, h);
      const backdrop = ctx.createRadialGradient(
        w / 2,
        h * 0.55,
        h * 0.1,
        w / 2,
        h * 0.55,
        h * 0.9,
      );
      backdrop.addColorStop(0, "rgba(30, 41, 59, 0.55)");
      backdrop.addColorStop(1, "rgba(2, 6, 23, 0)");
      ctx.fillStyle = backdrop;
      ctx.fillRect(0, 0, w, h);
      const phase = phaseRef.current - 0.5;
      const timeS = (phase * SPAN_MS) / 1000;
      const rot = rodrigues(omega, timeS);
      const offset: Vec3 = moving ? [speedMps * timeS, 0, 0] : [0, 0, 0];
      const place = (p: Vec3): Vec3 => add(apply(rot, p), offset);
      const zoom = baseZoom * camera.zoom;
      const yaw = (camera.azimuthDeg * Math.PI) / 180;
      const pitch = (camera.elevationDeg * Math.PI) / 180;

      const line = (pts: Vec3[], color: string, lw: number) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw * renderScale;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        pts.forEach((p, i) => {
          const [px, py] = project(p, w, h, zoom, yaw, pitch);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      };

      if (moving) {
        ctx.setLineDash([4, 6]);
        line(
          [
            [-0.5, -0.05, 0],
            [0.5, -0.05, 0],
          ],
          COLORS.body,
          0.8,
        );
        ctx.setLineDash([]);
      }

      // Put the mesh's forward extent (its face plane) at com_to_face
      // — exactly HEAD_DEPTH_M/2 for a normalized STL; parametric
      // heads keep their mass-scaled, loft-tilted extent.
      let shift: Vec3 = [0, 0, 0];
      if (source.mesh) {
        let xMax = -Infinity;
        for (const tri of source.mesh.triangles) {
          for (const v of tri) if (v[0] > xMax) xMax = v[0];
        }
        shift = [scenario.comToFaceMm / 1000 - xMax, 0, 0];
      }
      if (source.mesh) {
        // Painter's algorithm: camera forward axis from the orbit
        // angles (same basis as project()); triangles sorted by
        // centroid depth along it, farthest drawn first.
        const fwd: Vec3 = [
          Math.cos(pitch) * Math.cos(yaw),
          Math.sin(pitch),
          Math.cos(pitch) * Math.sin(yaw),
        ];
        const shaded = source.mesh.triangles.map((tri, t) => {
          const placed = tri.map((v) => place(add(v, shift))) as [
            Vec3,
            Vec3,
            Vec3,
          ];
          const cx = (placed[0][0] + placed[1][0] + placed[2][0]) / 3;
          const cy = (placed[0][1] + placed[1][1] + placed[2][1]) / 3;
          const cz = (placed[0][2] + placed[1][2] + placed[2][2]) / 3;
          const depth = cx * fwd[0] + cy * fwd[1] + cz * fwd[2];
          const n = apply(rot, source.mesh!.normals[t]);
          const lambert = Math.abs(
            n[0] * LIGHT_DIR[0] + n[1] * LIGHT_DIR[1] + n[2] * LIGHT_DIR[2],
          );
          const diffuse = (1 - MESH_AMBIENT - MESH_SPECULAR) * lambert;
          const specular = MESH_SPECULAR * lambert ** 20;
          const intensity = MESH_AMBIENT + diffuse + specular;
          return { placed, depth, intensity };
        });
        shaded.sort((a, b) => a.depth - b.depth);
        for (const { placed, intensity } of shaded) {
          const rgb = MESH_BASE_RGB.map((c) =>
            Math.round(Math.min(1, c * intensity) * 255),
          );
          ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
          ctx.beginPath();
          placed.forEach((p, i) => {
            const [px, py] = project(p, w, h, zoom, yaw, pitch);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.closePath();
          ctx.fill();
        }
      } else {
        line(parts.face.map(place), COLORS.face, 2.5);
        line(parts.back.map(place), COLORS.body, 1.2);
        parts.face.forEach((p, i) =>
          line([place(p), place(parts.back[i])], COLORS.body, 0.8),
        );
      }
      // Hosel-true shaft (H1): a generated head attaches the shaft
      // line at its per-type hosel point, along the lie angle.
      const generated = source.kind === "generated";
      let hosel: Vec3 | null = source.kind === "imported" ? null : parts.hosel;
      let shaftEnd: Vec3 | null =
        source.kind === "imported" ? null : parts.shaftEnd;
      if (generated && source.hosel) {
        hosel = add(source.hosel, shift);
        const lie = (scenario.lieAngleDeg * Math.PI) / 180;
        shaftEnd = [
          hosel[0],
          hosel[1] + Math.sin(lie) * SHAFT_LEN,
          hosel[2] - Math.cos(lie) * SHAFT_LEN,
        ];
      }
      if (hosel && shaftEnd)
        line([place(hosel), place(shaftEnd)], COLORS.shaft, 2.5);

      if (showCg) {
        // Generated heads show their uniform-density geometric centroid;
        // procedural/imported views show only the scenario reference datum.
        const cgModel: Vec3 =
          generated && source.geometricCentroid
            ? add(source.geometricCentroid, shift)
            : [0, 0, 0];
        const [cx, cy] = project(place(cgModel), w, h, zoom, yaw, pitch);
        const r = 5 * renderScale;
        drawEngineeringCgSymbol(ctx, cx, cy, r, COLORS.cog);
        ctx.fillStyle = COLORS.cog;
        ctx.font = `${11 * renderScale}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillText(
          generated ? "geometric centroid" : "scenario reference",
          cx + 9 * renderScale,
          cy - 8 * renderScale,
        );
      }

      const arrow = (origin: Vec3, vec: Vec3, color: string) => {
        const scale = 0.0035;
        const tip: Vec3 = [
          origin[0] + vec[0] * scale,
          origin[1] + vec[1] * scale,
          origin[2] + vec[2] * scale,
        ];
        const [ox, oy] = project(origin, w, h, zoom, yaw, pitch);
        const [tx, ty] = project(tip, w, h, zoom, yaw, pitch);
        const angle = Math.atan2(ty - oy, tx - ox);
        const headLen = 11 * renderScale;
        // Stop the shaft short so the filled head forms a clean point.
        const bx = tx - Math.cos(angle) * headLen * 0.7;
        const by = ty - Math.sin(angle) * headLen * 0.7;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5 * renderScale;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(
          tx - headLen * Math.cos(angle - 0.45),
          ty - headLen * Math.sin(angle - 0.45),
        );
        ctx.lineTo(
          tx - headLen * Math.cos(angle + 0.45),
          ty - headLen * Math.sin(angle + 0.45),
        );
        ctx.closePath();
        ctx.fill();
      };
      const vRefMps = scenario.clubheadSpeedMph * 0.44704;
      arrow(offset, [vRefMps, 0, 0], COLORS.vRef);
      arrow(place(parts.impact), result.pointVelocityMps, COLORS.vPoint);

      const [ix, iy] = project(place(parts.impact), w, h, zoom, yaw, pitch);
      ctx.fillStyle = COLORS.impact;
      ctx.shadowColor = "rgba(255, 214, 10, 0.6)";
      ctx.shadowBlur = 8 * renderScale;
      ctx.beginPath();
      ctx.arc(ix, iy, 5 * renderScale, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = "#94a3b8";
      ctx.font = `${12 * renderScale}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillText(
        `t = ${(timeS * 1000).toFixed(1)} ms`,
        12 * renderScale,
        h - 12 * renderScale,
      );

      const visible = canvas.getContext("2d");
      if (!visible)
        throw new Error("club canvas rendering context is unavailable");
      if (canvas.width !== bw) canvas.width = bw;
      if (canvas.height !== bh) canvas.height = bh;
      visible.drawImage(staging, 0, 0);
    };

    const safeDraw = (advance = false): boolean => {
      try {
        draw();
        if (advance)
          phaseRef.current = (phaseRef.current + speed / STEPS) % 1.0;
        onRenderError(null);
        return true;
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        onRenderError(`Clubhead render failed: ${detail.slice(0, 512)}`);
        return false;
      }
    };
    const initialDrawSucceeded = safeDraw();
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => {
            safeDraw(false);
          });
    observer?.observe(canvas);
    let timer: number | undefined;
    if (playing && initialDrawSucceeded) {
      timer = window.setInterval(() => {
        if (!safeDraw(true) && timer !== undefined) window.clearInterval(timer);
      }, 40);
    }
    return () => {
      observer?.disconnect();
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [
    canvasRef,
    phaseRef,
    scenario,
    result,
    playing,
    speed,
    mode,
    source,
    camera,
    showCg,
    onRenderError,
  ]);
}
