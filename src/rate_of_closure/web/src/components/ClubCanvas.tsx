/**
 * Animated 3D clubhead on a plain canvas — no WebGL dependency.
 *
 * Simple orthographic projection of the same wireframe the PyQt6 view
 * draws: face plate, body outline, shaft stub, impact point, and the
 * reference vs impact-point velocity arrows, spinning under the
 * scenario's angular velocity. Playback is user-controllable —
 * play/pause, 0.1x-3x speed, and Head Fixed vs Head Moving display
 * modes — matching the desktop app.
 *
 * Optional photorealistic mode: an STL file input (client-side
 * FileReader, nothing uploaded) swaps the procedural wireframe for the
 * user's clubhead mesh, normalized onto the same envelope and rendered
 * as flat-shaded painter's-algorithm triangles — depth-sorted by
 * distance along the camera's forward axis, shaded by |normal · light|
 * with the same fixed world light as the desktop app.
 */

import { useEffect, useRef, useState } from "react";

import { solve, type ImpactScenario } from "../model/impact";
import { loadHeadMesh, type HeadMesh } from "../model/mesh";
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
import { ClubCanvasPlaybackControls } from "./ClubCanvasPlaybackControls";
import { ClubCanvasViewport } from "./ClubCanvasViewport";
import {
  defaultCameraState,
  safeTrackingZoom,
  updateTrackingTarget,
  withCameraZoom,
  type CameraState,
} from "../model/cameraCommands";

const SPAN_MS = 8.0;
const STEPS = 48;

export const VIEW_MODES = [
  "Head Fixed in Place",
  "Head Moving Through Space",
] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

// H6 accent alignment (#4125): chart-palette accents come from the
// shared model/theme.ts palette; only the neutral body tone is local.
const COLORS = {
  face: getChartColor(0),
  body: "#8b949e",
  shaft: getChartColor(7),
  vRef: getChartColor(1),
  vPoint: getChartColor(3),
  impact: getChartColor(6),
  cog: getChartColor(2),
};

// STL-mesh shading constants — identical to the PyQt6 club view.
const LIGHT_LEN = Math.hypot(0.3, 0.8, 0.5);
const LIGHT_DIR: Vec3 = [0.3 / LIGHT_LEN, 0.8 / LIGHT_LEN, 0.5 / LIGHT_LEN];
const MESH_BASE_RGB = [0.56, 0.62, 0.7] as const;
const MESH_AMBIENT = 0.22;
const MESH_SPECULAR = 0.32;
const HEAD_CLEARANCE_RADIUS_M = 0.35;
const MOVING_BASE_HALF_EXTENT_M = 0.55;
const TRACKING_MAX_STEP_M = 0.04;



export function ClubCanvas({
  scenario,
  externalMesh = null,
  hoselPoint = null,
  cogPoint = null,
}: {
  scenario: ImpactScenario;
  /** A generated head (e.g. parametric club head) to render; the STL
   *  loader and the Procedural Head reset keep working alongside it. */
  externalMesh?: HeadMesh | null;
  /** Generated head's hosel — the shaft line attaches there (H1). */
  hoselPoint?: Vec3 | null;
  /** Generated head's divergence-theorem volumetric COG (H1). */
  cogPoint?: Vec3 | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const phaseRef = useRef(0);
  const [camera, setCamera] = useState(defaultCameraState);
  const cameraRef = useRef(camera);
  const subjectRef = useRef<Vec3>([0, 0, 0]);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [mode, setMode] = useState<ViewMode>(VIEW_MODES[1]);
  const [mesh, setMesh] = useState<HeadMesh | null>(null);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [showCg, setShowCg] = useState(true);

  const updateCamera = (transform: (current: CameraState) => CameraState) => {
    setCamera((current) => {
      const next = transform(current);
      cameraRef.current = next;
      return next;
    });
  };

  useEffect(() => {
    if (externalMesh) {
      setMesh(externalMesh);
      setMeshError(null);
    }
  }, [externalMesh]);

  const onStlChosen = (file: File | undefined) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setMesh(loadHeadMesh(reader.result as ArrayBuffer));
        setMeshError(null);
      } catch (err) {
        setMeshError(err instanceof Error ? err.message : String(err));
      }
    };
    reader.onerror = () => setMeshError("could not read the selected file");
    reader.readAsArrayBuffer(file);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const result = solve(scenario);
    const omega = result.omegaDps.map((c) => (c * Math.PI) / 180) as Vec3;
    const parts = headParts(scenario);
    const moving = mode === VIEW_MODES[1];
    const baseZoom = moving ? 0.9 : 1.6;
    const speedMps = scenario.clubheadSpeedMph * 0.44704;

    const draw = () => {
      // Render at device resolution so the canvas stays sharp on
      // high-DPI displays and at any layout width.
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const bw = Math.max(1, Math.round(rect.width * dpr));
      const bh = Math.max(1, Math.round(rect.width * 0.68 * dpr));
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
      const { width: w, height: h } = canvas;
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
      subjectRef.current = offset;
      let cameraState = updateTrackingTarget(
        cameraRef.current, offset, TRACKING_MAX_STEP_M,
      );
      if (cameraState.autoFitEnabled) {
        cameraState = withCameraZoom(
          cameraState,
          safeTrackingZoom(
            cameraState.zoom, HEAD_CLEARANCE_RADIUS_M, MOVING_BASE_HALF_EXTENT_M,
          ),
        );
      }
      cameraRef.current = cameraState;
      const place = (p: Vec3): Vec3 => add(apply(rot, p), offset);
      const zoom = baseZoom * cameraState.zoom;
      const yaw = cameraState.yawRad;
      const pitch = cameraState.pitchRad;
      const target = cameraState.targetM;

      const line = (pts: Vec3[], color: string, lw: number) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw * dpr;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        pts.forEach((p, i) => {
          const [px, py] = project(p, w, h, zoom, yaw, pitch, target);
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
      if (mesh) {
        let xMax = -Infinity;
        for (const tri of mesh.triangles) {
          for (const v of tri) if (v[0] > xMax) xMax = v[0];
        }
        shift = [scenario.comToFaceMm / 1000 - xMax, 0, 0];
      }
      if (mesh) {
        // Painter's algorithm: camera forward axis from the orbit
        // angles (same basis as project()); triangles sorted by
        // centroid depth along it, farthest drawn first.
        const fwd: Vec3 = [
          Math.cos(pitch) * Math.cos(yaw),
          Math.sin(pitch),
          Math.cos(pitch) * Math.sin(yaw),
        ];
        const shaded = mesh.triangles.map((tri, t) => {
          const placed = tri.map((v) => place(add(v, shift))) as [
            Vec3,
            Vec3,
            Vec3,
          ];
          const cx = (placed[0][0] + placed[1][0] + placed[2][0]) / 3;
          const cy = (placed[0][1] + placed[1][1] + placed[2][1]) / 3;
          const cz = (placed[0][2] + placed[1][2] + placed[2][2]) / 3;
          const depth = cx * fwd[0] + cy * fwd[1] + cz * fwd[2];
          const n = apply(rot, mesh.normals[t]);
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
            const [px, py] = project(p, w, h, zoom, yaw, pitch, target);
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
      const generated = mesh !== null && mesh === externalMesh;
      let hosel = parts.hosel;
      let shaftEnd = parts.shaftEnd;
      if (generated && hoselPoint) {
        hosel = add(hoselPoint, shift);
        const lie = (scenario.lieAngleDeg * Math.PI) / 180;
        shaftEnd = [
          hosel[0],
          hosel[1] + Math.sin(lie) * SHAFT_LEN,
          hosel[2] - Math.cos(lie) * SHAFT_LEN,
        ];
      }
      line([place(hosel), place(shaftEnd)], COLORS.shaft, 2.5);

      if (showCg) {
        // Volumetric COG marker (divergence theorem); wireframe and
        // non-watertight STLs fall back to the reference point, which
        // is the spec CG location.
        const cgModel: Vec3 =
          generated && cogPoint ? add(cogPoint, shift) : [0, 0, 0];
        const [cx, cy] = project(place(cgModel), w, h, zoom, yaw, pitch, target);
        const r = 5 * dpr;
        drawEngineeringCgSymbol(ctx, cx, cy, r, COLORS.cog);
        ctx.fillStyle = COLORS.cog;
        ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillText("CG", cx + 9 * dpr, cy - 8 * dpr);
      }

      const arrow = (origin: Vec3, vec: Vec3, color: string) => {
        const scale = 0.0035;
        const tip: Vec3 = [
          origin[0] + vec[0] * scale,
          origin[1] + vec[1] * scale,
          origin[2] + vec[2] * scale,
        ];
        const [ox, oy] = project(origin, w, h, zoom, yaw, pitch, target);
        const [tx, ty] = project(tip, w, h, zoom, yaw, pitch, target);
        const angle = Math.atan2(ty - oy, tx - ox);
        const headLen = 11 * dpr;
        // Stop the shaft short so the filled head forms a clean point.
        const bx = tx - Math.cos(angle) * headLen * 0.7;
        const by = ty - Math.sin(angle) * headLen * 0.7;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5 * dpr;
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

      const [ix, iy] = project(place(parts.impact), w, h, zoom, yaw, pitch, target);
      ctx.fillStyle = COLORS.impact;
      ctx.shadowColor = "rgba(255, 214, 10, 0.6)";
      ctx.shadowBlur = 8 * dpr;
      ctx.beginPath();
      ctx.arc(ix, iy, 5 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = "#94a3b8";
      ctx.font = `${12 * dpr}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillText(`t = ${(timeS * 1000).toFixed(1)} ms`, 12 * dpr, h - 12 * dpr);

      if (playing) {
        phaseRef.current = (phaseRef.current + speed / STEPS) % 1.0;
      }
    };

    const timer = window.setInterval(draw, 40);
    draw();
    return () => window.clearInterval(timer);
  }, [scenario, playing, speed, mode, mesh, externalMesh, hoselPoint, cogPoint, showCg]);

  return (
    <div className="space-y-2">
      <ClubCanvasPlaybackControls playing={playing} speed={speed} mode={mode}
        modes={VIEW_MODES} showCg={showCg} hasMesh={mesh !== null}
        meshError={meshError} fileInputRef={fileInputRef}
        onPlayingChange={setPlaying} onSpeedChange={setSpeed} onModeChange={setMode}
        onShowCgChange={setShowCg} onStlChosen={onStlChosen}
        onProceduralHead={() => { setMesh(null); setMeshError(null); }} />
      <ClubCanvasViewport camera={camera} canvasRef={canvasRef} dragRef={dragRef}
        subjectRef={subjectRef} clearanceRadiusM={HEAD_CLEARANCE_RADIUS_M}
        baseHalfExtentM={MOVING_BASE_HALF_EXTENT_M} updateCamera={updateCamera} />
    </div>
  );
}
