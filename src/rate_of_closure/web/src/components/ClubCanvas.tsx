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
import { FIELD_GUIDANCE } from "../model/units";
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
import {
  computeMeshFaceShift,
  drawCanvasBackdrop,
  drawProjectedLine,
  drawShadedTriangles,
  drawVelocityArrow,
  prepareShadedTriangles,
  type ProjectionView,
} from "./clubCanvasRendering";

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
  // Orbit camera state lives in refs so dragging never re-runs effects.
  // Defaults match the PyQt view (azimuth 150 deg, elevation 30 deg).
  const yawRef = useRef((150 * Math.PI) / 180);
  const pitchRef = useRef((30 * Math.PI) / 180);
  const zoomRef = useRef(1.0);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [mode, setMode] = useState<ViewMode>(VIEW_MODES[1]);
  const [mesh, setMesh] = useState<HeadMesh | null>(null);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [showCg, setShowCg] = useState(true);

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
      drawCanvasBackdrop(ctx, w, h);
      const phase = phaseRef.current - 0.5;
      const timeS = (phase * SPAN_MS) / 1000;
      const rot = rodrigues(omega, timeS);
      const offset: Vec3 = moving ? [speedMps * timeS, 0, 0] : [0, 0, 0];
      const place = (p: Vec3): Vec3 => add(apply(rot, p), offset);
      const zoom = baseZoom * zoomRef.current;
      const yaw = yawRef.current;
      const pitch = pitchRef.current;
      const view: ProjectionView = { width: w, height: h, zoom, yaw, pitch, dpr };
      const line = (points: Vec3[], color: string, lineWidth: number) =>
        drawProjectedLine(ctx, { points, color, lineWidth, view });

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
      const shift: Vec3 = mesh
        ? computeMeshFaceShift(mesh, scenario.comToFaceMm)
        : [0, 0, 0];
      if (mesh) {
        const shaded = prepareShadedTriangles({
          mesh,
          rotation: rot,
          shift,
          offset,
          yaw,
          pitch,
        });
        drawShadedTriangles(ctx, shaded, view);
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
        const [cx, cy] = project(place(cgModel), w, h, zoom, yaw, pitch);
        const r = 5 * dpr;
        drawEngineeringCgSymbol(ctx, cx, cy, r, COLORS.cog);
        ctx.fillStyle = COLORS.cog;
        ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillText("CG", cx + 9 * dpr, cy - 8 * dpr);
      }

      const vRefMps = scenario.clubheadSpeedMph * 0.44704;
      drawVelocityArrow(ctx, {
        origin: offset,
        vector: [vRefMps, 0, 0],
        color: COLORS.vRef,
        view,
      });
      drawVelocityArrow(ctx, {
        origin: place(parts.impact),
        vector: result.pointVelocityMps,
        color: COLORS.vPoint,
        view,
      });

      const [ix, iy] = project(place(parts.impact), w, h, zoom, yaw, pitch);
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
      <div
        aria-label="Playback controls"
        className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800/80 bg-slate-900/60 px-4 py-2.5 text-sm shadow-lg shadow-black/20 backdrop-blur"
      >
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          title="Play or pause the impact animation"
          className="w-16 rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <label className="flex items-center gap-2">
          <span className="text-slate-400">Playback Speed</span>
          <input
            type="range"
            min={0.1}
            max={3}
            step={0.1}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            aria-label="Playback speed multiplier"
          />
          <span className="w-8 text-slate-300">{speed.toFixed(1)}x</span>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-slate-400">Display</span>
          <select
            value={mode}
            title="Display mode: head fixed in place or moving through space"
            onChange={(e) => setMode(e.target.value as ViewMode)}
            className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none"
          >
            {VIEW_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <input
          ref={fileInputRef}
          type="file"
          accept=".stl"
          className="hidden"
          aria-hidden="true"
          tabIndex={-1}
          onChange={(e) => {
            onStlChosen(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          title="Render a user-supplied STL clubhead mesh in place of the procedural wireframe (read locally, never uploaded)."
          className="rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors hover:border-sky-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
        >
          Load Clubhead STL…
        </button>
        <label
          title={FIELD_GUIDANCE.showCgMarker}
          className="flex items-center gap-2 text-slate-300"
        >
          <input
            type="checkbox"
            checked={showCg}
            onChange={(e) => setShowCg(e.target.checked)}
            aria-label="Show CG"
          />
          Show CG
        </label>
        <button
          type="button"
          disabled={!mesh}
          onClick={() => {
            setMesh(null);
            setMeshError(null);
          }}
          title="Return to the default wireframe head."
          className="rounded-lg border border-slate-700 bg-slate-800/80 px-2 py-1 font-medium transition-colors enabled:hover:border-sky-400 disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
        >
          Procedural Head
        </button>
        {meshError && (
          <span role="alert" className="text-xs text-rose-400">
            STL load failed: {meshError}
          </span>
        )}
      </div>
      <canvas
        ref={canvasRef}
        width={840}
        height={571}
        className="w-full cursor-grab touch-none rounded-xl border border-slate-800/80 bg-slate-950/80 shadow-lg shadow-black/30 active:cursor-grabbing"
        role="img"
        aria-label="Animated 3D clubhead rotating under the scenario's angular velocity. Drag to orbit; scroll to zoom."
        onPointerDown={(e) => {
          dragRef.current = { x: e.clientX, y: e.clientY };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!dragRef.current) return;
          yawRef.current -= (e.clientX - dragRef.current.x) * 0.008;
          pitchRef.current = Math.max(
            -1.4,
            Math.min(1.4, pitchRef.current + (e.clientY - dragRef.current.y) * 0.008),
          );
          dragRef.current = { x: e.clientX, y: e.clientY };
        }}
        onPointerUp={(e) => {
          dragRef.current = null;
          e.currentTarget.releasePointerCapture(e.pointerId);
        }}
        onPointerLeave={() => {
          dragRef.current = null;
        }}
        onWheel={(e) => {
          zoomRef.current = Math.max(
            0.3,
            Math.min(4.0, zoomRef.current * (e.deltaY < 0 ? 1.1 : 1 / 1.1)),
          );
        }}
      />
      <p className="text-xs text-slate-500">
        Drag the view to orbit; scroll to zoom.
      </p>
    </div>
  );
}
