/**
 * Animated 3D clubhead on a plain canvas — no WebGL dependency.
 *
 * Simple orthographic projection of the same wireframe the PyQt6 view
 * draws: face plate, body outline, shaft stub, impact point, and the
 * reference vs impact-point velocity arrows, spinning under the
 * scenario's angular velocity. Playback is user-controllable —
 * play/pause, 0.1x-3x speed, and Head Fixed vs Head Moving display
 * modes — matching the desktop app.
 */

import { useEffect, useRef, useState } from "react";

import { solve, type ImpactScenario } from "../model/impact";

type Vec3 = [number, number, number];

const SPAN_MS = 8.0;
const STEPS = 48;
const FACE_W = 0.058;
const FACE_H = 0.028;
const BODY_DEPTH = 0.11;
const SHAFT_LEN = 0.3;

export const VIEW_MODES = [
  "Head Fixed in Place",
  "Head Moving Through Space",
] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

const COLORS = {
  face: "#0A84FF",
  body: "#8b949e",
  shaft: "#AC8E68",
  vRef: "#30D158",
  vPoint: "#FF375F",
  impact: "#FFD60A",
};

function rodrigues(omega: Vec3, dt: number): number[][] {
  const mag = Math.hypot(...omega);
  const theta = mag * dt;
  if (Math.abs(theta) < 1e-12) {
    return [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
  }
  const [x, y, z] = omega.map((c) => c / mag);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const t = 1 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
  ];
}

function apply(m: number[][], v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

/**
 * Orthographic projection under a user-controlled orbit camera.
 *
 * Model frame is the AffineDrift convention (x target, y up, z right).
 * The camera orbits the origin at `azimuth` (radians around the up
 * axis, measured from +x toward +z) and `elevation`. The defaults
 * (150 deg, 30 deg) match the PyQt view: behind the ball on the toe
 * side, so a right-handed club reads as right-handed.
 */
function project(
  v: Vec3,
  w: number,
  h: number,
  zoom: number,
  azimuth: number,
  elevation: number,
): [number, number] {
  const sinA = Math.sin(azimuth);
  const cosA = Math.cos(azimuth);
  const sinE = Math.sin(elevation);
  const cosE = Math.cos(elevation);
  const sx = v[0] * sinA - v[2] * cosA;
  const sy = -sinE * cosA * v[0] + cosE * v[1] - sinE * sinA * v[2];
  const scale = Math.min(w, h) * zoom;
  return [w / 2 + sx * scale, h * 0.62 - sy * scale];
}

function headParts(scenario: ImpactScenario) {
  const d = scenario.comToFaceMm / 1000;
  const lie = (scenario.lieAngleDeg * Math.PI) / 180;
  const face: Vec3[] = [
    [d, -FACE_H, -FACE_W],
    [d, -FACE_H, FACE_W],
    [d, FACE_H, FACE_W],
    [d, FACE_H, -FACE_W],
    [d, -FACE_H, -FACE_W],
  ];
  const back = face.map((p): Vec3 => [p[0] - BODY_DEPTH, p[1], p[2]]);
  const hosel: Vec3 = [d - 0.02, FACE_H, -FACE_W];
  const shaftEnd: Vec3 = [
    hosel[0],
    hosel[1] + Math.sin(lie) * SHAFT_LEN,
    hosel[2] - Math.cos(lie) * SHAFT_LEN,
  ];
  const impact: Vec3 = [
    d,
    scenario.impactOffsetHighMm / 1000,
    scenario.impactOffsetToeMm / 1000,
  ];
  return { face, back, hosel, shaftEnd, impact };
}

export function ClubCanvas({ scenario }: { scenario: ImpactScenario }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const phaseRef = useRef(0);
  // Orbit camera state lives in refs so dragging never re-runs effects.
  // Defaults match the PyQt view (azimuth 150 deg, elevation 30 deg).
  const yawRef = useRef((150 * Math.PI) / 180);
  const pitchRef = useRef((30 * Math.PI) / 180);
  const zoomRef = useRef(1.0);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [mode, setMode] = useState<ViewMode>(VIEW_MODES[0]);

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
      const place = (p: Vec3): Vec3 => add(apply(rot, p), offset);
      const zoom = baseZoom * zoomRef.current;
      const yaw = yawRef.current;
      const pitch = pitchRef.current;

      const line = (pts: Vec3[], color: string, lw: number) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw * dpr;
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

      line(parts.face.map(place), COLORS.face, 2.5);
      line(parts.back.map(place), COLORS.body, 1.2);
      parts.face.forEach((p, i) =>
        line([place(p), place(parts.back[i])], COLORS.body, 0.8),
      );
      line([place(parts.hosel), place(parts.shaftEnd)], COLORS.shaft, 2.5);

      const arrow = (origin: Vec3, vec: Vec3, color: string) => {
        const scale = 0.0035;
        const tip: Vec3 = [
          origin[0] + vec[0] * scale,
          origin[1] + vec[1] * scale,
          origin[2] + vec[2] * scale,
        ];
        line([origin, tip], color, 2.5);
        const [tx, ty] = project(tip, w, h, zoom, yaw, pitch);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(tx, ty, 4 * dpr, 0, Math.PI * 2);
        ctx.fill();
      };
      const vRefMps = scenario.clubheadSpeedMph * 0.44704;
      arrow(offset, [vRefMps, 0, 0], COLORS.vRef);
      arrow(place(parts.impact), result.pointVelocityMps, COLORS.vPoint);

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
  }, [scenario, playing, speed, mode]);

  return (
    <div className="space-y-2">
      <div
        aria-label="Playback controls"
        className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800/80 bg-slate-900/60 px-4 py-2.5 text-sm shadow-lg shadow-black/20 backdrop-blur"
      >
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
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
