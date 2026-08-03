/**
 * Animated 3D clubhead on a plain canvas — no WebGL dependency.
 *
 * Simple orthographic projection of the same wireframe the PyQt6 view
 * draws: face plate, body outline, shaft stub, impact point, and the
 * reference vs impact-point velocity arrows, spinning under the
 * scenario's angular velocity.
 */

import { useEffect, useRef } from "react";

import { solve, type ImpactScenario } from "../model/impact";

type Vec3 = [number, number, number];

const SPAN_MS = 8.0;
const STEPS = 48;
const FACE_W = 0.058;
const FACE_H = 0.028;
const BODY_DEPTH = 0.11;
const SHAFT_LEN = 0.3;

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

/**
 * Orthographic projection with a fixed pleasant viewing angle.
 *
 * Model frame is the AffineDrift convention (x target, y up, z right);
 * the projection treats z as across, x as depth, and y as vertical.
 */
function project(v: Vec3, w: number, h: number): [number, number] {
  const yaw = -0.6;
  const pitch = 0.35;
  const across = v[2];
  const depth = v[0];
  const up = v[1];
  const x1 = across * Math.cos(yaw) - depth * Math.sin(yaw);
  const y1 = across * Math.sin(yaw) + depth * Math.cos(yaw);
  const z1 = up * Math.cos(pitch) - y1 * Math.sin(pitch);
  const scale = Math.min(w, h) * 1.6;
  return [w / 2 + x1 * scale, h * 0.62 - z1 * scale];
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
  const stepRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const result = solve(scenario);
    const omega = result.omegaDps.map(
      (c) => (c * Math.PI) / 180,
    ) as Vec3;
    const parts = headParts(scenario);

    const draw = () => {
      const { width: w, height: h } = canvas;
      ctx.clearRect(0, 0, w, h);
      const phase = stepRef.current / (STEPS - 1) - 0.5;
      const rot = rodrigues(omega, (phase * SPAN_MS) / 1000);

      const line = (pts: Vec3[], color: string, lw: number) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.beginPath();
        pts.forEach((p, i) => {
          const [px, py] = project(apply(rot, p), w, h);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      };

      line(parts.face, COLORS.face, 2.5);
      line(parts.back, COLORS.body, 1.2);
      parts.face.forEach((p, i) => line([p, parts.back[i]], COLORS.body, 0.8));
      line([parts.hosel, parts.shaftEnd], COLORS.shaft, 2.5);

      const arrow = (origin: Vec3, vec: Vec3, color: string) => {
        const scale = 0.0035;
        const tip: Vec3 = [
          origin[0] + vec[0] * scale,
          origin[1] + vec[1] * scale,
          origin[2] + vec[2] * scale,
        ];
        line([origin, tip], color, 2.5);
        const [tx, ty] = project(apply(rot, tip), w, h);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(tx, ty, 4, 0, Math.PI * 2);
        ctx.fill();
      };
      const vRefMps = scenario.clubheadSpeedMph * 0.44704;
      arrow([0, 0, 0], [vRefMps, 0, 0], COLORS.vRef);
      arrow(
        apply(rot, parts.impact),
        result.pointVelocityMps,
        COLORS.vPoint,
      );

      const [ix, iy] = project(apply(rot, parts.impact), w, h);
      ctx.fillStyle = COLORS.impact;
      ctx.beginPath();
      ctx.arc(ix, iy, 5, 0, Math.PI * 2);
      ctx.fill();

      stepRef.current = (stepRef.current + 1) % STEPS;
    };

    const timer = window.setInterval(draw, 40);
    draw();
    return () => window.clearInterval(timer);
  }, [scenario]);

  return (
    <canvas
      ref={canvasRef}
      width={560}
      height={420}
      className="w-full rounded-lg border border-slate-700 bg-slate-900"
      role="img"
      aria-label="Animated 3D clubhead rotating under the scenario's angular velocity"
    />
  );
}
