/**
 * Strike-zone canvas (epic #4120, V2 web parity).
 *
 * Face-scale (millimetre) view: driver face outline, impact-offset
 * marker, and the delivered path / face-normal / attack-angle vectors
 * projected into the face plane — the web twin of the PyQt6
 * StrikeView. Never shows swing or flight scale content.
 */

import { useEffect, useRef } from "react";

import { FIELD_GUIDANCE } from "../model/units";

interface Props {
  toeMm: number;
  highMm: number;
  loftDeg: number;
  /** Delivered club path [deg, + in-to-out]; undefined until a run. */
  pathDeg?: number;
  /** Delivered attack angle [deg, + up]; undefined until a run. */
  aoaDeg?: number;
}

/** Driver face half-extents [mm] (200 g reference envelope). */
const HALF_W_MM = 58;
const HALF_H_MM = 28;
const EXTENT_MM = Math.max(HALF_W_MM, HALF_H_MM) * 1.35;
const ARROW_MM = EXTENT_MM * 0.55;

export function StrikeCanvas({ toeMm, highMm, loftDeg, pathDeg, aoaDeg }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    const scale = Math.min(width, height) / (2 * EXTENT_MM);
    const px = (mm: number) => width / 2 + mm * scale;
    const py = (mm: number) => height / 2 - mm * scale;

    // Center cross-hairs.
    ctx.strokeStyle = "#334155";
    ctx.beginPath();
    ctx.moveTo(px(-EXTENT_MM), py(0));
    ctx.lineTo(px(EXTENT_MM), py(0));
    ctx.moveTo(px(0), py(-EXTENT_MM));
    ctx.lineTo(px(0), py(EXTENT_MM));
    ctx.stroke();

    // Superellipse face outline (exponent 2.5, like the desktop view).
    ctx.strokeStyle = "#f472b6";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= 120; i += 1) {
      const t = (i / 120) * 2 * Math.PI;
      const e = 2 / 2.5;
      const x = HALF_W_MM * Math.sign(Math.cos(t)) * Math.abs(Math.cos(t)) ** e;
      const y = HALF_H_MM * Math.sign(Math.sin(t)) * Math.abs(Math.sin(t)) ** e;
      if (i === 0) ctx.moveTo(px(x), py(y));
      else ctx.lineTo(px(x), py(y));
    }
    ctx.closePath();
    ctx.stroke();
    ctx.lineWidth = 1;

    const arrow = (dxMm: number, dyMm: number, color: string, label: string, row: number) => {
      const x0 = px(toeMm);
      const y0 = py(highMm);
      const x1 = px(toeMm + dxMm);
      const y1 = py(highMm + dyMm);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      const angle = Math.atan2(y1 - y0, x1 - x0);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - 7 * Math.cos(angle - 0.4), y1 - 7 * Math.sin(angle - 0.4));
      ctx.lineTo(x1 - 7 * Math.cos(angle + 0.4), y1 - 7 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.font = "11px sans-serif";
      ctx.fillText(label, 10, height - 12 - row * 15);
    };

    // Delivery vectors projected into the face plane (toe -> x, up -> y).
    if (pathDeg !== undefined && aoaDeg !== undefined) {
      arrow(
        Math.sin((pathDeg * Math.PI) / 180) * ARROW_MM,
        Math.sin((aoaDeg * Math.PI) / 180) * ARROW_MM,
        "#38bdf8",
        `club path ${pathDeg >= 0 ? "+" : ""}${pathDeg.toFixed(1)}° / AoA ${
          aoaDeg >= 0 ? "+" : ""
        }${aoaDeg.toFixed(1)}°`,
        1,
      );
    }
    arrow(
      0,
      Math.sin((loftDeg * Math.PI) / 180) * ARROW_MM,
      "#a78bfa",
      `face normal (loft ${loftDeg.toFixed(1)}°)`,
      0,
    );

    // Impact marker.
    ctx.fillStyle = "#facc15";
    ctx.beginPath();
    ctx.arc(px(toeMm), py(highMm), 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px sans-serif";
    ctx.fillText(
      `impact (${toeMm >= 0 ? "+" : ""}${toeMm.toFixed(1)}, ${
        highMm >= 0 ? "+" : ""
      }${highMm.toFixed(1)}) mm — face scale, ±${EXTENT_MM.toFixed(0)} mm`,
      10,
      16,
    );
  }, [toeMm, highMm, loftDeg, pathDeg, aoaDeg]);

  return (
    <canvas
      ref={canvasRef}
      width={860}
      height={480}
      title={FIELD_GUIDANCE.strikeVectorsVisible}
      className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
      aria-label="Strike zone (face plane, millimetres)"
    />
  );
}
