/**
 * Flight-scale profile canvases (epic #4120, V2 web parity).
 *
 * Side profile (height vs carry) and top-down (lateral vs carry)
 * canvases for an app-frame trajectory, auto-scaled to the flight
 * regime with the landing point annotated — the web twin of the PyQt6
 * FlightView's 2D panels.
 */

import { useEffect, useRef } from "react";

import {
  courseColors,
  DEFAULT_COURSE_LAYOUT,
  type CourseLayout,
} from "../model/course";
import { type FlightPoint } from "../model/flight";
import { withAlpha } from "../model/theme";

interface Props {
  /** App-frame trajectory (x downrange, y up, z right), tee-origin. */
  points: FlightPoint[];
  emptyText?: string;
  /** Course furniture layout (#4125 H7a); defaults to the driver hole. */
  layout?: CourseLayout;
  /** Render the fairway/green/flag course elements (default on). */
  showCourse?: boolean;
}

const MIN_CARRY_M = 10.0;
const MIN_HEIGHT_M = 5.0;
const MIN_LATERAL_M = 5.0;
const MARGIN = 34;

function drawCourse(
  ctx: CanvasRenderingContext2D,
  vertical: "height" | "lateral",
  px: (x: number) => number,
  py: (v: number) => number,
  width: number,
  layout: CourseLayout,
): void {
  // Course styling (#4125 H7a) — palette-derived tones (model/course.ts).
  const course = courseColors();
  const { greenDistanceM: d, greenRadiusM: r, fairwayHalfWidthM: hw } = layout;
  if (vertical === "lateral") {
    // Fairway strip along the target line, green disc + hole/flag, tee.
    ctx.fillStyle = withAlpha(course.fairway, 0.4);
    ctx.fillRect(0, py(hw), width, py(-hw) - py(hw));
    if (px(d - r) <= width) {
      ctx.fillStyle = withAlpha(course.green, 0.6);
      ctx.beginPath();
      ctx.ellipse(px(d), py(0), px(d + r) - px(d), py(0) - py(r), 0, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = course.hole;
      ctx.beginPath();
      ctx.arc(px(d), py(0), 2.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = course.flag;
      ctx.beginPath();
      ctx.moveTo(px(d) + 3, py(0) - 7);
      ctx.lineTo(px(d) + 10, py(0) - 4);
      ctx.lineTo(px(d) + 3, py(0) - 1);
      ctx.closePath();
      ctx.fill();
    }
  } else if (px(d - r) <= width) {
    // Side profile: green band on the ground + flagstick at the hole.
    ctx.fillStyle = withAlpha(course.green, 0.85);
    ctx.fillRect(px(d - r), py(0) - 2, px(d + r) - px(d - r), 4);
    ctx.strokeStyle = course.flag;
    ctx.beginPath();
    ctx.moveTo(px(d), py(0));
    ctx.lineTo(px(d), py(0) - 16);
    ctx.stroke();
    ctx.fillStyle = course.flag;
    ctx.beginPath();
    ctx.moveTo(px(d), py(0) - 16);
    ctx.lineTo(px(d) + 8, py(0) - 12.5);
    ctx.lineTo(px(d), py(0) - 9);
    ctx.closePath();
    ctx.fill();
  }
  ctx.fillStyle = course.tee;
  ctx.fillRect(px(0) - 2, py(0) - 2, 4, 4);
}

function drawPanel(
  canvas: HTMLCanvasElement,
  points: FlightPoint[],
  vertical: "height" | "lateral",
  emptyText: string,
  layout: CourseLayout,
  showCourse: boolean,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (points.length < 2) {
    ctx.fillStyle = "#64748b";
    ctx.font = "13px sans-serif";
    ctx.fillText(emptyText, 14, 24);
    return;
  }

  const carryExt = Math.max(MIN_CARRY_M, ...points.map((p) => p.position[0])) * 1.05;
  const value = (p: FlightPoint) =>
    vertical === "height" ? p.position[1] : p.position[2];
  const vertExt =
    vertical === "height"
      ? Math.max(MIN_HEIGHT_M, ...points.map((p) => p.position[1])) * 1.2
      : Math.max(MIN_LATERAL_M, ...points.map((p) => Math.abs(p.position[2]))) * 1.3;
  const zeroY = vertical === "height" ? height - MARGIN : height / 2;
  const spanY = vertical === "height" ? height - 2 * MARGIN : height / 2 - MARGIN;
  const px = (x: number) => MARGIN + (x / carryExt) * (width - 2 * MARGIN);
  const py = (v: number) => zeroY - (v / vertExt) * spanY;

  // Course-styled ground (#4125 H7a): grass fill + ground/target line.
  const course = courseColors();
  if (vertical === "height") {
    ctx.fillStyle = withAlpha(course.rough, 0.35);
    ctx.fillRect(0, py(0), width, height - py(0));
  } else {
    ctx.fillStyle = withAlpha(course.rough, 0.25);
    ctx.fillRect(0, 0, width, height);
  }
  ctx.strokeStyle = course.fairway;
  ctx.beginPath();
  ctx.moveTo(0, py(0));
  ctx.lineTo(width, py(0));
  ctx.stroke();
  if (showCourse) drawCourse(ctx, vertical, px, py, width, layout);

  // Trajectory polyline.
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(px(p.position[0]), py(value(p)));
    else ctx.lineTo(px(p.position[0]), py(value(p)));
  });
  ctx.stroke();
  ctx.lineWidth = 1;

  // Landing annotation.
  const last = points[points.length - 1];
  ctx.fillStyle = "#facc15";
  ctx.beginPath();
  ctx.arc(px(last.position[0]), py(value(last)), 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  const label =
    vertical === "height"
      ? `carry ${last.position[0].toFixed(1)} m`
      : `lateral ${last.position[2] >= 0 ? "+" : ""}${last.position[2].toFixed(1)} m`;
  ctx.textAlign = "right";
  ctx.fillText(label, px(last.position[0]) - 8, py(value(last)) - 8);
  ctx.textAlign = "left";
  ctx.fillText(
    vertical === "height" ? "Side profile (height vs carry)" : "Top-down (right + vs carry)",
    10,
    16,
  );
}

export function FlightCanvases({ points, emptyText, layout, showCourse }: Props) {
  const sideRef = useRef<HTMLCanvasElement | null>(null);
  const topRef = useRef<HTMLCanvasElement | null>(null);
  const placeholder = emptyText ?? "Run a flight to populate the view.";
  const courseLayout = layout ?? DEFAULT_COURSE_LAYOUT;
  const course = showCourse ?? true;

  useEffect(() => {
    if (sideRef.current)
      drawPanel(sideRef.current, points, "height", placeholder, courseLayout, course);
    if (topRef.current)
      drawPanel(topRef.current, points, "lateral", placeholder, courseLayout, course);
  }, [points, placeholder, courseLayout, course]);

  return (
    <div className="grid min-w-0 gap-3">
      <canvas
        ref={sideRef}
        width={860}
        height={260}
        className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
        aria-label="Flight side profile (height vs carry)"
      />
      <canvas
        ref={topRef}
        width={860}
        height={220}
        className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
        aria-label="Flight top-down view (lateral vs carry)"
      />
    </div>
  );
}
