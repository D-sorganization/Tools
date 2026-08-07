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
import { type TargetRegionTs } from "../model/targets";
import { spatialTargetHalfExtents, type SpatialTargetTs } from "../model/spatialTarget";
import { formatDistanceM } from "../model/units";
import { withAlpha } from "../model/theme";
import { canvasContext, observeCanvas, type LogicalCanvasSize } from "./canvasDisplay";
import { spatialTargetSummary } from "./spatialTargetPresentation";

interface Props {
  /** App-frame trajectory (x downrange, y up, z right), tee-origin. */
  points: FlightPoint[];
  /** Optional common-input no-wind trajectory rendered as a dashed ghost. */
  comparisonPoints?: FlightPoint[];
  emptyText?: string;
  /** Course furniture layout (#4125 H7a); defaults to the driver hole. */
  layout?: CourseLayout;
  /** Render the fairway/green/flag course elements (default on). */
  showCourse?: boolean;
  /** Target region (#4125 H7b): dashed boundary in the top-down view. */
  target?: TargetRegionTs;
  /** Canonical 3D target rendered in both orthographic views. */
  spatialTarget?: SpatialTargetTs;
  /** Ball-flight distance display unit (#4125 H6): yards default. */
  distanceUnit?: string;
}

const MIN_CARRY_M = 10.0;
const MIN_HEIGHT_M = 5.0;
const MIN_LATERAL_M = 5.0;
const MARGIN = 34;
const SIDE_CANVAS_SIZE = { width: 860, height: 260 } as const;
const TOP_CANVAS_SIZE = { width: 860, height: 220 } as const;

function responsiveCanvasStyle(size: { width: number; height: number }) {
  return {
    width: "100%",
    height: "auto",
    aspectRatio: `${size.width} / ${size.height}`,
  };
}

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

function drawTarget(
  ctx: CanvasRenderingContext2D,
  target: TargetRegionTs,
  px: (x: number) => number,
  py: (v: number) => number,
): void {
  // Dashed target boundary (#4125 H7b), palette flag tone.
  ctx.strokeStyle = courseColors().flag;
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  if (target.kind === "green") {
    const { distanceM: d, radiusM: r, lateralM: z } = target;
    ctx.ellipse(px(d), py(z), px(d + r) - px(d), py(0) - py(r), 0, 0, 2 * Math.PI);
  } else {
    const { distanceM: d, bandHalfLengthM: b, halfWidthM: w } = target;
    ctx.rect(px(d - b), py(w), px(d + b) - px(d - b), py(-w) - py(w));
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 1;
}

function drawSpatialTarget(
  ctx: CanvasRenderingContext2D,
  target: SpatialTargetTs,
  vertical: "height" | "lateral",
  px: (value: number) => number,
  py: (value: number) => number,
  logicalWidth: number,
): void {
  const [downrange, elevation, right] = target.point.appCoordinatesM;
  const [halfDownrange, halfElevation, halfRight] = spatialTargetHalfExtents(target);
  const center = vertical === "height" ? elevation : right;
  const halfVertical = vertical === "height" ? halfElevation : halfRight;
  ctx.strokeStyle = "#f59e0b";
  ctx.fillStyle = withAlpha("#f59e0b", 0.14);
  ctx.setLineDash([5, 3]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  if (target.tolerance.kind === "sphere" || target.tolerance.kind === "surface_circle") {
    ctx.ellipse(
      px(downrange), py(center),
      Math.abs(px(downrange + halfDownrange) - px(downrange)),
      Math.max(2, Math.abs(py(center + halfVertical) - py(center))),
      0, 0, 2 * Math.PI,
    );
  } else {
    ctx.rect(
      px(downrange - halfDownrange), py(center + halfVertical),
      px(downrange + halfDownrange) - px(downrange - halfDownrange),
      Math.max(4, py(center - halfVertical) - py(center + halfVertical)),
    );
  }
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 1;
  ctx.fillStyle = "#fbbf24";
  ctx.font = "bold 11px sans-serif";
  const label = `ACTIVE · ${target.label}`;
  const anchorX = px(downrange);
  const proposedX = anchorX + 6;
  const wouldClipRight = proposedX + ctx.measureText(label).width > logicalWidth - 4;
  ctx.textAlign = wouldClipRight ? "right" : "left";
  ctx.fillText(label, wouldClipRight ? anchorX - 6 : proposedX, Math.max(14, py(center) - 7));
  ctx.textAlign = "left";
}

function drawPanel(
  canvas: HTMLCanvasElement,
  logicalSize: LogicalCanvasSize,
  points: FlightPoint[],
  comparisonPoints: FlightPoint[],
  vertical: "height" | "lateral",
  emptyText: string,
  layout: CourseLayout,
  showCourse: boolean,
  target?: TargetRegionTs,
  distanceUnit = "yd",
  spatialTarget?: SpatialTargetTs,
): void {
  const ctx = canvasContext(canvas, logicalSize);
  if (!ctx) return;
  const { width, height } = logicalSize;
  ctx.clearRect(0, 0, width, height);
  const allPoints = [...points, ...comparisonPoints];
  if (points.length < 2 && !spatialTarget) {
    ctx.fillStyle = "#64748b";
    ctx.font = "13px sans-serif";
    ctx.fillText(emptyText, 14, 24);
    return;
  }

  const spatialCenter = spatialTarget?.point.appCoordinatesM;
  const spatialExtents = spatialTarget ? spatialTargetHalfExtents(spatialTarget) : [0, 0, 0];
  const spatialCarry = spatialCenter ? spatialCenter[0] + spatialExtents[0] : 0;
  const carryExt = Math.max(MIN_CARRY_M, spatialCarry, ...allPoints.map((p) => p.position[0])) * 1.05;
  const value = (p: FlightPoint) =>
    vertical === "height" ? p.position[1] : p.position[2];
  const targetVertical = spatialCenter
    ? Math.abs(vertical === "height" ? spatialCenter[1] : spatialCenter[2]) +
      (vertical === "height" ? spatialExtents[1] : spatialExtents[2])
    : 0;
  const vertExt =
    vertical === "height"
      ? Math.max(MIN_HEIGHT_M, targetVertical, ...allPoints.map((p) => p.position[1])) * 1.2
      : Math.max(MIN_LATERAL_M, targetVertical, ...allPoints.map((p) => Math.abs(p.position[2]))) * 1.3;
  const zeroY = vertical === "height" ? height - MARGIN : height / 2;
  const usableY = vertical === "height" ? height - 2 * MARGIN : height / 2 - MARGIN;
  // A single metres-to-pixels scale prevents trajectory distortion.
  const physicalScale = Math.min((width - 2 * MARGIN) / carryExt, usableY / vertExt);
  const px = (x: number) => MARGIN + x * physicalScale;
  const py = (v: number) => zeroY - v * physicalScale;

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
  if (target && vertical === "lateral") drawTarget(ctx, target, px, py);
  if (spatialTarget) drawSpatialTarget(ctx, spatialTarget, vertical, px, py, width);

  if (points.length < 2) {
    ctx.fillStyle = "#64748b";
    ctx.font = "13px sans-serif";
    ctx.fillText(emptyText, 14, 24);
    return;
  }

  if (comparisonPoints.length >= 2) {
    ctx.strokeStyle = "#60a5fa";
    ctx.setLineDash([7, 5]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    comparisonPoints.forEach((point, index) => {
      const verticalValue = value(point);
      if (index === 0) ctx.moveTo(px(point.position[0]), py(verticalValue));
      else ctx.lineTo(px(point.position[0]), py(verticalValue));
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Selected-wind trajectory polyline.
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
  // Landing annotation follows the distance display unit (#4125 H6).
  const label =
    vertical === "height"
      ? `carry ${formatDistanceM(last.position[0], distanceUnit)}`
      : `lateral ${last.position[2] >= 0 ? "+" : "-"}${formatDistanceM(
          Math.abs(last.position[2]),
          distanceUnit,
        )}`;
  ctx.textAlign = "right";
  ctx.fillText(label, px(last.position[0]) - 8, py(value(last)) - 8);
  ctx.textAlign = "left";
  ctx.fillText(
    vertical === "height"
      ? `Side profile (height [m] vs carry [${distanceUnit}])`
      : `Top-down (right + vs carry [${distanceUnit}])`,
    10,
    16,
  );
  if (comparisonPoints.length >= 2) {
    ctx.fillStyle = "#60a5fa";
    ctx.fillText("- - No wind", width - 142, 16);
    ctx.fillStyle = "#34d399";
    ctx.fillText("— Selected wind", width - 76, 16);
  }
}

export function FlightCanvases({
  points,
  comparisonPoints = [],
  emptyText,
  layout,
  showCourse,
  target,
  spatialTarget,
  distanceUnit = "yd",
}: Props) {
  const sideRef = useRef<HTMLCanvasElement | null>(null);
  const topRef = useRef<HTMLCanvasElement | null>(null);
  const placeholder = emptyText ?? "Run a flight to populate the view.";
  const courseLayout = layout ?? DEFAULT_COURSE_LAYOUT;
  const course = showCourse ?? true;

  useEffect(() => {
    const drawSide = () => {
      if (!sideRef.current) return;
      drawPanel(
        sideRef.current,
        SIDE_CANVAS_SIZE,
        points,
        comparisonPoints,
        "height",
        placeholder,
        courseLayout,
        course,
        undefined,
        distanceUnit,
        spatialTarget,
      );
    };
    const drawTop = () => {
      if (!topRef.current) return;
      drawPanel(
        topRef.current,
        TOP_CANVAS_SIZE,
        points,
        comparisonPoints,
        "lateral",
        placeholder,
        courseLayout,
        course,
        target,
        distanceUnit,
        spatialTarget,
      );
    };
    const stopSide = observeCanvas(sideRef, drawSide);
    const stopTop = observeCanvas(topRef, drawTop);
    return () => {
      stopSide();
      stopTop();
    };
  }, [points, comparisonPoints, placeholder, courseLayout, course, target, spatialTarget, distanceUnit]);

  const targetDescription = spatialTarget
    ? ` Plot includes ${spatialTargetSummary(spatialTarget)}`
    : undefined;

  return (
    <div className="grid min-w-0 gap-3">
      <canvas
        ref={sideRef}
        width={SIDE_CANVAS_SIZE.width}
        height={SIDE_CANVAS_SIZE.height}
        style={responsiveCanvasStyle(SIDE_CANVAS_SIZE)}
        className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
        aria-label="Flight side profile (height vs carry)"
        aria-description={targetDescription}
      />
      <canvas
        ref={topRef}
        width={TOP_CANVAS_SIZE.width}
        height={TOP_CANVAS_SIZE.height}
        style={responsiveCanvasStyle(TOP_CANVAS_SIZE)}
        className="w-full min-w-0 rounded-lg border border-slate-800 bg-slate-950/60"
        aria-label="Flight top-down view (lateral vs carry)"
        aria-description={targetDescription}
      />
    </div>
  );
}
