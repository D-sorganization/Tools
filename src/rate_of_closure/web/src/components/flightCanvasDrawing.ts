import { courseColors, type CourseLayout } from "../model/course";
import type { FlightPoint } from "../model/flight";
import type { TargetRegionTs } from "../model/targets";
import {
  spatialTargetHalfExtents,
  type SpatialTargetTs,
} from "../model/spatialTarget";
import { formatDistanceM } from "../model/units";
import { withAlpha } from "../model/theme";
import { canvasContext, type LogicalCanvasSize } from "./canvasDisplay";
import { drawCourse, drawSpatialTarget, drawTarget } from "./flightCanvasOverlays";

const MIN_CARRY_M = 10;
const MIN_HEIGHT_M = 5;
const MIN_LATERAL_M = 5;
const MARGIN = 34;

export function drawFlightPanel(
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
  selectedRawIndex?: number,
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
  const spatialExtents = spatialTarget
    ? spatialTargetHalfExtents(spatialTarget)
    : [0, 0, 0];
  const spatialCarry = spatialCenter ? spatialCenter[0] + spatialExtents[0] : 0;
  const carryExt =
    Math.max(
      MIN_CARRY_M,
      spatialCarry,
      ...allPoints.map((point) => point.position[0]),
    ) * 1.05;
  const value = (point: FlightPoint) =>
    vertical === "height" ? point.position[1] : point.position[2];
  const targetVertical = spatialCenter
    ? Math.abs(vertical === "height" ? spatialCenter[1] : spatialCenter[2]) +
      (vertical === "height" ? spatialExtents[1] : spatialExtents[2])
    : 0;
  const vertExt =
    vertical === "height"
      ? Math.max(
          MIN_HEIGHT_M,
          targetVertical,
          ...allPoints.map((point) => point.position[1]),
        ) * 1.2
      : Math.max(
          MIN_LATERAL_M,
          targetVertical,
          ...allPoints.map((point) => Math.abs(point.position[2])),
        ) * 1.3;
  const zeroY = vertical === "height" ? height - MARGIN : height / 2;
  const usableY =
    vertical === "height" ? height - 2 * MARGIN : height / 2 - MARGIN;
  const physicalScale = Math.min(
    (width - 2 * MARGIN) / carryExt,
    usableY / vertExt,
  );
  const px = (x: number) => MARGIN + x * physicalScale;
  const py = (verticalValue: number) => zeroY - verticalValue * physicalScale;

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
  if (spatialTarget) {
    drawSpatialTarget(ctx, spatialTarget, vertical, px, py, width);
  }

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

  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(px(point.position[0]), py(value(point)));
    else ctx.lineTo(px(point.position[0]), py(value(point)));
  });
  ctx.stroke();
  ctx.lineWidth = 1;
  if (selectedRawIndex !== undefined && points[selectedRawIndex]) {
    const selected = points[selectedRawIndex];
    ctx.fillStyle = "#f472b6";
    ctx.beginPath();
    ctx.arc(px(selected.position[0]), py(value(selected)), 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  const last = points[points.length - 1];
  ctx.fillStyle = "#facc15";
  ctx.beginPath();
  ctx.arc(px(last.position[0]), py(value(last)), 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
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
