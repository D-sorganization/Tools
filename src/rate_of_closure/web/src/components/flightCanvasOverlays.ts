import {
  courseColors,
  type CourseLayout,
} from "../model/course";
import type { TargetRegionTs } from "../model/targets";
import {
  spatialTargetHalfExtents,
  type SpatialTargetTs,
} from "../model/spatialTarget";
import { withAlpha } from "../model/theme";

export function drawCourse(
  ctx: CanvasRenderingContext2D,
  vertical: "height" | "lateral",
  px: (x: number) => number,
  py: (v: number) => number,
  width: number,
  layout: CourseLayout,
): void {
  const course = courseColors();
  const { greenDistanceM: distance, greenRadiusM: radius, fairwayHalfWidthM } = layout;
  if (vertical === "lateral") {
    ctx.fillStyle = withAlpha(course.fairway, 0.4);
    ctx.fillRect(
      0,
      py(fairwayHalfWidthM),
      width,
      py(-fairwayHalfWidthM) - py(fairwayHalfWidthM),
    );
    if (px(distance - radius) <= width) {
      ctx.fillStyle = withAlpha(course.green, 0.6);
      ctx.beginPath();
      ctx.ellipse(
        px(distance),
        py(0),
        px(distance + radius) - px(distance),
        py(0) - py(radius),
        0,
        0,
        2 * Math.PI,
      );
      ctx.fill();
      ctx.fillStyle = course.hole;
      ctx.beginPath();
      ctx.arc(px(distance), py(0), 2.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = course.flag;
      ctx.beginPath();
      ctx.moveTo(px(distance) + 3, py(0) - 7);
      ctx.lineTo(px(distance) + 10, py(0) - 4);
      ctx.lineTo(px(distance) + 3, py(0) - 1);
      ctx.closePath();
      ctx.fill();
    }
  } else if (px(distance - radius) <= width) {
    ctx.fillStyle = withAlpha(course.green, 0.85);
    ctx.fillRect(
      px(distance - radius),
      py(0) - 2,
      px(distance + radius) - px(distance - radius),
      4,
    );
    ctx.strokeStyle = course.flag;
    ctx.beginPath();
    ctx.moveTo(px(distance), py(0));
    ctx.lineTo(px(distance), py(0) - 16);
    ctx.stroke();
    ctx.fillStyle = course.flag;
    ctx.beginPath();
    ctx.moveTo(px(distance), py(0) - 16);
    ctx.lineTo(px(distance) + 8, py(0) - 12.5);
    ctx.lineTo(px(distance), py(0) - 9);
    ctx.closePath();
    ctx.fill();
  }
  ctx.fillStyle = course.tee;
  ctx.fillRect(px(0) - 2, py(0) - 2, 4, 4);
}

export function drawTarget(
  ctx: CanvasRenderingContext2D,
  target: TargetRegionTs,
  px: (x: number) => number,
  py: (v: number) => number,
): void {
  ctx.strokeStyle = courseColors().flag;
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  if (target.kind === "green") {
    const { distanceM: distance, radiusM: radius, lateralM: lateral } = target;
    ctx.ellipse(
      px(distance),
      py(lateral),
      px(distance + radius) - px(distance),
      py(0) - py(radius),
      0,
      0,
      2 * Math.PI,
    );
  } else {
    const { distanceM: distance, bandHalfLengthM, halfWidthM } = target;
    ctx.rect(
      px(distance - bandHalfLengthM),
      py(halfWidthM),
      px(distance + bandHalfLengthM) - px(distance - bandHalfLengthM),
      py(-halfWidthM) - py(halfWidthM),
    );
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 1;
}

export function drawSpatialTarget(
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
  if (
    target.tolerance.kind === "sphere" ||
    target.tolerance.kind === "surface_circle"
  ) {
    ctx.ellipse(
      px(downrange),
      py(center),
      Math.abs(px(downrange + halfDownrange) - px(downrange)),
      Math.max(2, Math.abs(py(center + halfVertical) - py(center))),
      0,
      0,
      2 * Math.PI,
    );
  } else {
    ctx.rect(
      px(downrange - halfDownrange),
      py(center + halfVertical),
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
  const wouldClipRight =
    proposedX + ctx.measureText(label).width > logicalWidth - 4;
  ctx.textAlign = wouldClipRight ? "right" : "left";
  ctx.fillText(
    label,
    wouldClipRight ? anchorX - 6 : proposedX,
    Math.max(14, py(center) - 7),
  );
  ctx.textAlign = "left";
}
