/** Swing-canvas annotations for the canonical wedge clearance payload. */

import type { Vec3 } from "../model/impactPhysics";
import type { WedgeGroundClearancePayloadTs } from "../model/wedgeGroundClearance";

type ScreenPoint = [number, number];

const sequenceColor = (sequence: WedgeGroundClearancePayloadTs["sequence"]) =>
  sequence === "ball_first" || sequence === "ball_only" ? "#34d399"
    : sequence === "ground_first" ? "#fb7185" : "#fbbf24";

function dot(ctx: CanvasRenderingContext2D, point: ScreenPoint, color: string, radius: number) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point[0], point[1], radius, 0, 2 * Math.PI);
  ctx.fill();
}

/** Draw the swept sole path plus auditable ball, ground, low-point, and live-clearance markers. */
export function drawWedgeGroundOverlay(
  ctx: CanvasRenderingContext2D,
  result: WedgeGroundClearancePayloadTs,
  timeS: number,
  project: (point: Vec3) => ScreenPoint,
): void {
  if (result.envelope.length === 0) return;
  const nearest = (target: number) => result.envelope.reduce((best, sample) =>
    Math.abs(sample.timeS - target) < Math.abs(best.timeS - target) ? sample : best);
  const current = nearest(timeS);
  const path = result.envelope.map((sample) => project(sample.worldPointM));
  ctx.strokeStyle = "rgba(52, 211, 153, 0.45)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  path.forEach(([x, y], index) => index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
  ctx.stroke();

  const currentPoint = project(current.worldPointM);
  const groundPoint = project([current.worldPointM[0], 0, current.worldPointM[2]]);
  ctx.strokeStyle = current.minimumClearanceM >= 0 ? "#22d3ee" : "#fb7185";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(currentPoint[0], currentPoint[1]);
  ctx.lineTo(groundPoint[0], groundPoint[1]);
  ctx.stroke();
  ctx.setLineDash([]);
  dot(ctx, currentPoint, ctx.strokeStyle as string, 3.5);

  if (result.ballContactTimeS !== null) {
    const ballPoint = project(nearest(result.ballContactTimeS).worldPointM);
    dot(ctx, ballPoint, "#22d3ee", 4.5);
    ctx.fillStyle = "#67e8f9";
    ctx.fillText("Ball Contact", ballPoint[0] + 7, ballPoint[1] - 7);
  }
  if (result.firstGroundContact !== null) {
    const contactPoint = project(result.firstGroundContact.worldPointM);
    dot(ctx, contactPoint, sequenceColor(result.sequence), 5);
    ctx.fillStyle = sequenceColor(result.sequence);
    ctx.fillText("Ground Contact", contactPoint[0] + 7, contactPoint[1] + 15);
  }
  const lowPoint = project(result.lowPoint.worldPointM);
  ctx.strokeStyle = "#fbbf24";
  ctx.strokeRect(lowPoint[0] - 3, lowPoint[1] - 3, 6, 6);
  ctx.fillStyle = "#6ee7b7";
  ctx.font = "600 12px sans-serif";
  ctx.fillText("Wedge Sole Envelope", 12, 70);
  ctx.fillStyle = current.minimumClearanceM >= 0 ? "#67e8f9" : "#fda4af";
  ctx.fillText(`${(current.minimumClearanceM * 1000).toFixed(1)} mm clearance`, 12, 86);
  ctx.lineWidth = 1;
}
