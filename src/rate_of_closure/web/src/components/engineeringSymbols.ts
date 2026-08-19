/** Draw the quartered center-of-gravity target used on engineering drawings. */
export function drawEngineeringCgSymbol(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  color: string,
): void {
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1.5, radius * 0.22);
  for (const [start, end] of [[-Math.PI / 2, 0], [Math.PI / 2, Math.PI]] as const) {
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius * 0.78, start, end);
    ctx.closePath();
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
  ctx.moveTo(cx - radius * 1.35, cy);
  ctx.lineTo(cx + radius * 1.35, cy);
  ctx.moveTo(cx, cy - radius * 1.35);
  ctx.lineTo(cx, cy + radius * 1.35);
  ctx.stroke();
  ctx.restore();
}
