/**
 * Swing-scale scene renderer for the web Simulation panel (#4120 V2).
 *
 * Extracted from SimulationPanel so the panel stays under the repo's
 * 500-LOC limit. Side-on orthographic projection (x right, y up); the
 * scene holds SWING scale unless `showFlight` opts into the flight
 * envelope past impact (scale separation — flight dwarfs the swing).
 */

import { BALL_POSITION, type SimulationRunTs } from "../model/simulation";

export interface SwingSceneOptions {
  time: number;
  showBall: boolean;
  showGround: boolean;
  /** Opt-in flight display; off keeps the scene at swing scale. */
  showFlight: boolean;
}

export function drawSwingScene(
  canvas: HTMLCanvasElement,
  run: SimulationRunTs | null,
  { time, showBall, showGround, showFlight }: SwingSceneOptions,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (!run) {
    ctx.fillStyle = "#64748b";
    ctx.font = "14px sans-serif";
    ctx.fillText("Run a simulation to populate the scene.", 16, 28);
    return;
  }

  const swingEnd = run.swing[run.swing.length - 1].t;
  // Scale separation (#4120): the scene stays at swing scale unless
  // the opt-in 'Show Ball Flight' toggle expands it past impact.
  const inFlight = time > run.impactTimeS && showFlight;
  const extentX = inFlight
    ? Math.max(10, ...run.flight.map((p) => Math.abs(p.position[0]))) * 1.05
    : Math.max(1.5, ...run.swing.map((p) => Math.abs(p.position[0]))) * 1.15;
  const extentY = inFlight
    ? Math.max(5, ...run.flight.map((p) => p.position[1])) * 1.3
    : Math.max(1.5, ...run.swing.map((p) => Math.abs(p.position[1]))) * 1.15;
  const originX = inFlight ? 30 : width / 2;
  const scaleX = (width - 60) / (inFlight ? extentX : 2 * extentX);
  const scaleY = (height - 40) / (inFlight ? extentY : 2 * extentY);
  const s = Math.min(scaleX, scaleY);
  const groundY = inFlight ? height - 24 : height / 2 + extentY * s * 0.5;
  const px = (x: number) => originX + x * s;
  const py = (y: number) => groundY - y * s;

  if (showGround) {
    ctx.strokeStyle = "#475569";
    ctx.beginPath();
    ctx.moveTo(0, py(0));
    ctx.lineTo(width, py(0));
    ctx.stroke();
  }
  if (showBall) {
    ctx.fillStyle = "#facc15";
    ctx.beginPath();
    ctx.arc(px(BALL_POSITION[0]), py(BALL_POSITION[1]), 4, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Swing path (faint full arc + traversed portion + head marker).
  const drawPath = (
    points: Array<{ position: [number, number, number] }>,
    color: string,
    widthPx: number,
  ) => {
    if (points.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = widthPx;
    ctx.beginPath();
    points.forEach((point, index) => {
      const cx = px(point.position[0]);
      const cy = py(point.position[1]);
      if (index === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
    ctx.lineWidth = 1;
  };
  drawPath(run.swing, "rgba(56,189,248,0.25)", 1);
  const swingIndex = Math.min(
    run.swing.length - 1,
    Math.round((Math.min(time, swingEnd) / swingEnd) * (run.swing.length - 1)),
  );
  drawPath(run.swing.slice(0, swingIndex + 1), "#38bdf8", 2);
  const head = run.swing[swingIndex].position;
  ctx.fillStyle = "#f472b6";
  ctx.beginPath();
  ctx.arc(px(head[0]), py(head[1]), 4, 0, 2 * Math.PI);
  ctx.fill();

  // Flight trajectory polyline: opt-in only (scale separation).
  if (showFlight) drawPath(run.flight, "rgba(52,211,153,0.25)", 1);
  if (inFlight) {
    const flightT = time - run.impactTimeS;
    const upto = run.flight.filter((p) => p.time <= flightT);
    drawPath(upto, "#34d399", 2);
    if (upto.length) {
      const ball = upto[upto.length - 1].position;
      ctx.fillStyle = "#facc15";
      ctx.beginPath();
      ctx.arc(px(ball[0]), py(ball[1]), 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px sans-serif";
  ctx.fillText(
    `t = ${time.toFixed(3)} s (${inFlight ? "flight" : "swing"}) — impact at ${run.impactTimeS.toFixed(3)} s`,
    12,
    16,
  );
}
