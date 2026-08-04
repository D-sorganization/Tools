/**
 * Landing-dispersion canvas for the web Variation tab (#4120 V3):
 * top-down scatter (x = lateral, + right; y = carry) of every
 * successful run with the 2-sigma dispersion ellipse overlaid.
 */

import { useEffect, useRef } from "react";

import { type VariationDatasetTs } from "../model/variation";
import { dispersionEllipse } from "../model/variationAnalysis";

export function LandingCanvas({
  dataset,
}: {
  dataset: VariationDatasetTs;
}): JSX.Element {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const jc = dataset.outputNames.indexOf("carry_m");
    const jl = dataset.outputNames.indexOf("lateral_m");
    const points: Array<[number, number]> = [];
    dataset.outputs.forEach((row, i) => {
      if (dataset.success[i] && row[jc] !== null && row[jl] !== null) {
        points.push([row[jl] as number, row[jc] as number]); // x = lateral
      }
    });
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    if (points.length === 0) return;
    const ellipse = dispersionEllipse(dataset);
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const pad = 2.0;
    const reach = ellipse ? ellipse.semiMajorM : 0;
    const minX = Math.min(...xs, (ellipse?.centerLateralM ?? 0) - reach) - pad;
    const maxX = Math.max(...xs, (ellipse?.centerLateralM ?? 0) + reach) + pad;
    const minY = Math.min(...ys, (ellipse?.centerCarryM ?? 0) - reach) - pad;
    const maxY = Math.max(...ys, (ellipse?.centerCarryM ?? 0) + reach) + pad;
    const scale = Math.min(
      (width - 40) / (maxX - minX || 1),
      (height - 40) / (maxY - minY || 1),
    );
    const px = (x: number) => 20 + (x - minX) * scale;
    const py = (y: number) => height - 20 - (y - minY) * scale;

    ctx.fillStyle = "rgba(56, 189, 248, 0.65)";
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(px(x), py(y), 3, 0, 2 * Math.PI);
      ctx.fill();
    }
    if (ellipse) {
      ctx.strokeStyle = "#eb6a3c";
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      // Engine angle is CCW from the carry axis; canvas x = lateral.
      ctx.ellipse(
        px(ellipse.centerLateralM),
        py(ellipse.centerCarryM),
        ellipse.semiMajorM * scale,
        ellipse.semiMinorM * scale,
        -((90.0 - ellipse.angleDeg) * Math.PI) / 180.0,
        0,
        2 * Math.PI,
      );
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.fillText("lateral [m] (+ right) →", width / 2 - 50, height - 4);
    ctx.save();
    ctx.translate(12, height / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("carry [m] →", 0, 0);
    ctx.restore();
  }, [dataset]);
  return (
    <canvas
      ref={ref}
      width={560}
      height={420}
      className="w-full rounded-lg border border-slate-800 bg-slate-950/60"
      title="Landing positions of every successful run, viewed from above; the dashed ellipse is the 2-sigma dispersion fit."
    />
  );
}
